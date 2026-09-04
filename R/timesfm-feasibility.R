# Portions derived from TimesFM, Copyright 2025 Google LLC.
# Translated and modified for R/torch by the zukzeit authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Native R torch feasibility spike for the TimesFM 2.5 operator graph.
#
# This is deliberately smaller than the production architecture port. It
# exercises the exact hard operator path against a pinned upstream reference:
# RMS norms, fused QKV, rotary embeddings, Q/K normalization, per-dimension
# query scaling, unscaled causal attention, a SiLU feed-forward block, and the
# continuous quantile-head layout. Stage 3 will reuse the proven operations in
# the full checkpoint-backed module.

timesfm_spike_rms_norm <- torch::nn_module(
  "timesfm_spike_rms_norm",
  initialize = function(num_features, epsilon = 1e-6) {
    self$scale <- torch::nn_parameter(torch::torch_zeros(num_features))
    self$epsilon <- epsilon
  },
  forward = function(inputs) {
    variance <- torch::torch_mean(
      torch::torch_square(inputs),
      dim = -1,
      keepdim = TRUE
    )
    inputs * torch::torch_rsqrt(variance + self$epsilon) * self$scale
  }
)

timesfm_spike_rope <- torch::nn_module(
  "timesfm_spike_rope",
  initialize = function(embedding_dims,
                        min_timescale = 1,
                        max_timescale = 10000) {
    if (embedding_dims %% 2L != 0L) {
      zuk_abort_contract(
        "RoPE requires an even head dimension; got {embedding_dims}.",
        architecture = "timesfm",
        contract = "rotary embedding shape",
        expected = "even head dimension",
        actual = embedding_dims
      )
    }
    self$embedding_dims <- as.integer(embedding_dims)
    self$min_timescale <- min_timescale
    self$max_timescale <- max_timescale
  },
  forward = function(inputs, position) {
    if (inputs$shape[[length(inputs$shape)]] != self$embedding_dims) {
      zuk_abort_contract(
        "RoPE input and configured embedding dimensions differ.",
        architecture = "timesfm",
        contract = "rotary embedding shape",
        expected = self$embedding_dims,
        actual = inputs$shape[[length(inputs$shape)]]
      )
    }
    half <- self$embedding_dims %/% 2L
    fraction <- 2 * torch::torch_arange(
      0,
      half - 1L,
      dtype = torch::torch_float32(),
      device = inputs$device
    ) / self$embedding_dims
    timescale <- self$min_timescale * torch::torch_exp(
      log(self$max_timescale / self$min_timescale) * fraction
    )
    timescale <- timescale$to(dtype = inputs$dtype)$view(c(1, 1, 1, half))
    position <- position$to(dtype = inputs$dtype)$unsqueeze(-1)$unsqueeze(-1)
    angle <- position / timescale
    sin_angle <- torch::torch_sin(angle)
    cos_angle <- torch::torch_cos(angle)
    halves <- torch::torch_chunk(inputs, 2L, dim = -1)
    torch::torch_cat(
      list(
        halves[[1]] * cos_angle - halves[[2]] * sin_angle,
        halves[[2]] * cos_angle + halves[[1]] * sin_angle
      ),
      dim = -1
    )
  }
)

timesfm_spike_attention <- torch::nn_module(
  "timesfm_spike_attention",
  initialize = function(model_dim, num_heads, epsilon = 1e-6) {
    if (model_dim %% num_heads != 0L) {
      zuk_abort_contract(
        "{model_dim} model dimensions cannot be split over {num_heads} heads.",
        architecture = "timesfm",
        contract = "attention shape",
        expected = "model dimension divisible by number of heads",
        actual = c(model_dim = model_dim, num_heads = num_heads)
      )
    }
    self$model_dim <- as.integer(model_dim)
    self$num_heads <- as.integer(num_heads)
    self$head_dim <- as.integer(model_dim %/% num_heads)
    self$qkv_proj <- torch::nn_linear(model_dim, 3L * model_dim, bias = FALSE)
    self$out <- torch::nn_linear(model_dim, model_dim, bias = FALSE)
    self$query_ln <- timesfm_spike_rms_norm(self$head_dim, epsilon)
    self$key_ln <- timesfm_spike_rms_norm(self$head_dim, epsilon)
    self$per_dim_scale <- torch::nn_parameter(torch::torch_zeros(self$head_dim))
    self$rope <- timesfm_spike_rope(self$head_dim)
  },
  forward = function(inputs, patch_mask = NULL) {
    batch <- inputs$shape[[1]]
    n_patches <- inputs$shape[[2]]
    if (is.null(patch_mask)) {
      patch_mask <- torch::torch_zeros(
        c(batch, n_patches),
        dtype = torch::torch_bool(),
        device = inputs$device
      )
    }

    qkv <- torch::torch_chunk(self$qkv_proj(inputs), 3L, dim = -1)
    query <- qkv[[1]]$reshape(c(batch, n_patches, self$num_heads, self$head_dim))
    key <- qkv[[2]]$reshape(c(batch, n_patches, self$num_heads, self$head_dim))
    value <- qkv[[3]]$reshape(c(batch, n_patches, self$num_heads, self$head_dim))

    num_masked <- patch_mask$to(dtype = torch::torch_int32())$sum(dim = -1)
    position <- torch::torch_arange(
      0,
      n_patches - 1L,
      device = inputs$device
    )$unsqueeze(1) - num_masked$unsqueeze(-1)
    query <- self$rope(query, position)
    key <- self$rope(key, position)
    query <- self$query_ln(query)
    key <- self$key_ln(key)

    scale_factor <- 1.442695041 / sqrt(self$head_dim) *
      torch::nnf_softplus(self$per_dim_scale)
    query <- query * scale_factor

    query <- query$permute(c(1, 3, 2, 4))
    key <- key$permute(c(1, 3, 2, 4))
    value <- value$permute(c(1, 3, 2, 4))
    scores <- torch::torch_matmul(query, key$transpose(3, 4))

    query_index <- torch::torch_arange(
      0,
      n_patches - 1L,
      device = inputs$device
    )$view(c(1, 1, n_patches, 1))
    key_index <- torch::torch_arange(
      0,
      n_patches - 1L,
      device = inputs$device
    )$view(c(1, 1, 1, n_patches))
    causal <- query_index >= key_index
    observed <- key_index >= num_masked$view(c(batch, 1, 1, 1))
    mask <- torch::torch_logical_and(causal, observed)
    blocked <- torch::torch_full_like(
      scores,
      -torch::torch_finfo(scores$dtype)$max / 2
    )
    weights <- torch::nnf_softmax(torch::torch_where(mask, scores, blocked), dim = -1)
    attended <- torch::torch_matmul(weights, value)$permute(c(1, 3, 2, 4))
    self$out(attended$reshape(c(batch, n_patches, self$model_dim)))
  }
)

timesfm_spike_transformer <- torch::nn_module(
  "timesfm_spike_transformer",
  initialize = function(model_dim, num_heads, hidden_dim = model_dim,
                        epsilon = 1e-6) {
    self$pre_attn_ln <- timesfm_spike_rms_norm(model_dim, epsilon)
    self$post_attn_ln <- timesfm_spike_rms_norm(model_dim, epsilon)
    self$attn <- timesfm_spike_attention(model_dim, num_heads, epsilon)
    self$pre_ff_ln <- timesfm_spike_rms_norm(model_dim, epsilon)
    self$post_ff_ln <- timesfm_spike_rms_norm(model_dim, epsilon)
    self$ff0 <- torch::nn_linear(model_dim, hidden_dim, bias = FALSE)
    self$ff1 <- torch::nn_linear(hidden_dim, model_dim, bias = FALSE)
  },
  forward = function(inputs, patch_mask) {
    attended <- self$attn(self$pre_attn_ln(inputs), patch_mask)
    attended <- self$post_attn_ln(attended) + inputs
    feed_forward <- self$ff1(torch::nnf_silu(self$ff0(self$pre_ff_ln(attended))))
    self$post_ff_ln(feed_forward) + attended
  }
)

timesfm_spike_residual_head <- torch::nn_module(
  "timesfm_spike_residual_head",
  initialize = function(input_dim, hidden_dim, output_dim) {
    self$hidden_layer <- torch::nn_linear(input_dim, hidden_dim, bias = FALSE)
    self$output_layer <- torch::nn_linear(hidden_dim, output_dim, bias = FALSE)
    self$residual_layer <- torch::nn_linear(input_dim, output_dim, bias = FALSE)
  },
  forward = function(inputs) {
    self$output_layer(torch::nnf_silu(self$hidden_layer(inputs))) +
      self$residual_layer(inputs)
  }
)

timesfm_spike_module <- torch::nn_module(
  "timesfm_spike_module",
  initialize = function(model_dim = 4L, num_heads = 2L,
                        hidden_dim = model_dim,
                        quantile_horizon = 3L, n_outputs = 3L) {
    self$transformer <- timesfm_spike_transformer(
      model_dim, num_heads, hidden_dim
    )
    self$quantile_head <- timesfm_spike_residual_head(
      model_dim, hidden_dim, quantile_horizon * n_outputs
    )
    self$quantile_horizon <- as.integer(quantile_horizon)
    self$n_outputs <- as.integer(n_outputs)
  },
  forward = function(inputs, patch_mask) {
    embeddings <- self$transformer(inputs, patch_mask)
    shape <- embeddings$shape
    quantiles <- self$quantile_head(embeddings)$reshape(
      c(shape[[1]], shape[[2]], self$quantile_horizon, self$n_outputs)
    )
    list(embeddings = embeddings, quantiles = quantiles)
  }
)

# Copy the upstream [3 * model_dim, model_dim] fused tensor into the R linear
# layer without transposing. PyTorch and R torch use the same [out, in] layout.
timesfm_spike_load_fused_qkv <- function(module, weight) {
  target <- module$transformer$attn$qkv_proj$weight
  if (!inherits(weight, "torch_tensor") || !identical(weight$shape, target$shape)) {
    zuk_abort_checkpoint(
      "Fused QKV must have shape [{paste(target$shape, collapse = ', ')}].",
      model_id = "google/timesfm-2.5-200m-pytorch",
      tensor = "qkv_proj.weight",
      expected = target$shape,
      actual = if (inherits(weight, "torch_tensor")) weight$shape else class(weight)
    )
  }
  torch::with_no_grad({
    target$copy_(weight$to(dtype = target$dtype, device = target$device))
  })
  invisible(module)
}

timesfm_spike_tensor <- function(shape, offset, scale = 0.05, base = 0) {
  n <- prod(shape)
  values <- torch::torch_arange(0, n - 1L, dtype = torch::torch_float32())
  (base + scale * torch::torch_sin(values + offset))$reshape(shape)
}

# Deterministic state shared with the pinned Python reference generator.
timesfm_spike_fill_reference_weights <- function(module) {
  norm <- function(parameter, offset) {
    parameter$copy_(timesfm_spike_tensor(parameter$shape, offset, 0.02, 1))
  }
  matrix <- function(parameter, offset) {
    parameter$copy_(timesfm_spike_tensor(parameter$shape, offset, 0.05, 0))
  }
  torch::with_no_grad({
    norm(module$transformer$pre_attn_ln$scale, 1)
    norm(module$transformer$post_attn_ln$scale, 2)
    timesfm_spike_load_fused_qkv(
      module,
      timesfm_spike_tensor(module$transformer$attn$qkv_proj$weight$shape, 3)
    )
    norm(module$transformer$attn$query_ln$scale, 4)
    norm(module$transformer$attn$key_ln$scale, 5)
    module$transformer$attn$per_dim_scale$copy_(
      timesfm_spike_tensor(module$transformer$attn$per_dim_scale$shape, 6, 0.1)
    )
    matrix(module$transformer$attn$out$weight, 7)
    norm(module$transformer$pre_ff_ln$scale, 8)
    matrix(module$transformer$ff0$weight, 9)
    matrix(module$transformer$ff1$weight, 10)
    norm(module$transformer$post_ff_ln$scale, 11)
    matrix(module$quantile_head$hidden_layer$weight, 12)
    matrix(module$quantile_head$output_layer$weight, 13)
    matrix(module$quantile_head$residual_layer$weight, 14)
  })
  invisible(module)
}
