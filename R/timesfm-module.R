# Portions derived from TimesFM, Copyright 2025 Google LLC.
# Translated and modified for R/torch by the zukzeit authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Native R torch layers for the pinned TimesFM 2.5 200M checkpoint.

timesfm_meta_parameter <- function(shape) {
  torch::nn_parameter(torch::torch_empty(shape, device = torch::torch_device("meta")))
}

timesfm_linear <- torch::nn_module(
  "timesfm_linear",
  initialize = function(in_features, out_features, bias = FALSE) {
    self$weight <- timesfm_meta_parameter(c(out_features, in_features))
    if (isTRUE(bias)) {
      self$bias <- timesfm_meta_parameter(out_features)
    } else {
      self$bias <- NULL
    }
  },
  forward = function(inputs) {
    torch::nnf_linear(inputs, self$weight, self$bias)
  }
)

timesfm_rms_norm <- torch::nn_module(
  "timesfm_rms_norm",
  initialize = function(num_features, epsilon = 1e-6) {
    self$scale <- timesfm_meta_parameter(num_features)
    self$epsilon <- epsilon
  },
  forward = function(inputs) {
    variance <- torch::torch_mean(torch::torch_square(inputs), dim = -1, keepdim = TRUE)
    inputs * torch::torch_rsqrt(variance + self$epsilon) * self$scale
  }
)

timesfm_residual_block <- torch::nn_module(
  "timesfm_residual_block",
  initialize = function(input_dim, hidden_dim, output_dim, bias = FALSE) {
    self$hidden_layer <- timesfm_linear(input_dim, hidden_dim, bias)
    self$output_layer <- timesfm_linear(hidden_dim, output_dim, bias)
    self$residual_layer <- timesfm_linear(input_dim, output_dim, bias)
  },
  forward = function(inputs) {
    self$output_layer(torch::nnf_silu(self$hidden_layer(inputs))) +
      self$residual_layer(inputs)
  }
)

timesfm_per_dim_scale <- torch::nn_module(
  "timesfm_per_dim_scale",
  initialize = function(head_dim) {
    self$per_dim_scale <- timesfm_meta_parameter(head_dim)
  },
  forward = function(inputs) {
    inputs * (1.442695041 / sqrt(inputs$shape[[length(inputs$shape)]]) *
      torch::nnf_softplus(self$per_dim_scale))
  }
)

timesfm_rope <- torch::nn_module(
  "timesfm_rope",
  initialize = function(embedding_dims) {
    if (embedding_dims %% 2L != 0L) {
      zuk_abort_checkpoint(
        "TimesFM rotary embeddings require an even head dimension.",
        tensor = "config.json",
        expected = "an even head_dim",
        actual = embedding_dims
      )
    }
    self$embedding_dims <- as.integer(embedding_dims)
  },
  forward = function(inputs, position) {
    half <- self$embedding_dims %/% 2L
    fraction <- 2 * torch::torch_arange(
      0, half - 1L, dtype = torch::torch_float32(), device = inputs$device
    ) / self$embedding_dims
    timescale <- torch::torch_exp(log(10000) * fraction)$to(dtype = inputs$dtype)
    timescale <- timescale$view(c(1, 1, 1, half))
    angle <- position$to(dtype = inputs$dtype)$unsqueeze(-1)$unsqueeze(-1) / timescale
    halves <- torch::torch_chunk(inputs, 2L, dim = -1)
    sin_angle <- torch::torch_sin(angle)
    cos_angle <- torch::torch_cos(angle)
    torch::torch_cat(list(
      halves[[1]] * cos_angle - halves[[2]] * sin_angle,
      halves[[2]] * cos_angle + halves[[1]] * sin_angle
    ), dim = -1)
  }
)

timesfm_attention <- torch::nn_module(
  "timesfm_attention",
  initialize = function(model_dim, num_heads, head_dim, epsilon = 1e-6) {
    if (model_dim != num_heads * head_dim) {
      zuk_abort_checkpoint(
        "TimesFM attention dimensions are incompatible.",
        tensor = "config.json",
        expected = model_dim,
        actual = num_heads * head_dim
      )
    }
    self$model_dim <- as.integer(model_dim)
    self$num_heads <- as.integer(num_heads)
    self$head_dim <- as.integer(head_dim)
    self$qkv_proj <- timesfm_linear(model_dim, 3L * model_dim)
    self$out <- timesfm_linear(model_dim, model_dim)
    self$query_ln <- timesfm_rms_norm(head_dim, epsilon)
    self$key_ln <- timesfm_rms_norm(head_dim, epsilon)
    self$per_dim_scale <- timesfm_per_dim_scale(head_dim)
    self$rope <- timesfm_rope(head_dim)
  },
  forward = function(inputs, patch_mask, decode_cache = NULL) {
    batch <- inputs$shape[[1]]
    n_patches <- inputs$shape[[2]]
    qkv <- torch::torch_chunk(self$qkv_proj(inputs), 3L, dim = -1)
    query <- qkv[[1]]$reshape(c(batch, n_patches, self$num_heads, self$head_dim))
    key <- qkv[[2]]$reshape(c(batch, n_patches, self$num_heads, self$head_dim))
    value <- qkv[[3]]$reshape(c(batch, n_patches, self$num_heads, self$head_dim))

    newly_masked <- patch_mask$to(dtype = torch::torch_int32())$sum(dim = -1)
    if (is.null(decode_cache)) {
      next_index <- 0L
      num_masked <- newly_masked
      all_value <- value
    } else {
      next_index <- decode_cache$key$shape[[2]]
      num_masked <- decode_cache$num_masked + newly_masked
      all_value <- torch::torch_cat(list(decode_cache$value, value), dim = 2L)
    }

    position <- torch::torch_arange(
      0, n_patches - 1L, device = inputs$device
    )$unsqueeze(1) + next_index - num_masked$unsqueeze(-1)
    query <- self$query_ln(self$rope(query, position))
    key <- self$key_ln(self$rope(key, position))
    # The cache holds rotated keys, so concatenate only after applying RoPE.
    all_key <- if (is.null(decode_cache)) {
      key
    } else {
      torch::torch_cat(list(decode_cache$key, key), dim = 2L)
    }

    query <- self$per_dim_scale(query)
    query <- query$permute(c(1, 3, 2, 4))
    key_for_scores <- all_key$permute(c(1, 3, 2, 4))
    value_for_scores <- all_value$permute(c(1, 3, 2, 4))
    kv_length <- all_key$shape[[2]]
    query_index <- torch::torch_arange(
      next_index, next_index + n_patches - 1L, device = inputs$device
    )$view(c(1, 1, n_patches, 1))
    key_index <- torch::torch_arange(
      0, kv_length - 1L, device = inputs$device
    )$view(c(1, 1, 1, kv_length))
    mask <- torch::torch_logical_and(
      query_index >= key_index,
      key_index >= num_masked$view(c(batch, 1, 1, 1))
    )
    attended <- torch::torch_scaled_dot_product_attention(
      query, key_for_scores, value_for_scores, attn_mask = mask, scale = 1
    )$permute(c(1, 3, 2, 4))
    output <- self$out(attended$reshape(c(batch, n_patches, self$model_dim)))
    list(
      output = output,
      cache = list(key = all_key, value = all_value, num_masked = num_masked)
    )
  }
)

timesfm_transformer <- torch::nn_module(
  "timesfm_transformer",
  initialize = function(model_dim, hidden_dim, num_heads, head_dim, epsilon = 1e-6) {
    self$pre_attn_ln <- timesfm_rms_norm(model_dim, epsilon)
    self$post_attn_ln <- timesfm_rms_norm(model_dim, epsilon)
    self$attn <- timesfm_attention(model_dim, num_heads, head_dim, epsilon)
    self$pre_ff_ln <- timesfm_rms_norm(model_dim, epsilon)
    self$post_ff_ln <- timesfm_rms_norm(model_dim, epsilon)
    self$ff0 <- timesfm_linear(model_dim, hidden_dim)
    self$ff1 <- timesfm_linear(hidden_dim, model_dim)
  },
  forward = function(inputs, patch_mask, decode_cache = NULL) {
    attention <- self$attn(self$pre_attn_ln(inputs), patch_mask, decode_cache)
    attended <- self$post_attn_ln(attention$output) + inputs
    feed_forward <- self$ff1(torch::nnf_silu(self$ff0(self$pre_ff_ln(attended))))
    list(
      output = self$post_ff_ln(feed_forward) + attended,
      cache = attention$cache
    )
  }
)

timesfm_network <- torch::nn_module(
  "timesfm_network",
  initialize = function(config) {
    model_dim <- as.integer(config$hidden_size)
    hidden_dim <- as.integer(config$intermediate_size)
    heads <- as.integer(config$num_attention_heads)
    head_dim <- as.integer(config$head_dim)
    layers <- as.integer(config$num_hidden_layers)
    patch <- as.integer(config$patch_length)
    point_horizon <- as.integer(config$horizon_length)
    quantile_horizon <- as.integer(config$quantile_horizon_length)
    outputs <- length(config$quantiles) + 1L
    epsilon <- as.numeric(config$rms_norm_eps %||% 1e-6)

    self$tokenizer <- timesfm_residual_block(2L * patch, hidden_dim, model_dim, TRUE)
    self$stacked_xf <- torch::nn_module_list(lapply(seq_len(layers), function(i) {
      timesfm_transformer(model_dim, hidden_dim, heads, head_dim, epsilon)
    }))
    self$output_projection_point <- timesfm_residual_block(
      model_dim, hidden_dim, point_horizon * outputs
    )
    self$output_projection_quantiles <- timesfm_residual_block(
      model_dim, hidden_dim, quantile_horizon * outputs
    )
  },
  forward = function(inputs, masks, decode_caches = NULL) {
    embeddings <- self$tokenizer(torch::torch_cat(
      list(inputs, masks$to(dtype = inputs$dtype)), dim = -1
    ))
    if (is.null(decode_caches)) {
      decode_caches <- rep(list(NULL), length(self$stacked_xf))
    }
    output <- embeddings
    new_caches <- vector("list", length(self$stacked_xf))
    patch_mask <- masks[, , masks$shape[[3]]]
    for (i in seq_along(self$stacked_xf)) {
      transformed <- self$stacked_xf[[i]](output, patch_mask, decode_caches[[i]])
      output <- transformed$output
      new_caches[[i]] <- transformed$cache
    }
    list(
      input_embeddings = embeddings,
      output_embeddings = output,
      point = self$output_projection_point(output),
      quantiles = self$output_projection_quantiles(output),
      caches = new_caches
    )
  }
)

timesfm_module <- function(config) {
  timesfm_network(config)
}
