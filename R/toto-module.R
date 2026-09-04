# Portions derived from Toto 2.0, Copyright 2026 Datadog, Inc.
# Translated and modified for R/torch by the zukzeit authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Native R torch layers for the pinned Toto 2.0 4M checkpoint.
#
# Toto is trained under u-muP, and its reference implementation runs every
# linear, activation, residual, and per-dimension scale through
# `dd_unit_scaling`. Those factors apply at *inference*, not just training, and
# nothing in the state dict hints at them: a port that treats `uu.Linear` as a
# plain linear layer maps all 48 tensors perfectly and computes the wrong
# answer everywhere. The primitives below are the whole of that contract, each
# verified against the pinned reference.

# `uu.Linear`: the bias is inside the scale, not added after it.
toto_unit_linear <- function(input, weight, bias = NULL) {
  torch::nnf_linear(input, weight, bias) / sqrt(weight$shape[[2]])
}

# `uu.LinearReadout`: readout layers divide by fan_in rather than its square
# root. Only the output head's second and skip projections use this.
toto_unit_readout <- function(input, weight, bias = NULL) {
  torch::nnf_linear(input, weight, bias) / weight$shape[[2]]
}

# `U.silu` at mult = 1: a logarithmic interpolation between 2 and
# sqrt(2 / (1 - 1/pi)) at alpha = 0.8.
toto_silu_mult <- exp(0.2 * log(2) + 0.8 * log(sqrt(2 / (1 - 1 / pi))))

toto_unit_silu <- function(input) {
  torch::nnf_silu(input) * toto_silu_mult
}

# `U.residual_add`. Its partner `U.residual_split` is the identity in the
# forward pass --- it only shapes gradients --- so it has no counterpart here.
toto_residual_add <- function(hidden, skip, tau) {
  (tau * hidden + skip) / torch::torch_sqrt(1 + tau * tau)
}

# `uu.PerDimScale`. Note the absence of a 1/sqrt(head_dim) factor: unit-scaled
# attention accounts for it elsewhere, which is exactly where a port that
# borrows another architecture's per-dimension scale goes silently wrong.
toto_per_dim_scale <- function(input, scale) {
  input * (torch::nnf_softplus(scale) / log(2))
}

# `uu.RMSNorm` with include_weight = FALSE: parameter-free, which is why the
# checkpoint carries no normalization tensors at all.
toto_rms_norm <- function(input, epsilon) {
  variance <- torch::torch_mean(torch::torch_square(input), dim = -1, keepdim = TRUE)
  input * torch::torch_rsqrt(variance + epsilon)
}

toto_meta_parameter <- function(shape) {
  torch::nn_parameter(torch::torch_empty(shape, device = torch::torch_device("meta")))
}

# The two residual gains are 0-dimension tensors in the checkpoint, and
# `torch_empty()` cannot express an empty shape, so they are built directly.
toto_meta_scalar <- function() {
  torch::nn_parameter(
    torch::torch_scalar_tensor(0, device = torch::torch_device("meta"))
  )
}

# ---- residual MLP ------------------------------------------------------------

# `ResidualMLP`. The `readout` flag selects `OutputResidualMLP`, whose second
# and skip projections are readouts; the input and hidden variants use ordinary
# unit-scaled linears throughout. Both are constructed with tau = 1.
toto_residual_mlp <- torch::nn_module(
  "toto_residual_mlp",
  initialize = function(in_dim, hidden_dim, out_dim, readout = FALSE, tau = 1) {
    self$linear1 <- toto_meta_parameter(c(hidden_dim, in_dim))
    self$linear1_bias <- toto_meta_parameter(hidden_dim)
    self$linear2 <- toto_meta_parameter(c(out_dim, hidden_dim))
    self$linear2_bias <- toto_meta_parameter(out_dim)
    self$skip <- toto_meta_parameter(c(out_dim, in_dim))
    self$skip_bias <- toto_meta_parameter(out_dim)
    self$readout <- readout
    self$tau <- tau
  },
  forward = function(input) {
    second <- if (isTRUE(self$readout)) toto_unit_readout else toto_unit_linear
    hidden <- toto_unit_silu(toto_unit_linear(input, self$linear1, self$linear1_bias))
    hidden <- second(hidden, self$linear2, self$linear2_bias)
    skip <- second(input, self$skip, self$skip_bias)
    toto_residual_add(hidden, skip, torch::torch_tensor(self$tau))
  }
)

# ---- rotary position ---------------------------------------------------------

# The rotation pairs *adjacent* channels rather than splitting the head in
# half, so each frequency is repeated twice in the tables to match.
toto_rope_tables <- function(width, max_len, device) {
  theta <- 1 / torch::torch_pow(
    10000,
    torch::torch_arange(0, width - 1, 2, dtype = torch::torch_float32(), device = device) / width
  )
  angle <- torch::torch_outer(
    torch::torch_arange(0, max_len - 1, dtype = torch::torch_float32(), device = device),
    theta
  )
  angle <- angle$unsqueeze(-1)$expand(c(angle$shape, 2))$flatten(start_dim = -2)
  list(cos = torch::torch_cos(angle), sin = torch::torch_sin(angle))
}

toto_rope_rotate <- function(input) {
  width <- input$shape[[length(input$shape)]]
  last <- length(input$shape)
  pick <- function(from) torch::torch_index_select(
    input, last,
    torch::torch_tensor(seq(from, width, by = 2L), dtype = torch::torch_long(),
                        device = input$device)
  )
  torch::torch_stack(list(-pick(2L), pick(1L)), dim = -1)$flatten(start_dim = -2)
}

# xPos length-extrapolation decay. Query and key take opposite exponents, so
# their product decays with distance; using one exponent for both leaves every
# shape intact and quietly removes the decay.
toto_xpos_scale <- function(positions, width, exponent, scale_base = 256) {
  base <- (torch::torch_arange(
    0, width - 1, 2, dtype = torch::torch_float32(), device = positions$device
  ) + 0.4 * width) / (1.4 * width)
  centre <- torch::torch_floor((torch::torch_max(positions) + 1) / 2)
  power <- (positions$to(dtype = torch::torch_float32()) - centre) / scale_base
  scale <- torch::torch_pow(base$unsqueeze(1), power$unsqueeze(-1))
  scale <- scale$unsqueeze(-1)$expand(c(scale$shape, 2))$flatten(start_dim = -2)
  torch::torch_pow(scale, exponent)
}

toto_apply_rope <- function(input, tables, positions, width, exponent) {
  cosine <- tables$cos[positions + 1L, ]$unsqueeze(1)
  sine <- tables$sin[positions + 1L, ]$unsqueeze(1)
  rotated <- cosine * input + sine * toto_rope_rotate(input)
  rotated * toto_xpos_scale(positions, width, exponent)$unsqueeze(1)
}

# ---- attention ---------------------------------------------------------------

# Time layers attend causally along the sequence and carry rotary position;
# variate layers attend across series at each step, receive no rotary
# projection at all, and are not causal.
toto_attention <- torch::nn_module(
  "toto_attention",
  initialize = function(model_dim, num_heads, qk_dim, v_dim, variate = FALSE) {
    fused <- qk_dim * num_heads + qk_dim * num_heads + v_dim * num_heads
    self$in_proj <- toto_meta_parameter(c(fused, model_dim))
    self$in_proj_bias <- toto_meta_parameter(fused)
    self$out_proj <- toto_meta_parameter(c(model_dim, v_dim * num_heads))
    self$out_proj_bias <- toto_meta_parameter(model_dim)
    self$per_dim_scale <- toto_meta_parameter(qk_dim)
    self$num_heads <- as.integer(num_heads)
    self$qk_dim <- as.integer(qk_dim)
    self$variate <- isTRUE(variate)
  },
  forward = function(input, mask = NULL) {
    batch <- input$shape[[1]]
    len <- input$shape[[2]]
    heads <- self$num_heads
    width <- self$qk_dim
    fused <- toto_unit_linear(input, self$in_proj, self$in_proj_bias)
    parts <- torch::torch_split(fused, width * heads, dim = -1)
    reshape <- function(x) x$view(c(batch, len, heads, width))$permute(c(1, 3, 2, 4))
    query <- reshape(parts[[1]])
    key <- reshape(parts[[2]])
    value <- reshape(parts[[3]])
    query <- toto_per_dim_scale(query, self$per_dim_scale)

    if (!self$variate) {
      # partial_factor = (0, 0.5): only the leading half of each head rotates.
      half <- width %/% 2L
      tables <- toto_rope_tables(half, 8192L, input$device)
      positions <- torch::torch_arange(0, len - 1, dtype = torch::torch_long(),
                                       device = input$device)
      lead <- function(x) x$narrow(-1, 1L, half)
      rest <- function(x) x$narrow(-1, half + 1L, width - half)
      query <- torch::torch_cat(list(
        toto_apply_rope(lead(query), tables, positions, half, 1), rest(query)
      ), dim = -1)
      key <- torch::torch_cat(list(
        toto_apply_rope(lead(key), tables, positions, half, -1), rest(key)
      ), dim = -1)
    }

    # MuP scales by 1/qk_dim, not its square root, to stop logits growing with
    # width. At qk_dim = 64 the conventional choice is wrong by a factor of 8.
    scores <- torch::torch_matmul(query, key$transpose(-2, -1)) * (1 / width)
    if (!self$variate) {
      causal <- torch::torch_tril(
        torch::torch_ones(len, len, dtype = torch::torch_bool(), device = input$device)
      )$unsqueeze(1)$unsqueeze(1)
      if (!is.null(mask)) causal <- causal & mask
      scores <- scores$masked_fill(!causal, -Inf)
    }
    attended <- torch::torch_matmul(torch::nnf_softmax(scores, dim = -1), value)
    attended <- attended$permute(c(1, 3, 2, 4))$reshape(c(batch, len, heads * width))
    toto_unit_linear(attended, self$out_proj, self$out_proj_bias)
  }
)

# ---- transformer -------------------------------------------------------------

# `attn_tau` and `mlp_tau` are derived from the residual scaling rule upstream,
# but they are persistent buffers and therefore present in the checkpoint, so
# they are read rather than recomputed.
toto_transformer_layer <- torch::nn_module(
  "toto_transformer_layer",
  initialize = function(model_dim, ff_dim, num_heads, qk_dim, v_dim, epsilon,
                        variate = FALSE) {
    self$attn <- toto_attention(model_dim, num_heads, qk_dim, v_dim, variate)
    self$ff1 <- toto_meta_parameter(c(2L * ff_dim, model_dim))
    self$ff2 <- toto_meta_parameter(c(model_dim, ff_dim))
    self$attn_tau <- toto_meta_scalar()
    self$mlp_tau <- toto_meta_scalar()
    self$epsilon <- epsilon
  },
  forward = function(input, mask = NULL) {
    attended <- self$attn(toto_rms_norm(input, self$epsilon), mask)
    input <- toto_residual_add(attended, input, self$attn_tau)
    # SwiGLU: fc1 emits 2 x d_ff, split into gate and value.
    projected <- toto_unit_linear(toto_rms_norm(input, self$epsilon), self$ff1)
    halves <- torch::torch_chunk(projected, 2L, dim = -1)
    hidden <- toto_unit_linear(halves[[1]] * torch::nnf_silu(halves[[2]]), self$ff2)
    toto_residual_add(hidden, input, self$mlp_tau)
  }
)

toto_network <- torch::nn_module(
  "toto_network",
  initialize = function(config) {
    model_dim <- as.integer(config$d_model)
    ff_dim <- as.integer(config$d_ff)
    patch <- as.integer(config$patch_size)
    epsilon <- as.numeric(config$norm_eps)
    outputs <- length(config$quantiles %||% seq(0.1, 0.9, by = 0.1))
    self$patch_proj <- toto_residual_mlp(2L * patch, 4L * model_dim, model_dim)
    self$layers <- torch::nn_module_list(lapply(
      seq_len(as.integer(config$num_layers)) - 1L,
      function(idx) toto_transformer_layer(
        model_dim, ff_dim,
        as.integer(config$num_heads), as.integer(config$qk_dim),
        as.integer(config$v_dim), epsilon,
        variate = toto_is_variate_layer(idx, config)
      )
    ))
    self$output_head <- toto_residual_mlp(
      model_dim, 4L * model_dim,
      patch * as.integer(config$num_output_patches) * outputs,
      readout = TRUE
    )
    self$epsilon <- epsilon
    self$patch <- patch
    self$outputs <- outputs
  },
  # Split from forward() so the decode path can run the stack and the head
  # separately: next-patch alignment slices the stack's output before the head
  # ever sees it.
  run_layers = function(embedded, mask = NULL) {
    # Variate layers attend across series at a fixed time step, so the series
    # axis has to become the sequence axis for the duration of the layer and
    # the time axis moves into the batch. Contract v1 is univariate, so that
    # axis has extent one --- attention over a single token is still a learned
    # transform through the value and output projections, not an identity.
    batch <- embedded$shape[[1]]
    len <- embedded$shape[[2]]
    dim <- embedded$shape[[3]]
    state <- embedded
    for (i in seq_along(self$layers)) {
      layer <- self$layers[[i]]
      if (isTRUE(layer$attn$variate)) {
        state <- state$reshape(c(batch * len, 1L, dim))
        state <- layer(state)
        state <- state$reshape(c(batch, len, dim))
      } else {
        state <- layer(state, mask)
      }
    }
    toto_rms_norm(state, self$epsilon)
  },
  forward = function(patches, masks) {
    # Channel layout is [scaled values, mask-as-float], hence 2 x patch in.
    embedded <- self$patch_proj(torch::torch_cat(
      list(patches, masks$to(dtype = patches$dtype)), dim = -1
    ))
    quantiles <- self$output_head(self$run_layers(embedded))
    quantiles$unflatten(-1, c(self$patch, self$outputs))
  }
)

# One group of `layer_group_size` layers holds `num_variate_layers_per_group`
# variate layers, placed last unless `variate_layer_first`.
toto_is_variate_layer <- function(layer_idx, config) {
  group <- as.integer(config$layer_group_size)
  variate <- as.integer(config$num_variate_layers_per_group)
  if (isTRUE(config$variate_layer_first)) {
    layer_idx %% group < variate
  } else {
    layer_idx %% group >= group - variate
  }
}

toto_module <- function(config) {
  toto_network(config)
}
