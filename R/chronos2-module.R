# Portions derived from Chronos-2, Copyright Amazon.com, Inc. or its affiliates.
# Translated and modified for R/torch by the zukzeit authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Native R torch layers for the pinned Chronos-2 checkpoint.
#
# An encoder-only T5 whose every block pairs attention over time with attention
# across the series in a task. Unlike the Toto port there is no unit scaling
# here: plain linear layers reproduce the reference exactly.

chronos2_meta_parameter <- function(shape) {
  torch::nn_parameter(torch::torch_empty(shape, device = torch::torch_device("meta")))
}

# T5-style: no mean subtraction, a learnable weight, variance in float32.
chronos2_layer_norm <- function(input, weight, epsilon) {
  variance <- torch::torch_mean(
    torch::torch_square(input$to(dtype = torch::torch_float32())),
    dim = -1, keepdim = TRUE
  )
  weight * (input * torch::torch_rsqrt(variance + epsilon))
}

# Split-half rotation, the convention this checkpoint uses. Note it is the
# opposite of the Toto port's interleaved adjacent pairs: the two ports cannot
# share a rotary implementation.
chronos2_rotate_half <- function(input) {
  half <- input$shape[[length(input$shape)]] %/% 2L
  torch::torch_cat(
    list(-input$narrow(-1, half + 1L, half), input$narrow(-1, 1L, half)),
    dim = -1
  )
}

chronos2_rope <- function(seq_len, head_dim, theta, device) {
  inverse <- 1 / torch::torch_pow(
    theta,
    torch::torch_arange(0, head_dim - 1, 2, dtype = torch::torch_float32(),
                        device = device) / head_dim
  )
  positions <- torch::torch_arange(0, seq_len - 1, dtype = torch::torch_float32(),
                                   device = device)
  angle <- torch::torch_outer(positions, inverse)
  angle <- torch::torch_cat(list(angle, angle), dim = -1)
  list(cos = torch::torch_cos(angle)$unsqueeze(1),
       sin = torch::torch_sin(angle)$unsqueeze(1))
}

# T5 attention is *unscaled*: the reference passes scale = 1.0, folding the
# usual 1/sqrt(d_kv) into initialization. Using the conventional factor here
# would be wrong by sqrt(64) = 8.
chronos2_attention <- torch::nn_module(
  "chronos2_attention",
  initialize = function(model_dim, num_heads, head_dim) {
    inner <- num_heads * head_dim
    self$q <- chronos2_meta_parameter(c(inner, model_dim))
    self$k <- chronos2_meta_parameter(c(inner, model_dim))
    self$v <- chronos2_meta_parameter(c(inner, model_dim))
    self$o <- chronos2_meta_parameter(c(model_dim, inner))
    self$num_heads <- as.integer(num_heads)
    self$head_dim <- as.integer(head_dim)
  },
  forward = function(input, mask, rope = NULL) {
    batch <- input$shape[[1]]
    len <- input$shape[[2]]
    shape <- function(weight) {
      torch::nnf_linear(input, weight)$view(
        c(batch, len, self$num_heads, self$head_dim)
      )$permute(c(1, 3, 2, 4))
    }
    query <- shape(self$q)
    key <- shape(self$k)
    value <- shape(self$v)
    if (!is.null(rope)) {
      query <- query * rope$cos + chronos2_rotate_half(query) * rope$sin
      key <- key * rope$cos + chronos2_rotate_half(key) * rope$sin
    }
    scores <- torch::torch_matmul(query, key$transpose(-2, -1)) + mask
    attended <- torch::torch_matmul(torch::nnf_softmax(scores, dim = -1), value)
    attended <- attended$permute(c(1, 3, 2, 4))$reshape(
      c(batch, len, self$num_heads * self$head_dim)
    )
    torch::nnf_linear(attended, self$o)
  }
)

# Each block attends twice: along time, carrying rotary position, and across the
# series of a task, which has no natural ordering and so takes no rotary.
chronos2_block <- torch::nn_module(
  "chronos2_block",
  initialize = function(model_dim, ff_dim, num_heads, head_dim, epsilon) {
    self$time_attn <- chronos2_attention(model_dim, num_heads, head_dim)
    self$time_norm <- chronos2_meta_parameter(model_dim)
    self$group_attn <- chronos2_attention(model_dim, num_heads, head_dim)
    self$group_norm <- chronos2_meta_parameter(model_dim)
    self$ff_norm <- chronos2_meta_parameter(model_dim)
    self$wi <- chronos2_meta_parameter(c(ff_dim, model_dim))
    self$wo <- chronos2_meta_parameter(c(model_dim, ff_dim))
    self$epsilon <- epsilon
  },
  forward = function(input, time_mask, group_time_mask, rope) {
    state <- input + self$time_attn(
      chronos2_layer_norm(input, self$time_norm, self$epsilon), time_mask, rope
    )
    # Series become the sequence for the group pass, and time moves aside.
    swapped <- state$permute(c(2, 1, 3))
    swapped <- swapped + self$group_attn(
      chronos2_layer_norm(swapped, self$group_norm, self$epsilon),
      group_time_mask, NULL
    )
    state <- swapped$permute(c(2, 1, 3))
    hidden <- chronos2_layer_norm(state, self$ff_norm, self$epsilon)
    state + torch::nnf_linear(
      torch::nnf_relu(torch::nnf_linear(hidden, self$wi)), self$wo
    )
  }
)

# `hidden_layer -> act -> output_layer`, plus a linear skip from the input.
chronos2_residual_block <- torch::nn_module(
  "chronos2_residual_block",
  initialize = function(in_dim, hidden_dim, out_dim) {
    self$hidden <- chronos2_meta_parameter(c(hidden_dim, in_dim))
    self$hidden_bias <- chronos2_meta_parameter(hidden_dim)
    self$out <- chronos2_meta_parameter(c(out_dim, hidden_dim))
    self$out_bias <- chronos2_meta_parameter(out_dim)
    self$skip <- chronos2_meta_parameter(c(out_dim, in_dim))
    self$skip_bias <- chronos2_meta_parameter(out_dim)
  },
  forward = function(input) {
    hidden <- torch::nnf_relu(torch::nnf_linear(input, self$hidden, self$hidden_bias))
    torch::nnf_linear(hidden, self$out, self$out_bias) +
      torch::nnf_linear(input, self$skip, self$skip_bias)
  }
)

chronos2_network <- torch::nn_module(
  "chronos2_network",
  initialize = function(config) {
    model_dim <- as.integer(config$d_model)
    ff_dim <- as.integer(config$d_ff)
    patch <- as.integer(config$input_patch_size)
    quantiles <- length(config$quantiles)
    epsilon <- as.numeric(config$layer_norm_epsilon)
    # Three channels per patch position: a time encoding, the values, the mask.
    self$input_embedding <- chronos2_residual_block(3L * patch, ff_dim, model_dim)
    self$output_embedding <- chronos2_residual_block(
      model_dim, ff_dim, quantiles * as.integer(config$output_patch_size)
    )
    self$shared <- chronos2_meta_parameter(c(2L, model_dim))
    self$blocks <- torch::nn_module_list(lapply(
      seq_len(as.integer(config$num_layers)),
      function(i) chronos2_block(model_dim, ff_dim, as.integer(config$num_heads),
                                 as.integer(config$d_kv), epsilon)
    ))
    self$final_norm <- chronos2_meta_parameter(model_dim)
    self$epsilon <- epsilon
    self$head_dim <- as.integer(config$d_kv)
    self$theta <- as.numeric(config$rope_theta)
    self$quantiles <- quantiles
    self$patch <- patch
  },
  forward = function(embeddings, time_mask, group_time_mask) {
    rope <- chronos2_rope(embeddings$shape[[2]], self$head_dim, self$theta,
                          embeddings$device)
    state <- embeddings
    for (i in seq_along(self$blocks)) {
      state <- self$blocks[[i]](state, time_mask, group_time_mask, rope)
    }
    chronos2_layer_norm(state, self$final_norm, self$epsilon)
  }
)

chronos2_module <- function(config) {
  chronos2_network(config)
}
