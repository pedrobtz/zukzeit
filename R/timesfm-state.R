# Portions derived from TimesFM, Copyright 2025 Google LLC.
# Translated and modified for R/torch by the tsfm authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Exact state-dict layout for the pinned TimesFM 2.5 200M checkpoint.

timesfm_expected_state_spec <- function(config) {
  model_dim <- as.integer(config$hidden_size %||% 1280L)
  hidden_dim <- as.integer(config$intermediate_size %||% model_dim)
  head_dim <- as.integer(config$head_dim %||% 80L)
  layers <- as.integer(config$num_hidden_layers %||% 20L)
  patch_length <- as.integer(config$patch_length %||% 32L)
  point_horizon <- as.integer(config$horizon_length %||% 128L)
  quantile_horizon <- as.integer(config$quantile_horizon_length %||% 1024L)
  outputs <- length(config$quantiles %||% seq(0.1, 0.9, by = 0.1)) + 1L

  spec <- list(
    "output_projection_point.hidden_layer.weight" = c(hidden_dim, model_dim),
    "output_projection_point.output_layer.weight" = c(point_horizon * outputs, hidden_dim),
    "output_projection_point.residual_layer.weight" = c(point_horizon * outputs, model_dim),
    "output_projection_quantiles.hidden_layer.weight" = c(hidden_dim, model_dim),
    "output_projection_quantiles.output_layer.weight" = c(quantile_horizon * outputs, hidden_dim),
    "output_projection_quantiles.residual_layer.weight" = c(quantile_horizon * outputs, model_dim)
  )
  for (layer in seq_len(layers) - 1L) {
    prefix <- sprintf("stacked_xf.%d", layer)
    layer_spec <- list(
      c(head_dim),
      c(model_dim, model_dim),
      c(head_dim),
      c(3L * model_dim, model_dim),
      c(head_dim),
      c(hidden_dim, model_dim),
      c(model_dim, hidden_dim),
      c(model_dim),
      c(model_dim),
      c(model_dim),
      c(model_dim)
    )
    names(layer_spec) <- paste0(
      prefix,
      c(
        ".attn.key_ln.scale",
        ".attn.out.weight",
        ".attn.per_dim_scale.per_dim_scale",
        ".attn.qkv_proj.weight",
        ".attn.query_ln.scale",
        ".ff0.weight",
        ".ff1.weight",
        ".post_attn_ln.scale",
        ".post_ff_ln.scale",
        ".pre_attn_ln.scale",
        ".pre_ff_ln.scale"
      )
    )
    spec <- c(spec, layer_spec)
  }
  tokenizer <- list(
    c(model_dim),
    c(model_dim, 2L * patch_length),
    c(model_dim),
    c(model_dim, model_dim),
    c(model_dim),
    c(model_dim, 2L * patch_length)
  )
  names(tokenizer) <- c(
    "tokenizer.hidden_layer.bias",
    "tokenizer.hidden_layer.weight",
    "tokenizer.output_layer.bias",
    "tokenizer.output_layer.weight",
    "tokenizer.residual_layer.bias",
    "tokenizer.residual_layer.weight"
  )
  c(spec, tokenizer)
}

validate_timesfm_state_metadata <- function(metadata, config,
                                             model_id = NA_character_,
                                             revision = NA_character_) {
  expected <- timesfm_expected_state_spec(config)
  missing <- setdiff(names(expected), names(metadata))
  extra <- setdiff(names(metadata), names(expected))
  if (length(missing) || length(extra)) {
    tsfm_abort_checkpoint(
      c(
        "The TimesFM checkpoint tensor names do not match the pinned layout.",
        "x" = "Missing: {.val {missing}}.",
        "x" = "Unexpected: {.val {extra}}."
      ),
      model_id = model_id,
      revision = revision,
      expected = names(expected),
      actual = names(metadata)
    )
  }
  for (name in names(expected)) {
    actual_shape <- as.integer(metadata[[name]]$shape)
    expected_shape <- as.integer(expected[[name]])
    if (!identical(actual_shape, expected_shape) ||
        !identical(metadata[[name]]$dtype, "F32")) {
      tsfm_abort_checkpoint(
        "Tensor {.val {name}} is incompatible with the pinned TimesFM layout.",
        model_id = model_id,
        revision = revision,
        tensor = name,
        expected = list(shape = expected_shape, dtype = "F32"),
        actual = list(shape = actual_shape, dtype = metadata[[name]]$dtype)
      )
    }
  }
  invisible(metadata)
}
