# Portions derived from Toto 2.0, Copyright 2026 Datadog, Inc.
# Translated and modified for R/torch by the zukzeit authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Native Toto 2.0 4M architecture entry point.

toto_config_value <- function(config, name, default) {
  value <- config[[name]] %||% default
  if (length(value) != 1L || !is.numeric(value) || is.na(value) ||
      !is.finite(value) || value <= 0 || value != floor(value)) {
    zuk_abort_checkpoint(
      "Toto config field {.val {name}} must be one positive integer.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      tensor = "config.json",
      expected = "one positive integer",
      actual = value
    )
  }
  as.integer(value)
}

# The checkpoint's trained context length is not stored anywhere, but it is
# recoverable: upstream derives `residual_attn_ratio` as sqrt(S / log S) with
# S = context_length / patch_size. Inverting the stored constant reproduces
# S = 128 exactly in double precision, so the same arithmetic both recovers the
# value and validates that this is the configuration the port was written for.
toto_context_length <- function(config) {
  patch <- as.integer(config$patch_size %||% 32L)
  ratio <- as.numeric(config$residual_attn_ratio %||% NA_real_)
  expected <- sqrt(128 / log(128))
  if (!isTRUE(all.equal(ratio, expected))) {
    zuk_abort_checkpoint(
      c(
        "This Toto checkpoint variant is not compatible with the native port.",
        "i" = "Only the pinned Toto 2.0 4M configuration is supported."
      ),
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      tensor = "config.json",
      expected = expected,
      actual = ratio
    )
  }
  128L * patch
}

validate_toto_config <- function(config) {
  actual <- list(
    d_model = toto_config_value(config, "d_model", 256L),
    d_ff = toto_config_value(config, "d_ff", 688L),
    num_heads = toto_config_value(config, "num_heads", 4L),
    num_layers = toto_config_value(config, "num_layers", 4L),
    qk_dim = toto_config_value(config, "qk_dim", 64L),
    v_dim = toto_config_value(config, "v_dim", 64L),
    patch_size = toto_config_value(config, "patch_size", 32L),
    num_output_patches = toto_config_value(config, "num_output_patches", 1L),
    layer_group_size = toto_config_value(config, "layer_group_size", 4L),
    num_variate_layers_per_group = toto_config_value(
      config, "num_variate_layers_per_group", 1L
    )
  )
  expected <- list(
    d_model = 256L, d_ff = 688L, num_heads = 4L, num_layers = 4L,
    qk_dim = 64L, v_dim = 64L, patch_size = 32L, num_output_patches = 1L,
    layer_group_size = 4L, num_variate_layers_per_group = 1L
  )
  # Each of these silently changes the arithmetic rather than the shapes, so
  # they are checked rather than assumed: unit scaling is not optional, the
  # normalization carries no weights, per-dimension query scaling is on, and
  # the rotary path is xPos rather than plain RoPE.
  flags <- list(
    use_xpos = isTRUE(config$use_xpos %||% FALSE),
    per_dim_scale = isTRUE(config$per_dim_scale %||% FALSE),
    pre_norm = isTRUE(config$pre_norm %||% FALSE),
    attn_bias = isTRUE(config$attn_bias %||% FALSE),
    mlp_bias = isFALSE(config$mlp_bias %||% TRUE),
    qk_norm = isFALSE(config$qk_norm %||% TRUE),
    norm_include_weight = isFALSE(config$norm_include_weight %||% TRUE),
    variate_layer_last = isFALSE(config$variate_layer_first %||% TRUE)
  )
  wrong <- names(expected)[vapply(
    names(expected),
    function(name) !identical(actual[[name]], expected[[name]]),
    logical(1)
  )]
  bad_flags <- names(flags)[!vapply(flags, isTRUE, logical(1))]
  quantiles <- as.numeric(config$quantiles %||% seq(0.1, 0.9, by = 0.1))
  if (length(wrong) || length(bad_flags) ||
      !isTRUE(all.equal(quantiles, seq(0.1, 0.9, by = 0.1)))) {
    zuk_abort_checkpoint(
      c(
        "This Toto checkpoint variant is not compatible with the native port.",
        "i" = "Only the pinned Toto 2.0 4M configuration is supported."
      ),
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      tensor = "config.json",
      expected = c(expected, list(quantiles = seq(0.1, 0.9, by = 0.1))),
      actual = c(actual, list(quantiles = quantiles, flags = bad_flags))
    )
  }
  invisible(config)
}

toto_capabilities <- function(config) {
  new_zuk_capabilities(
    architecture      = "toto2",
    max_context       = as.integer(config$context_length),
    # This port decodes in a single pass. Upstream chunks longer horizons into
    # blocks of `decode_block_size` and feeds medians back between them, which
    # is a different computation and is deliberately not implemented here, so
    # the declared ceiling is the largest horizon one block covers. Requesting
    # more is refused rather than silently decoded outside the trained regime.
    max_horizon       = 768L,
    quantiles         = "native",
    quantile_levels   = as.numeric(config$quantiles %||% seq(0.1, 0.9, by = 0.1)),
    multivariate      = FALSE,
    samples           = FALSE,
    past_covariates   = FALSE,
    future_covariates = FALSE,
    static_covariates = FALSE,
    fine_tunable      = FALSE,
    license           = "Apache-2.0"
  )
}

toto_constructor <- function(config, weights = NULL) {
  zuk_require_namespace("torch", reason = "It is needed for native Toto inference.")
  config$quantiles <- as.numeric(config$quantiles %||% seq(0.1, 0.9, by = 0.1))
  config$context_length <- toto_context_length(config)
  validate_toto_config(config)
  if (is.null(weights)) {
    zuk_abort_checkpoint(
      "Native Toto construction requires checkpoint weights.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      expected = "a complete named state dict",
      actual = NULL
    )
  }
  module <- toto_module(config)
  toto_load_module_weights(module, weights, config)
  device <- config$device %||% "cpu"
  module$to(device = torch::torch_device(device))
  module$eval()

  # See the note in arch-timesfm.R: writing `device = device` here would make
  # the default a recursive self-reference rather than the loaded device.
  predict_batch <- function(contexts, horizons, quantile_levels,
                            device = config$device) {
    if (!identical(as.character(device), as.character(config$device))) {
      zuk_abort_device(
        c(
          "Inference device {.val {device}} differs from the loaded Toto handle.",
          "i" = "Load a separate handle with {.code zuk_pretrained(..., device = {device})}."
        ),
        requested_device = device,
        resolved_device = config$device
      )
    }
    toto_predict_batch(module, contexts, horizons, quantile_levels, config, device)
  }
  predict_one <- function(context, h, quantile_levels) {
    predict_batch(list(context), as.integer(h), quantile_levels, device)[[1]]
  }

  new_zuk_model(
    architecture = "toto2",
    config = config,
    capabilities = toto_capabilities(config),
    predict_fn = predict_one,
    predict_batch_fn = predict_batch,
    model_id = config$model_id %||% NA_character_,
    revision = config$revision %||% NA_character_,
    device = device,
    params = module
  )
}
