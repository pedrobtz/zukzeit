# Portions derived from TimesFM, Copyright 2025 Google LLC.
# Translated and modified for R/torch by the tsfm authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Native TimesFM 2.5 architecture entry point.

timesfm_config_value <- function(config, name, default) {
  value <- config[[name]] %||% default
  if (length(value) != 1L || !is.numeric(value) || is.na(value) ||
      !is.finite(value) || value <= 0 || value != floor(value)) {
    tsfm_abort_checkpoint(
      "TimesFM config field {.val {name}} must be one positive integer.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      tensor = "config.json",
      expected = "one positive integer",
      actual = value
    )
  }
  as.integer(value)
}

validate_timesfm_config <- function(config) {
  actual <- list(
    hidden_size = timesfm_config_value(config, "hidden_size", 1280L),
    intermediate_size = timesfm_config_value(config, "intermediate_size", 1280L),
    head_dim = timesfm_config_value(config, "head_dim", 80L),
    num_attention_heads = timesfm_config_value(config, "num_attention_heads", 16L),
    num_hidden_layers = timesfm_config_value(config, "num_hidden_layers", 20L),
    patch_length = timesfm_config_value(config, "patch_length", 32L),
    horizon_length = timesfm_config_value(config, "horizon_length", 128L),
    quantile_horizon_length = timesfm_config_value(
      config, "quantile_horizon_length", 1024L
    ),
    context_length = timesfm_config_value(config, "context_length", 16384L)
  )
  expected <- list(
    hidden_size = 1280L,
    intermediate_size = 1280L,
    head_dim = 80L,
    num_attention_heads = 16L,
    num_hidden_layers = 20L,
    patch_length = 32L,
    horizon_length = 128L,
    quantile_horizon_length = 1024L,
    context_length = 16384L
  )
  quantiles <- as.numeric(config$quantiles %||% seq(0.1, 0.9, by = 0.1))
  wrong <- names(expected)[vapply(
    names(expected),
    function(name) !identical(actual[[name]], expected[[name]]),
    logical(1)
  )]
  if (length(wrong) || !isTRUE(all.equal(quantiles, seq(0.1, 0.9, by = 0.1))) ||
      !isTRUE(all.equal(as.numeric(config$rms_norm_eps %||% 1e-6), 1e-6))) {
    tsfm_abort_checkpoint(
      c(
        "This TimesFM checkpoint variant is not compatible with the native port.",
        "i" = "Only the pinned TimesFM 2.5 200M configuration is supported."
      ),
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      tensor = "config.json",
      expected = c(expected, list(quantiles = seq(0.1, 0.9, by = 0.1), rms_norm_eps = 1e-6)),
      actual = c(actual, list(quantiles = quantiles, rms_norm_eps = config$rms_norm_eps %||% 1e-6))
    )
  }
  invisible(config)
}

timesfm_capabilities <- function(config) {
  # max_context is the best case: the checkpoint's 16,384 positions cover the
  # context *and* the forecast, and every request consumes at least one complete
  # 128-value output block. Longer horizons round up to further whole blocks, so
  # the history actually used is `context_length - round_up(h, horizon_length)`
  # --- see timesfm_usable_context(), which the forward pass applies per series.
  # At the maximum horizon of 1,024 that leaves 15,360 observations.
  context_limit <- as.integer(config$context_length %||% 16384L)
  output_patch <- as.integer(config$horizon_length %||% 128L)
  new_tsfm_capabilities(
    architecture      = "timesfm",
    max_context       = context_limit - output_patch,
    max_horizon       = as.integer(config$quantile_horizon_length %||% 1024L),
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

# Checkpoint and R module paths intentionally match. Keeping the map explicit
# makes unsupported checkpoint variants diagnosable before tensor assignment.
timesfm_weight_map <- function(config) {
  expected <- timesfm_expected_state_spec(config)
  stats::setNames(names(expected), names(expected))
}

timesfm_load_module_weights <- function(module, weights, config) {
  if (!is.list(weights) || is.null(names(weights))) {
    tsfm_abort_checkpoint(
      "TimesFM requires a complete named state dict.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      expected = names(timesfm_weight_map(config)),
      actual = names(weights)
    )
  }
  map <- timesfm_weight_map(config)
  missing <- setdiff(names(map), names(weights))
  unexpected <- setdiff(names(weights), names(map))
  module_names <- names(module$state_dict())
  if (length(missing) || length(unexpected) ||
      !setequal(unname(map), module_names)) {
    tsfm_abort_checkpoint(
      "TimesFM checkpoint tensors do not map exactly onto the native module.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      expected = module_names,
      actual = names(weights)
    )
  }
  mapped <- weights[names(map)]
  names(mapped) <- unname(map)
  # The module is created with meta-device placeholders, so assignment adopts
  # the safetensors storage rather than allocating and retaining a second copy.
  module$load_state_dict(mapped, .refer_to_state_dict = TRUE)
  invisible(module)
}

timesfm_constructor <- function(config, weights = NULL) {
  tsfm_require_namespace("torch", reason = "It is needed for native TimesFM inference.")
  validate_timesfm_config(config)
  if (is.null(weights)) {
    tsfm_abort_checkpoint(
      "Native TimesFM construction requires checkpoint weights.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      expected = "a complete named state dict",
      actual = NULL
    )
  }
  module <- timesfm_module(config)
  timesfm_load_module_weights(module, weights, config)
  device <- config$device %||% "cpu"
  module$to(device = torch::torch_device(device))
  module$eval()
  caps <- timesfm_capabilities(config)

  # Default to the handle's own device. Writing `device = device` here would be
  # a recursive default argument reference: omitting the argument raises an
  # opaque promise error instead of using the device the module was loaded onto.
  predict_batch <- function(contexts, horizons, quantile_levels,
                            device = config$device) {
    if (!identical(as.character(device), as.character(config$device))) {
      tsfm_abort_device(
        c(
          "Inference device {.val {device}} differs from the loaded TimesFM handle.",
          "i" = "Load a separate handle with {.code tsfm_pretrained(..., device = {device})}."
        ),
        requested_device = device,
        resolved_device = config$device
      )
    }
    timesfm_predict_batch(
      module, contexts, horizons, quantile_levels, config, device
    )
  }
  predict_one <- function(context, h, quantile_levels) {
    predict_batch(list(context), as.integer(h), quantile_levels, device)[[1]]
  }

  new_tsfm_model(
    architecture = "timesfm",
    config = config,
    capabilities = caps,
    predict_fn = predict_one,
    predict_batch_fn = predict_batch,
    model_id = config$model_id %||% NA_character_,
    revision = config$revision %||% NA_character_,
    device = device,
    params = module
  )
}
