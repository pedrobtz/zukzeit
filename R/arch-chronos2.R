# Portions derived from Chronos-2, Copyright Amazon.com, Inc. or its affiliates.
# Translated and modified for R/torch by the zukzeit authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Native Chronos-2 architecture entry point. The first architecture written
# against contract 1.1, so the first that can be handed grouped inputs.

chronos2_config_value <- function(config, name, default) {
  value <- config[[name]] %||% default
  if (length(value) != 1L || !is.numeric(value) || is.na(value) ||
      !is.finite(value) || value <= 0 || value != floor(value)) {
    zuk_abort_checkpoint(
      "Chronos-2 config field {.val {name}} must be one positive integer.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      tensor = "config.json", expected = "one positive integer", actual = value
    )
  }
  as.integer(value)
}

chronos2_quantiles <- function() {
  c(0.01, seq(0.05, 0.95, by = 0.05), 0.99)
}

validate_chronos2_config <- function(config) {
  actual <- list(
    d_model = chronos2_config_value(config, "d_model", 768L),
    d_ff = chronos2_config_value(config, "d_ff", 3072L),
    d_kv = chronos2_config_value(config, "d_kv", 64L),
    num_heads = chronos2_config_value(config, "num_heads", 12L),
    num_layers = chronos2_config_value(config, "num_layers", 12L),
    input_patch_size = chronos2_config_value(config, "input_patch_size", 16L),
    output_patch_size = chronos2_config_value(config, "output_patch_size", 16L),
    input_patch_stride = chronos2_config_value(config, "input_patch_stride", 16L),
    context_length = chronos2_config_value(config, "context_length", 8192L),
    max_output_patches = chronos2_config_value(config, "max_output_patches", 64L)
  )
  expected <- list(
    d_model = 768L, d_ff = 3072L, d_kv = 64L, num_heads = 12L, num_layers = 12L,
    input_patch_size = 16L, output_patch_size = 16L, input_patch_stride = 16L,
    context_length = 8192L, max_output_patches = 64L
  )
  # These change the arithmetic without changing a shape, so they are checked
  # rather than assumed: a gated activation, a missing register token, or
  # scaling without arcsinh would all load cleanly and compute the wrong answer.
  flags <- list(
    relu_activation = identical(as.character(config$dense_act_fn %||% ""), "relu"),
    not_gated = isFALSE(config$is_gated_act %||% TRUE),
    use_arcsinh = isTRUE(config$use_arcsinh %||% FALSE),
    use_reg_token = isTRUE(config$use_reg_token %||% FALSE)
  )
  wrong <- names(expected)[vapply(
    names(expected),
    function(name) !identical(actual[[name]], expected[[name]]), logical(1)
  )]
  bad_flags <- names(flags)[!vapply(flags, isTRUE, logical(1))]
  quantiles <- as.numeric(config$quantiles %||% chronos2_quantiles())
  if (length(wrong) || length(bad_flags) ||
      !isTRUE(all.equal(quantiles, chronos2_quantiles()))) {
    zuk_abort_checkpoint(
      c(
        "This Chronos-2 checkpoint variant is not compatible with the native port.",
        "i" = "Only the pinned Chronos-2 configuration is supported."
      ),
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      tensor = "config.json",
      expected = c(expected, list(quantiles = chronos2_quantiles())),
      actual = c(actual, list(quantiles = quantiles, flags = bad_flags))
    )
  }
  invisible(config)
}

chronos2_capabilities <- function(config) {
  new_zuk_capabilities(
    architecture      = "chronos2",
    max_context       = as.integer(config$context_length),
    # The head emits `max_output_patches` patches in one pass, so the horizon
    # ceiling is a property of the checkpoint rather than a policy choice.
    max_horizon       = as.integer(config$max_output_patches) *
                          as.integer(config$output_patch_size),
    quantiles         = "native",
    quantile_levels   = as.numeric(config$quantiles %||% chronos2_quantiles()),
    # The first architecture to declare these. Every series is a row and
    # `group_ids` ties rows into a task, so multivariate targets and covariates
    # are one mechanism rather than three.
    multivariate      = TRUE,
    past_covariates   = TRUE,
    future_covariates = TRUE,
    samples           = FALSE,
    static_covariates = FALSE,
    fine_tunable      = FALSE,
    license           = "Apache-2.0"
  )
}

chronos2_constructor <- function(config, weights = NULL) {
  zuk_require_namespace("torch", reason = "It is needed for native Chronos-2 inference.")
  # The Hub config nests the forecasting settings; flatten them so one record
  # describes the checkpoint.
  if (!is.null(config$chronos_config)) {
    nested <- config$chronos_config
    config$chronos_config <- NULL
    config <- utils::modifyList(config, nested)
  }
  config$quantiles <- as.numeric(config$quantiles %||% chronos2_quantiles())
  config$time_encoding_scale <- as.numeric(
    config$time_encoding_scale %||% config$context_length
  )
  validate_chronos2_config(config)
  if (is.null(weights)) {
    zuk_abort_checkpoint(
      "Native Chronos-2 construction requires checkpoint weights.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      expected = "a complete named state dict", actual = NULL
    )
  }
  module <- chronos2_module(config)
  chronos2_load_module_weights(module, weights, config)
  device <- config$device %||% "cpu"
  module$to(device = torch::torch_device(device))
  module$eval()

  predict_batch <- function(contexts, horizons, quantile_levels,
                            device = config$device, groups = NULL) {
    if (!identical(as.character(device), as.character(config$device))) {
      zuk_abort_device(
        c(
          "Inference device {.val {device}} differs from the loaded Chronos-2 handle.",
          "i" = "Load a separate handle with {.code zuk_pretrained(..., device = {device})}."
        ),
        requested_device = device, resolved_device = config$device
      )
    }
    chronos2_predict_batch(module, contexts, horizons, quantile_levels, config,
                           device, groups)
  }
  predict_one <- function(context, h, quantile_levels) {
    predict_batch(list(context), as.integer(h), quantile_levels, device)[[1]]
  }

  new_zuk_model(
    architecture = "chronos2",
    config = config,
    capabilities = chronos2_capabilities(config),
    predict_fn = predict_one,
    predict_batch_fn = predict_batch,
    model_id = config$model_id %||% NA_character_,
    revision = config$revision %||% NA_character_,
    device = device,
    params = module,
    contract_version = "1.1.0"
  )
}
