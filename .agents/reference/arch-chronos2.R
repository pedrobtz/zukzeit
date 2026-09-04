# Chronos-2 --- unregistered reference adapter.
#
# This bridge predates the 0.1.0 native-engine scope. It is intentionally not
# registered: its output conversion and per-series model construction have not
# passed contract conformance or numerical parity. Keep it only as prior art for
# a future native port after contract v2 can represent Chronos-2 inputs.

chronos2_capabilities <- function(config) {
  new_zuk_capabilities(
    architecture      = "chronos2",
    max_context       = as.integer(config$max_context %||% 8192L),
    quantiles         = "native",
    multivariate      = FALSE,
    samples           = FALSE,
    past_covariates   = FALSE,
    future_covariates = FALSE,
    static_covariates = FALSE,
    fine_tunable      = FALSE,
    license           = "Apache-2.0"
  )
}

# Reference-only forecast bridge. Builds a minimal data frame
# from the raw context (synthetic integer time index), asks brulee for `h`
# steps at the requested quantile levels, and returns an h x length(levels)
# matrix. brulee's precise argument names are resolved defensively so this keeps
# working across brulee versions; validated in the brulee-gated tests.
chronos2_forecast_series <- function(context, h, quantile_levels, config) {
  zuk_require_namespace(
    "brulee",
    reason = "to inspect the unregistered Chronos-2 reference adapter."
  )
  train <- data.frame(
    .t = seq_along(context),
    .y = as.numeric(context)
  )
  new_data <- data.frame(.t = length(context) + seq_len(h))

  fit <- brulee::brulee_chronos(
    .y ~ .t,
    data = train,
    quantile_levels = quantile_levels
  )
  pred <- stats::predict(fit, new_data = new_data, type = "quantile")

  as_quantile_matrix(pred, h = h, quantile_levels = quantile_levels)
}

# Coerce whatever shape brulee returns (a tibble of quantile columns, or a
# list-column of quantiles) into a plain h x length(levels) matrix.
as_quantile_matrix <- function(pred, h, quantile_levels) {
  q <- length(quantile_levels)
  if (is.data.frame(pred)) {
    qcols <- grep("^\\.pred", names(pred), value = TRUE)
    if (length(qcols) == q) {
      return(matrix(as.numeric(as.matrix(pred[qcols])), nrow = h, ncol = q))
    }
  }
  m <- matrix(as.numeric(unlist(pred)), nrow = h, ncol = q, byrow = TRUE)
  m
}

chronos2_constructor <- function(config, weights = NULL) {
  caps <- chronos2_capabilities(config)
  predict_fn <- function(context, h, quantile_levels) {
    chronos2_forecast_series(context, h, quantile_levels, config)
  }
  new_zuk_model(
    architecture = "chronos2",
    config       = config,
    capabilities = caps,
    predict_fn   = predict_fn,
    model_id     = config$model_id %||% "amazon/chronos-2",
    revision     = config$revision %||% NA_character_,
    params       = NULL
  )
}
