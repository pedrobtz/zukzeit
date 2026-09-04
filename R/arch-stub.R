# Stage 0 stub architecture.
#
# A weight-free, torch-free forecaster used to exercise the whole contract
# (loader -> capabilities -> fit bridge -> batched inference -> forecast
# object -> fable/yardstick adapters) before any native model exists. It is a
# random-walk model: the point forecast is the last observed value and the
# predictive spread grows with the square root of the horizon, scaled by the
# empirical standard deviation of the context's first differences. This yields
# honest, monotone quantiles without depending on torch, so Stage 0 tests run
# anywhere.

stub_predict_batch <- function(context, h, quantile_levels) {
  context <- as.numeric(context)
  context <- context[!is.na(context)]
  h <- as.integer(h)

  if (length(context) == 0L) {
    tsfm_abort_capability(
      "The stub model requires at least one observed value of context.",
      model_id = "stub",
      capability = "context",
      requested = 0L,
      supported = "at least one observation"
    )
  }

  last <- context[length(context)]

  # Per-step standard deviation from first differences; fall back sensibly when
  # the context is too short or perfectly flat.
  step_sd <- if (length(context) >= 2L) stats::sd(diff(context)) else 0
  if (!is.finite(step_sd) || step_sd == 0) {
    step_sd <- max(abs(last) * 1e-6, .Machine$double.eps)
  }

  horizon_sd <- step_sd * sqrt(seq_len(h))            # length h
  z <- stats::qnorm(quantile_levels)                  # length q

  # Outer product: rows = horizons, cols = quantile levels.
  preds <- outer(horizon_sd, z) + last
  matrix(preds, nrow = h, ncol = length(quantile_levels))
}

stub_constructor <- function(config, weights = NULL) {
  max_context <- as.integer(config$max_context %||% 512L)
  caps <- new_tsfm_capabilities(
    architecture = "stub",
    max_context  = max_context,
    quantiles    = "native",
    quantile_levels = NULL,
    multivariate = FALSE,
    samples      = FALSE,
    fine_tunable = FALSE,
    license      = "MIT"
  )
  new_tsfm_model(
    architecture = "stub",
    config       = config,
    capabilities = caps,
    predict_fn   = stub_predict_batch,
    model_id     = config$model_id %||% "stub",
    revision     = config$revision %||% NA_character_,
    params       = NULL
  )
}
