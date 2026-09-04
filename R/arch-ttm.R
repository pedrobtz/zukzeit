# TinyTimeMixer (TTM) --- deferred scaffold.
#
# STATUS: scaffold. Capabilities, config parsing, registry dispatch, and the
# weight-map contract are in place; the numerical forward pass and the
# safetensors -> nn_module state-dict mapping land once golden parity fixtures
# are generated. That generation requires resources unavailable in a sandboxed
# CI-only setting (the real ibm-granite/granite-timeseries-ttm-r2 checkpoint
# from the Hub, R torch with a libtorch backend, and the reference `granite-zukzeit`
# Python implementation to produce expected outputs). Until then the forward
# pass errors with a clear pointer rather than returning unverified numbers.
#
# The selected TTM-R2 checkpoint is point-forecasting, while contract v1
# requires predictive quantiles. Keep the architecture sketch for a later
# point-output contract; it is not on the 0.1.0 release path.
#
# ---- Weight-map contract ----------------------------------------------------
#
# The loader maps safetensors tensor names from the checkpoint to R torch module
# parameters. TTM's published structure (granite-zukzeit `TinyTimeMixerForPrediction`)
# is, at a high level:
#
#   backbone.encoder.patcher            patch embedding (Linear over patch_length)
#   backbone.encoder.mixers.{i}.*       N mixer blocks, each:
#       .norm                             LayerNorm / BatchNorm (config: norm_mlp)
#       .mlp_time / .mlp_feature          gated MLP mixing across time / channels
#       .gating_block                     optional gated attention weights
#   decoder.*                           lightweight decoder mixers (mirrors encoder)
#   head.base_forecast_block            Linear projection to prediction_length
#   (optional) head.quantile_*          quantile projection when config has it
#
# The exact tensor names, mixer count, patch/stride, normalisation type, and
# whether channel mixing / gating are enabled are all read from config.json.
# `ttm_weight_map()` below is the single place that translates those names; it
# is intentionally isolated so the port is a matter of filling one function plus
# the nn_module, with the rest of the package (loader, batching, forecast
# object, parsnip) already wired.

ttm_capabilities <- function(config) {
  new_zuk_capabilities(
    architecture      = "ttm",
    max_context       = as.integer(config$context_length %||% config$seq_len %||% 512L),
    quantiles         = "none",
    quantile_levels   = numeric(),
    multivariate      = FALSE,
    samples           = FALSE,
    past_covariates   = FALSE,
    future_covariates = FALSE,
    static_covariates = FALSE,
    fine_tunable      = FALSE,
    license           = "Apache-2.0"
  )
}

# Translate checkpoint tensor names -> module parameter paths. Returns a named
# character vector (checkpoint name -> module path). Filled during the numerical
# port; kept here so the mapping lives in exactly one place.
ttm_weight_map <- function(config) {
  zuk_abort_checkpoint(c(
    "The TTM weight map is not implemented yet.",
    "i" = "It is derived from the checkpoint's {.file config.json} at port time."
  ),
  model_id = config$model_id %||% "ibm-granite/granite-timeseries-ttm-r2",
  revision = config$revision %||% NA_character_,
  expected = "complete state-dict mapping",
  actual = "scaffold")
}

# Build the R torch nn_module for TTM from a parsed config. Deferred to the
# numerical port (needs torch); isolated so only this + ttm_weight_map change.
ttm_module <- function(config) {
  zuk_require_namespace("torch", reason = "It is needed to build the TTM network.")
  zuk_abort_checkpoint(
    "The native TTM nn_module is deferred until a point-output contract exists.",
    model_id = config$model_id %||% "ibm-granite/granite-timeseries-ttm-r2",
    revision = config$revision %||% NA_character_,
    expected = "checkpoint-compatible nn_module",
    actual = "scaffold"
  )
}

ttm_constructor <- function(config, weights = NULL) {
  caps <- ttm_capabilities(config)

  not_ready <- function(...) {
    zuk_abort_capability(c(
      "The native TTM forward pass is not implemented.",
      "i" = "Numerical parity against {.val ibm-granite/granite-timeseries-ttm-r2} \\
             requires golden fixtures generated with the Hub checkpoint, torch, \\
             and reference {.pkg granite-zukzeit}.",
      "i" = "TTM is deferred until the engine can represent point-only output.",
      "i" = "Use {.code zuk_pretrained(\"stub\")} to exercise the engine shell."
    ),
    model_id = config$model_id %||% "ibm-granite/granite-timeseries-ttm-r2",
    revision = config$revision %||% NA_character_,
    capability = "model_state",
    requested = "scaffold",
    supported = "supported")
  }

  new_zuk_model(
    architecture = "ttm",
    config       = config,
    capabilities = caps,
    predict_fn   = not_ready,
    model_id     = config$model_id %||% NA_character_,
    revision     = config$revision %||% NA_character_,
    params       = weights
  )
}
