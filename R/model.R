# A `tsfm_model` is the uniform handle returned by `tsfm_pretrained()`. It wraps
# whatever executes a forward pass (`predict_fn`) together with the metadata the
# rest of the package needs to dispatch, validate, and report. Every
# architecture — the Stage 0 stub, and the native torch modules that follow —
# produces one of these, so the fit/forecast layers never branch on architecture.

#' Construct a model handle
#'
#' @param architecture Character scalar architecture key.
#' @param config Parsed configuration list (from `config.json` or synthesised
#'   for the stub).
#' @param capabilities A [new_tsfm_capabilities()] object.
#' @param predict_fn A function `function(context, h, quantile_levels)` that
#'   forecasts a single numeric series. `context` is the observed history (a
#'   numeric vector, oldest first); it returns a numeric matrix with `h` rows
#'   and `length(quantile_levels)` columns of predictive quantiles.
#' @param model_id,revision Provenance of the checkpoint.
#' @param device Resolved execution device attached to the handle.
#' @param params Optional opaque parameter/state object (e.g. a torch module);
#'   `NULL` for the weight-free stub.
#' @param predict_batch_fn Optional vectorised forward pass for true batched
#'   inference: `function(contexts, horizons, quantile_levels, device)` where
#'   `contexts`/`horizons` are lists aligned by series, returning a list of
#'   per-series quantile matrices. When `NULL`, [tsfm_run_batches()] falls back
#'   to looping `predict_fn`. Native torch architectures supply this; the stub
#'   does not.
#' @param contract_version The architecture-contract version this model was
#'   written against; see `?`[tsfm-architecture-contract]. Defaults to the
#'   version this installation implements.
#' @return A `tsfm_model` object.
#' @seealso `?`[tsfm-architecture-contract] for the full specification, and
#'   [tsfm_check_architecture()] to verify an implementation against it.
#' @export
#' @examples
#' # The smallest conforming architecture: forecast the last observed value,
#' # with a Gaussian spread across the requested quantile levels.
#' model <- new_tsfm_model(
#'   architecture = "demo",
#'   config = list(),
#'   capabilities = new_tsfm_capabilities("demo", max_context = 128L),
#'   predict_fn = function(context, h, quantile_levels) {
#'     if (length(context) == 0L) stop("empty context")
#'     last <- context[length(context)]
#'     outer(rep(last, h), stats::qnorm(quantile_levels), `+`)
#'   }
#' )
#' model
#' model$predict_fn(c(1, 2, 3), h = 2L, quantile_levels = c(0.1, 0.5, 0.9))
new_tsfm_model <- function(architecture,
                           config,
                           capabilities,
                           predict_fn,
                           model_id = NA_character_,
                           revision = NA_character_,
                           device = config$device %||% "cpu",
                           params = NULL,
                           predict_batch_fn = NULL,
                           contract_version = tsfm_contract_version()) {
  if (!inherits(capabilities, "tsfm_capabilities")) {
    tsfm_abort_contract(
      "{.arg capabilities} must be a {.cls tsfm_capabilities} object.",
      architecture = architecture,
      model_id = model_id,
      contract = "model construction",
      expected = "tsfm_capabilities",
      actual = class(capabilities)
    )
  }
  if (!is.function(predict_fn)) {
    tsfm_abort_contract(
      "{.arg predict_fn} must be a function.",
      architecture = architecture,
      model_id = model_id,
      contract = "model construction",
      expected = "function(context, h, quantile_levels)",
      actual = class(predict_fn)
    )
  }
  if (!is.null(predict_batch_fn) && !is.function(predict_batch_fn)) {
    tsfm_abort_contract(
      "{.arg predict_batch_fn} must be a function or {.code NULL}.",
      architecture = architecture,
      model_id = model_id,
      contract = "model construction",
      expected = "function or NULL",
      actual = class(predict_batch_fn)
    )
  }
  structure(
    list(
      architecture     = as.character(architecture),
      config           = config,
      capabilities     = capabilities,
      predict_fn       = predict_fn,
      predict_batch_fn = predict_batch_fn,
      model_id         = as.character(model_id),
      revision         = as.character(revision),
      device           = as.character(device),
      params           = params,
      contract_version = package_version(contract_version)
    ),
    class = "tsfm_model"
  )
}

#' @export
tsfm_capabilities.tsfm_model <- function(x, ...) {
  x$capabilities
}

#' @export
print.tsfm_model <- function(x, ...) {
  cli::cli_text("{.cls tsfm_model} {.strong {x$architecture}}")
  cli::cli_text("model_id: {.val {x$model_id}}  revision: {.val {x$revision}}")
  cli::cli_text("device: {.val {x$device}}")
  print(x$capabilities)
  invisible(x)
}
