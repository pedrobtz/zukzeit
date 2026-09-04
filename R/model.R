# A `zuk_model` is the uniform handle returned by `zuk_pretrained()`. It wraps
# whatever executes a forward pass (`predict_fn`) together with the metadata the
# rest of the package needs to dispatch, validate, and report. Every
# architecture — the Stage 0 stub, and the native torch modules that follow —
# produces one of these, so the fit/forecast layers never branch on architecture.

#' Construct a model handle
#'
#' @param architecture Character scalar architecture key.
#' @param config Parsed configuration list (from `config.json` or synthesised
#'   for the stub).
#' @param capabilities A [new_zuk_capabilities()] object.
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
#'   per-series quantile matrices. When `NULL`, [zuk_run_batches()] falls back
#'   to looping `predict_fn`. Native torch architectures supply this; the stub
#'   does not.
#' @param contract_version The architecture-contract version this model was
#'   written against; see `?`[zuk-architecture-contract]. Defaults to the
#'   version this installation implements.
#' @return A `zuk_model` object.
#' @seealso `?`[zuk-architecture-contract] for the full specification, and
#'   [zuk_check_architecture()] to verify an implementation against it.
#' @export
#' @examples
#' # The smallest conforming architecture: forecast the last observed value,
#' # with a Gaussian spread across the requested quantile levels.
#' model <- new_zuk_model(
#'   architecture = "demo",
#'   config = list(),
#'   capabilities = new_zuk_capabilities("demo", max_context = 128L),
#'   predict_fn = function(context, h, quantile_levels) {
#'     if (length(context) == 0L) stop("empty context")
#'     last <- context[length(context)]
#'     outer(rep(last, h), stats::qnorm(quantile_levels), `+`)
#'   }
#' )
#' model
#' model$predict_fn(c(1, 2, 3), h = 2L, quantile_levels = c(0.1, 0.5, 0.9))
new_zuk_model <- function(architecture,
                           config,
                           capabilities,
                           predict_fn,
                           model_id = NA_character_,
                           revision = NA_character_,
                           device = config$device %||% "cpu",
                           params = NULL,
                           predict_batch_fn = NULL,
                           contract_version = zuk_contract_version()) {
  if (!inherits(capabilities, "zuk_capabilities")) {
    zuk_abort_contract(
      "{.arg capabilities} must be a {.cls zuk_capabilities} object.",
      architecture = architecture,
      model_id = model_id,
      contract = "model construction",
      expected = "zuk_capabilities",
      actual = class(capabilities)
    )
  }
  if (!is.function(predict_fn)) {
    zuk_abort_contract(
      "{.arg predict_fn} must be a function.",
      architecture = architecture,
      model_id = model_id,
      contract = "model construction",
      expected = "function(context, h, quantile_levels)",
      actual = class(predict_fn)
    )
  }
  if (!is.null(predict_batch_fn) && !is.function(predict_batch_fn)) {
    zuk_abort_contract(
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
    class = "zuk_model"
  )
}

#' @export
zuk_capabilities.zuk_model <- function(x, ...) {
  x$capabilities
}

#' @export
print.zuk_model <- function(x, ...) {
  cli::cli_text("{.cls zuk_model} {.strong {x$architecture}}")
  cli::cli_text("model_id: {.val {x$model_id}}  revision: {.val {x$revision}}")
  cli::cli_text("device: {.val {x$device}}")
  print(x$capabilities)
  invisible(x)
}
