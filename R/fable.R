# ADAPTER: the tidyverts model definition. Not part of the engine core.
#
# `TSFM()` lets a foundation model be composed inside `fabletools::model()`
# alongside ARIMA, ETS, and friends, so it can be compared and reconciled with
# the rest of the tidyverts stack.
#
# There are two routes onto the same engine, and they trade convenience for
# throughput:
#
#   * `TSFM()` --- fabletools trains and forecasts **one key at a time**, so the
#     engine sees a batch of one per series. It is the composable route.
#   * `forecast(model, panel, h) |> fabletools::as_fable()` --- one batched call
#     across every series. It is the fast route.
#
# `TSFM()` calls `zuk_pretrained(reuse = TRUE)` per key, so the checkpoint is
# constructed once and served from the resident LRU for every remaining key.
# Without that reuse a 200-key panel would load a 925 MB checkpoint 200 times.

# The model class is built lazily: fabletools is an optional adapter dependency,
# so it must not be needed to install or load zukzeit. Memoised because the class
# object is immutable and identical across calls --- dispatch happens on the
# fitted object's own class, not on this generator.
.zuk_fable_class <- new.env(parent = emptyenv())

zuk_fable_model_class <- function() {
  if (!is.null(.zuk_fable_class$class)) {
    return(.zuk_fable_class$class)
  }
  cls <- fabletools::new_model_class(
    "TSFM",
    train = train_tsfm,
    specials = fabletools::new_specials()
  )
  .zuk_fable_class$class <- cls
  cls
}

# fabletools calls this once per key, with `.data` reduced to the index and the
# single response column.
train_tsfm <- function(.data, specials, model_id, revision = NULL,
                       quantile_levels = NULL, device = NULL, reuse = TRUE,
                       ...) {
  zuk_require_namespace(
    "tsibble",
    reason = "It is needed to read the response from a tsibble."
  )
  response <- tsibble::measured_vars(.data)
  if (length(response) != 1L) {
    zuk_abort_capability(
      "Contract v1 supports exactly one response column; got {length(response)}.",
      model_id = model_id,
      capability = "multivariate",
      requested = length(response),
      supported = 1L
    )
  }
  y <- .data[[response]]
  if (!is.numeric(y)) {
    zuk_abort_capability(
      "Response {.val {response}} must be numeric; got {.cls {class(y)}}.",
      model_id = model_id,
      capability = "target_type",
      requested = class(y),
      supported = "numeric"
    )
  }

  model <- zuk_pretrained(
    model_id,
    revision = revision,
    device = device,
    reuse = reuse,
    ...
  )
  levels <- check_quantile_levels(
    model$capabilities,
    quantile_levels %||% c(0.1, 0.5, 0.9),
    model$model_id,
    model$revision
  )

  structure(
    list(
      model           = model,
      history         = as.numeric(y),
      response        = response,
      index           = tsibble::index_var(.data),
      quantile_levels = levels
    ),
    class = "model_tsfm"
  )
}

#' Compose a foundation model inside a tidyverts model definition
#'
#' A [fabletools::new_model_definition()] wrapper so a pretrained checkpoint can
#' be used inside `fabletools::model()` next to any other tidyverts model. The
#' model is zero-shot: "training" binds the observed history for a key, and
#' `fabletools::forecast()` runs the engine over it.
#'
#' @section Per key, not batched:
#' fabletools evaluates one key at a time, so `TSFM()` runs the engine once per
#' series. The checkpoint itself is loaded once and reused from the bounded
#' resident cache (see [zuk_pretrained()] and [zuk_cache_status()]), but the
#' forward passes are not batched across series. For a wide panel, prefer the
#' batched route:
#'
#' ```r
#' forecast(model, panel, h = 12) |> fabletools::as_fable()
#' ```
#'
#' @section What `.mean` holds:
#' Every route carries the same predictive distribution, but they label the
#' point forecast differently. `fabletools` derives `.mean` from the
#' distribution, so a `TSFM()` fable reports the distribution's **mean**;
#' [forecast.zuk_model()] and [fabletools::as_fable()] report the engine's
#' exactly evaluated **median**. With few quantile levels the two diverge on a
#' skewed forecast. `median()` of the distribution recovers the engine's point
#' forecast on any route:
#'
#' ```r
#' fc <- fabletools::forecast(fits, h = 12)
#' median(fc$value)   # the engine's point forecast
#' ```
#'
#' @param formula A model formula naming the response, for example `value`.
#' @param model_id Checkpoint id, as listed by [zuk_models()]. Use `"stub"` for
#'   the weight-free test fixture.
#' @param revision `NULL` for the catalogue's immutable revision, or that exact
#'   revision string.
#' @param quantile_levels Quantile levels to forecast; defaults to
#'   `c(0.1, 0.5, 0.9)`.
#' @param device Device requested for construction; see [zuk_resolve_device()].
#' @param ... Further load-affecting arguments passed to [zuk_pretrained()].
#'
#' @return A `fabletools` model definition.
#' @seealso [forecast.zuk_model()] for the batched panel route.
#' @export
#' @examplesIf requireNamespace("fabletools", quietly = TRUE) && requireNamespace("tsibble", quietly = TRUE)
#' history <- tsibble::tsibble(time = 1:12, value = cumsum(1:12), index = time)
#' fits <- fabletools::model(history, zukzeit = TSFM(value, model_id = "stub"))
#' fabletools::forecast(fits, h = 3)
#' zuk_unload("stub")
TSFM <- function(formula, model_id, revision = NULL, quantile_levels = NULL,
                 device = NULL, ...) {
  zuk_require_namespace(
    c("fabletools", "tsibble"),
    reason = "They provide the tidyverts model interface (the engine itself does not need them)."
  )
  fabletools::new_model_definition(
    zuk_fable_model_class(),
    !!rlang::enquo(formula),
    model_id = model_id,
    revision = revision,
    quantile_levels = quantile_levels,
    device = device,
    ...
  )
}

#' Forecast from a composed tidyverts foundation model
#'
#' The [generics::forecast()] method for a `TSFM()` fit. Returns a
#' `distributional` vector of predictive quantiles, one element per row of
#' `new_data`, which is what `fabletools` assembles into a fable.
#'
#' @param object A fitted `model_tsfm`.
#' @param new_data Future rows supplied by `fabletools`.
#' @param specials Unused; supplied by `fabletools`.
#' @param ... Unused.
#' @return A `distributional` vector.
#' @export
#' @examplesIf requireNamespace("fabletools", quietly = TRUE) && requireNamespace("tsibble", quietly = TRUE)
#' history <- tsibble::tsibble(time = 1:24, value = cumsum(1:24), index = time)
#' fits <- fabletools::model(history, zukzeit = TSFM(value, model_id = "stub"))
#'
#' # fabletools calls this method one key at a time; it returns the predictive
#' # distributions it then assembles into a fable.
#' fabletools::forecast(fits, h = 3)
#'
#' zuk_unload("stub")
forecast.model_tsfm <- function(object, new_data, specials = NULL, ...) {
  h <- nrow(new_data)
  if (h == 0L) {
    return(build_distribution(
      matrix(numeric(0), ncol = length(object$quantile_levels)),
      object$quantile_levels
    ))
  }
  qmat <- zuk_run_batches(
    object$model,
    list(object$history),
    h,
    object$quantile_levels
  )[[1]]
  build_distribution(qmat, object$quantile_levels)
}

# A zero-shot model is never fitted in sample, so there are no fitted values or
# residuals to report. Returning NA is the honest answer: fabletools can still
# assemble augment() output, and out-of-sample accuracy() against held-out data
# --- which is what these models are evaluated with --- is unaffected.
#' @export
fitted.model_tsfm <- function(object, ...) {
  rep(NA_real_, length(object$history))
}

#' @export
residuals.model_tsfm <- function(object, ...) {
  rep(NA_real_, length(object$history))
}

# Registered onto fabletools' generic from .onLoad(); see zzz.R.
model_sum.model_tsfm <- function(x) {
  sprintf("TSFM[%s]", x$model$architecture)
}

#' @export
print.model_tsfm <- function(x, ...) {
  cli::cli_text("{.cls model_tsfm} {.strong {x$model$architecture}}")
  cli::cli_text(
    "model_id: {.val {x$model$model_id}}  revision: {.val {x$model$revision}}"
  )
  cli::cli_text(
    "response {.val {x$response}} over {.val {length(x$history)}} observations"
  )
  cli::cli_text("quantile levels: {.val {x$quantile_levels}}")
  invisible(x)
}
