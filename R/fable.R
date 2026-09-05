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
    # `xreg()` keeps fable's meaning exactly: a regressor whose future values
    # the caller supplies. fabletools re-evaluates specials against `new_data`
    # at forecast time, so those values arrive without plumbing of our own.
    # `past_xreg()` has no tidyverts precedent --- no fable model distinguishes
    # the two kinds --- so it is named unlike anything fable defines.
    specials = fabletools::new_specials(
      # `.named = TRUE` deparses each bare argument into a name, so a regressor
      # can be matched between fit and forecast time. Without it the values
      # arrive correctly but anonymously, and nothing can be paired up.
      xreg = function(...) {
        lapply(rlang::enexprs(..., .named = TRUE), rlang::eval_tidy,
               data = self$data)
      },
      past_xreg = function(...) {
        # fabletools re-evaluates every special against `new_data` at forecast
        # time, and a past-only regressor is by definition absent there. Not
        # finding it is the expected case, not a failure.
        lapply(rlang::enexprs(..., .named = TRUE), function(expression) {
          tryCatch(rlang::eval_tidy(expression, data = self$data),
                   error = function(e) NULL)
        })
      },
      .required_specials = NULL
    )
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
      quantile_levels = levels,
      covariates      = fable_covariates(specials, model)
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
  # fabletools has already re-evaluated the model's right-hand side against
  # `new_data`, so `specials$xreg` holds the *future* values of each regressor.
  request <- fable_request(object, fable_covariates(specials, object$model), h)
  qmat <- zuk_run_batches(
    object$model,
    request$contexts,
    request$horizons,
    object$quantile_levels,
    groups = request$groups
  )[[1]]
  build_distribution(qmat, object$quantile_levels)
}

# Flatten the two specials into named numeric series, refusing them at the
# boundary when the checkpoint cannot condition on covariates at all.
fable_covariates <- function(specials, model) {
  flatten <- function(name) {
    values <- unlist(specials[[name]] %||% list(), recursive = FALSE)
    if (!length(values)) return(NULL)
    stats::setNames(lapply(values, as.numeric), names(values))
  }
  known <- flatten("xreg")
  past <- flatten("past_xreg")
  if (is.null(known) && is.null(past)) return(NULL)
  caps <- model$capabilities
  if (!is.null(known) && !isTRUE(caps$future_covariates)) {
    zuk_abort_capability(
      "This checkpoint does not accept future-known regressors.",
      model_id = model$model_id, revision = model$revision,
      capability = "future_covariates", requested = names(known), supported = FALSE
    )
  }
  if (!is.null(past) && !isTRUE(caps$past_covariates)) {
    zuk_abort_capability(
      "This checkpoint does not accept past-only regressors.",
      model_id = model$model_id, revision = model$revision,
      capability = "past_covariates", requested = names(past), supported = FALSE
    )
  }
  list(known = known, past = past)
}

# One task: the response, then its regressors. fabletools evaluates a single key
# at a time, so this task never spans series --- covariates are available
# through this route, cross-series learning is not.
fable_request <- function(object, ahead, h) {
  # The *fitted* model decides which rows exist; `ahead` only fills in the
  # future values of those that have any.
  fitted <- object$covariates
  if (is.null(fitted)) {
    return(list(contexts = list(object$history), horizons = h, groups = NULL))
  }
  contexts <- list(object$history)
  futures <- list(NULL)
  for (name in names(fitted$known)) {
    contexts <- c(contexts, list(fitted$known[[name]]))
    futures <- c(futures, list(ahead$known[[name]]))
  }
  for (name in names(fitted$past)) {
    contexts <- c(contexts, list(fitted$past[[name]]))
    futures <- c(futures, list(NULL))
  }
  n <- length(contexts)
  list(
    contexts = contexts,
    horizons = rep(as.integer(h), n),
    groups = list(id = rep("1", n), target = c(TRUE, rep(FALSE, n - 1L)),
                  future = futures)
  )
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
