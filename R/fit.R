# ADAPTER: tidymodels. Not part of the engine core.
#
# The engine contract is `predict_fn(context, h, quantile_levels)` -> quantile
# matrix (see ?`tsfm-architecture-contract`). This file is one of several
# optional surfaces onto it; hardhat and parsnip are Suggests, and the engine
# works fully without them. Peers: as_fable.tsfm_forecast() for the tidyverts,
# and forecast.tsfm_model() for plain data frames and tsibbles.
#
# For a zero-shot foundation model, "fitting" binds the loaded model to a panel
# of history: it validates the request against the model's capabilities and
# stores each series' observed values. `predict()` then forecasts the horizons
# implied by `new_data` and returns a tidymodels-conformant data frame,
# row-aligned to `new_data`.
#
# There is deliberately no hardhat mold here. A zero-shot model consumes no
# predictors, so a blueprint would describe columns nothing reads, and forging
# `new_data` through it would demand training predictors the caller has no
# reason to supply at forecast time. Keeping it out means `tsfm_fit()` needs no
# optional dependency at all.

#' Fit (bind) a foundation model to training history
#'
#' @param x A `formula` or a `data.frame`.
#' @param ... Passed to methods.
#' @return A `tsfm_fit` object.
#' @export
#' @examples
#' model <- tsfm_pretrained("stub")
#' train <- data.frame(
#'   store = rep(c("a", "b"), each = 40),
#'   day   = rep(1:40, 2),
#'   sales = c(cumsum(rep(2, 40)) + 100, cumsum(rep(1, 40)) + 50)
#' )
#'
#' # Formula interface: the left-hand side names the target series.
#' fit <- tsfm_fit(sales ~ ., data = train, model = model,
#'                 index = "day", id = "store")
#' fit
#'
#' # Or name the columns directly.
#' tsfm_fit(train, model = model, target = "sales", index = "day", id = "store")
#'
#' tsfm_unload("stub")
tsfm_fit <- function(x, ...) {
  UseMethod("tsfm_fit")
}

#' @param data Training data: one row per (series, time) with the target,
#'   a time `index` column, and an optional series `id` column.
#' @param model A `tsfm_model` from [tsfm_pretrained()].
#' @param index,id Column names of the time index and (optional) series id.
#' @param quantile_levels Numeric vector of quantile levels to forecast.
#' @rdname tsfm_fit
#' @export
tsfm_fit.formula <- function(x, data, model, index, id = NULL,
                             quantile_levels = c(0.1, 0.5, 0.9), ...) {
  if (!inherits(model, "tsfm_model")) {
    tsfm_abort_contract(
      "{.arg model} must be a {.cls tsfm_model} (see {.fn tsfm_pretrained}).",
      contract = "fit input",
      expected = "tsfm_model",
      actual = class(model)
    )
  }
  targets <- all.vars(rlang::f_lhs(x))
  # Pre-flight: reject multivariate targets before any work if unsupported.
  check_capabilities(model$capabilities, multivariate = length(targets) > 1)
  if (length(targets) != 1L) {
    tsfm_abort_capability(
      "Contract v1 supports exactly one target column; got {length(targets)}.",
      model_id = model$model_id,
      revision = model$revision,
      capability = "multivariate",
      requested = length(targets),
      supported = 1L
    )
  }
  tsfm_fit_data_frame(data, model, target = targets, index = index, id = id,
                      quantile_levels = quantile_levels)
}

#' @param target Column name of the target series (data.frame method).
#' @rdname tsfm_fit
#' @export
tsfm_fit.data.frame <- function(x, model, target, index, id = NULL,
                                quantile_levels = c(0.1, 0.5, 0.9), ...) {
  tsfm_fit_data_frame(x, model, target = target, index = index, id = id,
                      quantile_levels = quantile_levels)
}

tsfm_fit_data_frame <- function(data, model, target, index, id,
                                quantile_levels) {
  if (!inherits(model, "tsfm_model")) {
    tsfm_abort_contract(
      "{.arg model} must be a {.cls tsfm_model} (see {.fn tsfm_pretrained}).",
      contract = "fit input",
      expected = "tsfm_model",
      actual = class(model)
    )
  }
  quantile_levels <- check_quantile_levels(
    model$capabilities,
    quantile_levels,
    model$model_id,
    model$revision
  )
  fields <- list(target = target, index = index)
  if (!is.null(id)) fields$id <- id
  invalid <- vapply(
    fields,
    function(name) length(name) != 1L || !is.character(name) ||
      is.na(name) || !nzchar(name),
    logical(1)
  )
  if (any(invalid)) {
    tsfm_abort_capability(
      "Column arguments must each name exactly one column.",
      model_id = model$model_id,
      revision = model$revision,
      capability = "input_columns",
      requested = fields[invalid],
      supported = names(data)
    )
  }
  required <- c(target, index, id)
  missing <- setdiff(required, names(data))
  if (length(missing)) {
    tsfm_abort_capability(
      "Column{?s} {.val {missing}} not found in {.arg data}.",
      model_id = model$model_id,
      revision = model$revision,
      capability = "input_columns",
      requested = missing,
      supported = names(data)
    )
  }

  y <- data[[target]]
  if (!is.numeric(y)) {
    tsfm_abort_capability(
      "Target {.val {target}} must be numeric; got {.cls {class(y)}}.",
      model_id = model$model_id,
      revision = model$revision,
      capability = "target_type",
      requested = class(y),
      supported = "numeric"
    )
  }

  # Synthesise a single series id when none is supplied.
  if (is.null(id)) {
    id <- ".series"
    data[[id]] <- ".series"
  }
  # Series are keyed by label, never by factor level code: predict() looks
  # histories up by name, and a factor index into a list silently resolves by
  # code instead. Normalising here keeps fit and predict in the same key space.
  data[[id]] <- as.character(data[[id]])
  data <- data[order(data[[id]], data[[index]]), , drop = FALSE]
  histories <- split(as.numeric(data[[target]]), data[[id]])

  structure(
    list(
      model           = model,
      target          = target,
      index           = index,
      id              = id,
      quantile_levels = quantile_levels,
      histories       = histories
    ),
    class = "tsfm_fit"
  )
}

#' @export
print.tsfm_fit <- function(x, ...) {
  cli::cli_text("{.cls tsfm_fit} ({.val {x$model$architecture}})")
  cli::cli_text("target {.val {x$target}} over {.val {length(x$histories)}} series")
  invisible(x)
}

#' Forecast from a fitted foundation model
#'
#' Returns a tidymodels-conformant data frame with one row per row of
#' `new_data` (same order): `.pred` (point), `.pred_lower`, `.pred_upper`, and
#' one `.pred_qXX` column per fitted quantile level, where `XX` is the level as
#' a percentage (`0.1` becomes `.pred_q10`). Levels finer than a whole percent
#' carry the extra digits after an underscore (`0.025` becomes `.pred_q02_5`),
#' so no two levels can ever map to the same column.
#'
#' @param object A `tsfm_fit`.
#' @param new_data Future rows to forecast: the `index` (and `id`) columns that
#'   were used at fit time, one row per horizon to predict.
#' @param ... Unused.
#' @return A `data.frame` of predictions row-aligned to `new_data`.
#' @export
#' @examples
#' model <- tsfm_pretrained("stub")
#' train <- data.frame(
#'   store = rep(c("a", "b"), each = 40),
#'   day   = rep(1:40, 2),
#'   sales = c(cumsum(rep(2, 40)) + 100, cumsum(rep(1, 40)) + 50)
#' )
#' fit <- tsfm_fit(sales ~ ., data = train, model = model,
#'                 index = "day", id = "store")
#'
#' # One row of predictions per row of `new_data`, in that same order.
#' future <- data.frame(store = c("b", "a", "b"), day = c(41L, 41L, 42L))
#' predict(fit, new_data = future)
#'
#' tsfm_unload("stub")
predict.tsfm_fit <- function(object, new_data, ...) {
  index <- object$index
  id <- object$id
  if (!index %in% names(new_data)) {
    tsfm_abort_capability(
      "Column {.val {index}} not found in {.arg new_data}.",
      model_id = object$model$model_id,
      revision = object$model$revision,
      capability = "input_columns",
      requested = index,
      supported = names(new_data)
    )
  }
  nd <- as.data.frame(new_data)
  if (!id %in% names(nd)) {
    nd[[id]] <- ".series"
  }
  # See tsfm_fit_data_frame(): keys are labels. Without this, a factor `id`
  # column --- anything that has been through droplevels() or subset() --- makes
  # `object$histories[ids]` select by level code, which either forecasts the
  # wrong series or fails several frames down with an error about horizons.
  nd[[id]] <- as.character(nd[[id]])

  # Every series in new_data must have a fitted history to condition on.
  novel <- setdiff(unique(nd[[id]]), names(object$histories))
  if (length(novel)) {
    tsfm_abort_capability(c(
      "No fitted history for series {.val {novel}}.",
      "i" = "Every series in {.arg new_data} must appear in the training data."
    ),
    model_id = object$model$model_id,
    revision = object$model$revision,
    capability = "series_history",
    requested = novel,
    supported = names(object$histories))
  }

  nd$.row <- seq_len(nrow(nd))
  nd_sorted <- nd[order(nd[[id]], nd[[index]]), , drop = FALSE]

  ids <- unique(nd_sorted[[id]])
  histories <- object$histories[ids]
  future_index <- split(nd_sorted[[index]], nd_sorted[[id]])

  fc <- tsfm_infer(
    object$model,
    histories = histories,
    future_index = future_index,
    quantile_levels = object$quantile_levels,
    key_name = id,
    index_name = index,
    target = object$target
  )

  cols <- tsfm_quantile_columns(fc)
  cols$.row <- nd_sorted$.row
  cols <- cols[order(cols$.row), , drop = FALSE]
  cols$.row <- NULL
  rownames(cols) <- NULL
  cols
}
