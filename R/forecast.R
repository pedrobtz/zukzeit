# The forecast object and the two adapter surfaces.
#
# A single inference driver (`zuk_infer()`) turns per-series histories into a
# `zuk_forecast`. That object stores, row-aligned:
#   * key + index                (which series, which future step)
#   * .mean                      (point forecast)
#   * .distribution              (a distributional vector, for as_fable())
#   * a quantile matrix + levels (attributes, for the tidy predict() path)
# Both views come from the same predicted quantiles, so the fabletools and
# tidymodels surfaces never have to translate through each other.

# ---- inference driver -------------------------------------------------------

# histories:    named list (by key) of numeric vectors, oldest-first.
# future_index: named list (by key) of future index values, one per horizon.
# The actual forward pass (truncation, batching, device) is delegated to
# zuk_run_batches(); this driver only assembles the forecast object.
# Returns a `zuk_forecast`.
zuk_infer <- function(model, histories, future_index, quantile_levels,
                       key_name = "key", index_name = "index",
                       target = ".response",
                       batch_size = NULL, device = NULL,
                       covariates = NULL, tasks = NULL) {
  quantile_levels <- check_quantile_levels(
    model$capabilities,
    quantile_levels,
    model$model_id,
    model$revision
  )
  # Always evaluate the median so the point forecast is exact. Resolve it
  # against the checkpoint's own trained levels: check_quantile_levels() has
  # already rewritten `quantile_levels` into the checkpoint's spelling, so
  # appending a literal 0.5 can add a second spelling of a level already
  # requested --- which zuk_run_batches() then rejects as a duplicate.
  median_level <- resolve_median_level(model$capabilities, quantile_levels)
  levels <- sort(unique(c(quantile_levels, median_level)))
  median_col <- match(median_level, levels)

  keys <- names(histories)
  unmatched <- setdiff(keys, names(future_index))
  if (length(unmatched)) {
    zuk_abort_contract(
      "Every history must have a future index under the same key.",
      architecture = model$architecture,
      model_id = model$model_id,
      contract = "engine input alignment",
      expected = names(future_index),
      actual = keys
    )
  }
  horizons <- vapply(keys, function(k) length(future_index[[k]]), integer(1))

  if (is.null(covariates) && is.null(tasks)) {
    qmats <- zuk_run_batches(model, histories[keys], horizons, levels,
                             batch_size = batch_size, device = device)
  } else {
    # One target row per key, each followed by its covariate rows in the same
    # task. The engine answers for target rows only, in this same order.
    rows <- list(); ids <- character(); is_target <- logical()
    ahead <- list(); row_horizons <- integer()
    for (key in keys) {
      task <- if (is.null(tasks)) key else tasks[[key]]
      rows <- c(rows, list(histories[[key]])); ids <- c(ids, task)
      is_target <- c(is_target, TRUE); ahead <- c(ahead, list(NULL))
      row_horizons <- c(row_horizons, horizons[[key]])
      for (covariate in covariates[[key]]) {
        rows <- c(rows, list(covariate$values)); ids <- c(ids, task)
        is_target <- c(is_target, FALSE); ahead <- c(ahead, list(covariate$future))
        row_horizons <- c(row_horizons, horizons[[key]])
      }
    }
    qmats <- zuk_run_batches(
      model, rows, row_horizons, levels,
      batch_size = batch_size, device = device,
      groups = list(id = ids, target = is_target, future = ahead)
    )
  }

  key_out <- vector("list", length(keys))
  idx_out <- vector("list", length(keys))
  qmat_out <- vector("list", length(keys))
  mean_out <- vector("list", length(keys))

  for (i in seq_along(keys)) {
    h <- horizons[[i]]
    qmat <- matrix(as.numeric(qmats[[i]]), nrow = h, ncol = length(levels))
    key_out[[i]] <- rep(keys[[i]], h)
    idx_out[[i]] <- future_index[[keys[[i]]]]
    qmat_out[[i]] <- qmat
    mean_out[[i]] <- qmat[, median_col]
  }

  key_vec <- unlist(key_out, use.names = FALSE)
  idx_vec <- do.call(c, idx_out)
  mean_vec <- unlist(mean_out, use.names = FALSE)
  qmatrix <- do.call(rbind, qmat_out)
  if (is.null(qmatrix)) qmatrix <- matrix(numeric(0), ncol = length(levels))

  # distributional column (only the user-requested levels).
  req_cols <- match(quantile_levels, levels)
  dist <- build_distribution(qmatrix[, req_cols, drop = FALSE], quantile_levels)

  # Build an n-row, 0-column frame first so column assignment sets the rows
  # (assigning a length-n vector into a 0-row frame errors).
  df <- data.frame(matrix(nrow = length(key_vec), ncol = 0))
  df[[key_name]] <- key_vec
  df[[index_name]] <- idx_vec
  df[[".mean"]] <- mean_vec
  df[[".distribution"]] <- dist

  new_zuk_forecast(
    df,
    key_name = key_name,
    index_name = index_name,
    target = target,
    quantile_levels = quantile_levels,
    quantiles = qmatrix[, req_cols, drop = FALSE],
    levels = quantile_levels
  )
}

# The level whose forecast becomes the point forecast. A checkpoint that
# declares its trained levels supplies the value in its own spelling; one that
# accepts arbitrary levels takes the exact median. A checkpoint trained only on
# tail quantiles has no median to evaluate, so the requested level nearest 0.5
# stands in rather than failing a request that is otherwise fully supported.
resolve_median_level <- function(caps, levels) {
  supported <- caps$quantile_levels
  if (is.null(supported)) {
    return(0.5)
  }
  matched <- zuk_match_quantile_levels(0.5, supported)
  if (!is.na(matched)) {
    return(supported[[matched]])
  }
  levels[[which.min(abs(levels - 0.5))]]
}

# Build a distributional vector of predictive distributions from a quantile
# matrix (rows = observations, cols = `levels`).
build_distribution <- function(qmatrix, levels) {
  zuk_require_namespace(
    "distributional",
    reason = "It is needed to store predictive distributions."
  )
  n <- nrow(qmatrix)
  if (n == 0L) {
    return(distributional::dist_percentile(list(), list()))
  }
  values <- lapply(seq_len(n), function(i) as.numeric(qmatrix[i, ]))
  pcts <- rep(list(as.numeric(levels) * 100), n)
  distributional::dist_percentile(values, pcts)
}

# ---- grouped panel requests --------------------------------------------------

# Resolve the task each key belongs to. `group = NULL` keeps today's behaviour:
# one task per key, so a forecast depends on its own series and the model, not
# on what else happened to be in the call. Naming a column opts in to
# cross-series learning by letting several keys share a task.
panel_tasks <- function(spec, keys, group, call = rlang::caller_env()) {
  if (is.null(group)) {
    return(stats::setNames(as.character(keys), keys))
  }
  if (length(group) != 1L || !is.character(group) || !group %in% names(spec$data)) {
    zuk_abort_capability(
      "{.arg group} must name one column of {.arg new_data}.",
      capability = "input_columns", requested = group,
      supported = names(spec$data), call = call
    )
  }
  by_key <- split(spec$data[[group]], spec$data[[spec$key]], drop = TRUE)
  mixed <- names(by_key)[vapply(by_key, function(g) length(unique(g)) > 1L, logical(1))]
  if (length(mixed)) {
    zuk_abort_capability(
      c(
        "{.arg group} must be constant within a series.",
        "x" = "Varies within {.val {mixed}}."
      ),
      capability = "input_columns", requested = group,
      supported = "one task label per series", call = call
    )
  }
  vapply(keys, function(k) as.character(by_key[[k]][[1]]), character(1))
}

# Assemble the covariate rows that ride alongside each target series. A
# covariate present in `future` is future-known; one absent is past-only. That
# is inferred rather than declared, which is one fewer argument than naming both
# sets and cannot disagree with itself.
panel_covariates <- function(spec, keys, covariates, future, horizons,
                             call = rlang::caller_env()) {
  if (is.null(covariates)) return(NULL)
  covariates <- as.character(covariates)
  reserved <- c(spec$index, spec$key, spec$target)
  unknown <- setdiff(covariates, setdiff(names(spec$data), reserved))
  if (length(unknown)) {
    zuk_abort_capability(
      c(
        "{.arg covariates} must name columns of {.arg new_data}.",
        "x" = "Not available as covariates: {.val {unknown}}."
      ),
      capability = "input_columns", requested = unknown,
      supported = setdiff(names(spec$data), reserved), call = call
    )
  }
  observed <- lapply(covariates, function(column) {
    split(spec$data[[column]], spec$data[[spec$key]], drop = TRUE)
  })
  names(observed) <- covariates

  known <- character()
  if (!is.null(future)) {
    if (!inherits(future, "data.frame")) {
      zuk_abort_capability(
        "{.arg future} must be a data frame of future covariate values.",
        capability = "input_data", requested = class(future),
        supported = "data.frame", call = call
      )
    }
    future <- as.data.frame(future)
    known <- intersect(covariates, names(future))
    if (!length(known)) {
      zuk_abort_capability(
        c(
          "{.arg future} contains none of the named covariates.",
          "i" = "Drop it, or supply future values for at least one covariate."
        ),
        capability = "input_columns", requested = names(future),
        supported = covariates, call = call
      )
    }
    if (spec$key %in% names(future)) {
      future[[spec$key]] <- as.character(future[[spec$key]])
    } else {
      future[[spec$key]] <- as.character(keys[[1]])
    }
  }

  lapply(keys, function(key) {
    rows <- if (is.null(future)) NULL else future[future[[spec$key]] == key, , drop = FALSE]
    lapply(covariates, function(column) {
      ahead <- NULL
      if (column %in% known) {
        ahead <- as.numeric(rows[[column]])
        if (length(ahead) != horizons[[key]]) {
          zuk_abort_capability(
            c(
              "Future values must cover the whole horizon.",
              "x" = "{.val {column}} has {length(ahead)} row{?s} for series {.val {key}}; {horizons[[key]]} needed."
            ),
            capability = "future_covariates", requested = length(ahead),
            supported = horizons[[key]], call = call
          )
        }
      }
      list(values = as.numeric(observed[[column]][[key]]), future = ahead)
    })
  }) -> per_key
  stats::setNames(per_key, keys)
}

# ---- forecast object --------------------------------------------------------

new_zuk_forecast <- function(data, key_name, index_name, target,
                              quantile_levels, quantiles, levels) {
  structure(
    data,
    key_name = key_name,
    index_name = index_name,
    target = target,
    quantile_levels = quantile_levels,
    quantiles = quantiles,
    levels = levels,
    class = c("zuk_forecast", "data.frame")
  )
}

#' @export
print.zuk_forecast <- function(x, ...) {
  cli::cli_text("{.cls zuk_forecast} of {.val {attr(x, 'target')}}")
  cli::cli_text(
    "{.val {length(unique(x[[attr(x, 'key_name')]]))}} series x horizon, ",
    "quantile levels {.val {attr(x, 'quantile_levels')}}"
  )
  print(as.data.frame(utils::head(x, 10L)))
  invisible(x)
}

#' @export
as.data.frame.zuk_forecast <- function(x, ...) {
  attrs <- c("key_name", "index_name", "target", "quantile_levels",
             "quantiles", "levels")
  for (a in attrs) attr(x, a) <- NULL
  class(x) <- "data.frame"
  x
}

# Expand the stored quantile matrix into tidymodels prediction columns:
# .pred (point), .pred_lower / .pred_upper (extreme requested levels), and one
# .pred_qXX column per requested quantile.
zuk_quantile_columns <- function(fc) {
  levels <- attr(fc, "levels")
  qmatrix <- attr(fc, "quantiles")
  out <- data.frame(.pred = fc[[".mean"]], check.names = FALSE)
  if (length(levels)) {
    out[[".pred_lower"]] <- qmatrix[, which.min(levels)]
    out[[".pred_upper"]] <- qmatrix[, which.max(levels)]
    names <- quantile_column_names(levels)
    for (j in seq_along(levels)) {
      out[[names[[j]]]] <- qmatrix[, j]
    }
  }
  out
}

# Column names for the per-quantile prediction columns.
#
# Whole-percent levels keep the familiar two-digit form (0.1 -> `.pred_q10`).
# Finer levels do not: rounding 0.025 and 0.02 both to `.pred_q02` would make
# one column silently overwrite the other, dropping a requested quantile from
# the prediction frame without an error. So the label carries exactly as many
# decimals as the level set needs to be represented without loss, with `.`
# written as `_` to keep the name syntactic (0.025 -> `.pred_q02_5`).
quantile_column_names <- function(levels) {
  percent <- as.numeric(levels) * 100
  digits <- 0L
  while (digits < 6L && any(abs(round(percent, digits) - percent) > 1e-9)) {
    digits <- digits + 1L
  }
  labels <- formatC(
    percent,
    format = "f", digits = digits,
    width = 2L + (digits > 0L) + digits, flag = "0"
  )
  labels <- paste0(".pred_q", gsub(".", "_", labels, fixed = TRUE))
  if (anyDuplicated(labels)) {
    zuk_abort_contract(
      "Quantile levels do not map to distinct prediction columns.",
      contract = "prediction column names",
      expected = "one column per requested level",
      actual = labels
    )
  }
  labels
}

# ---- fable adapter ----------------------------------------------------------

#' Convert a zukzeit forecast to a fable
#'
#' A method for [fabletools::as_fable()], so a `zuk_forecast` flows into
#' `fabletools::accuracy()`, reconciliation, and the tidyverts plotting stack.
#'
#' fabletools is an optional adapter dependency, so this method is registered
#' lazily from `.onLoad()` via [vctrs::s3_register()] — it becomes available as
#' soon as fabletools is loaded, and its absence never blocks loading zukzeit.
#' Call it as `fabletools::as_fable(x)`; zukzeit deliberately does not define a
#' competing `as_fable()` generic.
#'
#' @param x A `zuk_forecast`.
#' @param ... Unused.
#' @return A `fable` object.
#' @name as_fable.zuk_forecast
#' @keywords internal
#' @examplesIf requireNamespace("fabletools", quietly = TRUE) && requireNamespace("tsibble", quietly = TRUE)
#' model <- zuk_pretrained("stub")
#' history <- tsibble::tsibble(
#'   month = tsibble::yearmonth(seq(as.Date("2020-01-01"), by = "month",
#'                                  length.out = 36)),
#'   sales = as.numeric(1:36),
#'   index = month
#' )
#'
#' # Call it qualified: zukzeit registers a method rather than a rival generic.
#' fabletools::as_fable(forecast(model, history, h = 3))
#'
#' zuk_unload("stub")
as_fable.zuk_forecast <- function(x, ...) {
  zuk_require_namespace(
    c("tsibble", "fabletools"),
    reason = "They are needed to convert forecasts to a fable."
  )
  key_name <- attr(x, "key_name")
  index_name <- attr(x, "index_name")
  target <- attr(x, "target")

  df <- as.data.frame(x)
  # Name the distribution column after the response, as fable expects.
  df[[target]] <- df[[".distribution"]]
  dimnames(df[[target]]) <- target
  df[[".distribution"]] <- NULL
  # Keep `.mean`. `as_fable()` does not synthesise it the way
  # `fabletools::forecast()` does, so dropping it here produced a fable whose
  # `.mean` was silently NULL. The retained value is the engine's exact median
  # (see zuk_infer()), which is also `median()` of the distribution beside it.

  # Inject symbols so tsibble's tidy-select interface does not treat character
  # column names as deprecated external selection vectors.
  tsbl <- rlang::inject(
    tsibble::build_tsibble(
      df,
      key = !!rlang::sym(key_name),
      index = !!rlang::sym(index_name)
    )
  )
  # `distribution` is captured as a bare symbol by as_fable(); inject it.
  rlang::inject(
    fabletools::as_fable(
      tsbl,
      response = !!target,
      distribution = !!rlang::sym(target)
    )
  )
}

# ---- forecast() convenience path (tsibble / data.frame) ---------------------

# The `forecast()` verb is shared across the R forecasting ecosystem
# (fabletools, modeltime, forecast) via the generic that lives in `generics`.
# Re-exporting it — rather than defining a competing local generic — is what
# keeps dispatch correct when zukzeit is attached alongside those packages.

#' @importFrom generics forecast
#' @export
generics::forecast

#' Forecast future values with a foundation model
#'
#' The convenience surface: hand a panel of history and a horizon, get a
#' `zuk_forecast` back. This is a method for [generics::forecast()], the same
#' generic fabletools and modeltime dispatch on, so attaching zukzeit alongside
#' them never masks the verb.
#'
#' @param object A `zuk_model`.
#' @param new_data A `tsibble` (index/key inferred) or a `data.frame`.
#' @param h Integer forecast horizon.
#' @param quantile_levels Numeric vector of quantile levels in `(0, 1)`.
#' @param index,key,target Column names; required for the `data.frame` method,
#'   inferred for a `tsibble`.
#' @param covariates Character vector naming columns of `new_data` to condition
#'   on. Named rather than inferred from the leftover columns, because an
#'   unspecified `target` is itself inferred from those. A covariate that also
#'   appears in `future` is future-known; one that does not is past-only.
#' @param future Data frame of future covariate values, carrying the `key`
#'   column and one column per future-known covariate, with `h` rows per series.
#' @param group Optional column of `new_data` naming the task a series belongs
#'   to. `NULL` gives one task per series, which is the existing behaviour and
#'   keeps a forecast a function of its own series. Naming a column opts in to
#'   cross-series learning, where series sharing a task inform one another.
#' @param batch_size,device Passed to [zuk_run_batches()].
#' @param ... Unused.
#' @return A `zuk_forecast`.
#' @export
#' @examples
#' model <- zuk_pretrained("stub")
#'
#' # A plain data frame needs its index and target columns named.
#' history <- data.frame(day = 1:60, sales = cumsum(rep(2, 60)) + 100)
#' fc <- forecast(model, history, h = 5, index = "day", target = "sales")
#' fc
#'
#' as.data.frame(fc)
#'
#' # A panel: one row per series and time step, keyed by `store`.
#' panel <- data.frame(
#'   store = rep(c("a", "b"), each = 60),
#'   day   = rep(1:60, 2),
#'   sales = c(cumsum(rep(2, 60)) + 100, cumsum(rep(1, 60)) + 50)
#' )
#' forecast(model, panel, h = 3, index = "day", key = "store", target = "sales")
#'
#' zuk_unload("stub")
forecast.zuk_model <- function(object, new_data, h = 1L,
                                quantile_levels = c(0.1, 0.5, 0.9),
                                index = NULL, key = NULL, target = NULL,
                                covariates = NULL, future = NULL, group = NULL,
                                batch_size = NULL, device = NULL, ...) {
  check_horizon(
    object$capabilities,
    h,
    object$model_id,
    object$revision
  )
  h <- as.integer(h)
  spec <- panel_spec(new_data, index = index, key = key, target = target)

  histories <- split(spec$data[[spec$target]], spec$data[[spec$key]], drop = TRUE)
  future_index <- future_index_for(spec, h)
  keys <- names(histories)
  horizons <- stats::setNames(rep(h, length(keys)), keys)

  tasks <- if (is.null(group)) NULL else panel_tasks(spec, keys, group)
  covariate_rows <- panel_covariates(spec, keys, covariates, future, horizons)

  zuk_infer(
    object,
    histories = histories,
    future_index = future_index,
    quantile_levels = quantile_levels,
    key_name = spec$key,
    index_name = spec$index,
    target = spec$target,
    batch_size = batch_size,
    device = device,
    covariates = covariate_rows,
    tasks = tasks
  )
}

# Normalise a tsibble or data.frame into a common spec: a plain data.frame plus
# the resolved key/index/target column names. A synthetic single key is added
# when none is supplied so downstream code always groups by a key.
panel_spec <- function(new_data, index = NULL, key = NULL, target = NULL) {
  if (inherits(new_data, "tbl_ts")) {
    zuk_require_namespace("tsibble")
    index <- index %||% as.character(tsibble::index_var(new_data))
    key_vars <- tsibble::key_vars(new_data)
    key <- key %||% (if (length(key_vars)) key_vars[[1]] else NULL)
    measured <- setdiff(names(new_data), c(index, key_vars))
    target <- target %||% if (length(measured)) measured[[1]] else NULL
    data <- as.data.frame(new_data)
  } else {
    if (!inherits(new_data, "data.frame")) {
      zuk_abort_capability(
        "{.arg new_data} must be a data frame or tsibble.",
        capability = "input_data",
        requested = class(new_data),
        supported = c("data.frame", "tbl_ts")
      )
    }
    if (is.null(index) || is.null(target)) {
      zuk_abort_capability(
        "For a {.cls data.frame}, both {.arg index} and {.arg target} are required.",
        capability = "input_columns",
        requested = c(index = index %||% NA_character_, target = target %||% NA_character_),
        supported = names(new_data)
      )
    }
    data <- as.data.frame(new_data)
  }
  if (is.null(key)) {
    key <- ".series"
    data[[key]] <- ".series"
  }
  fields <- list(index = index, key = key, target = target)
  invalid_names <- vapply(
    fields,
    function(name) length(name) != 1L || !is.character(name) ||
      is.na(name) || !nzchar(name),
    logical(1)
  )
  missing <- names(fields)[invalid_names]
  if (!length(missing)) missing <- setdiff(unlist(fields), names(data))
  if (length(missing)) {
    zuk_abort_capability(
      "Required column{?s} {.val {missing}} {?is/are} not available.",
      capability = "input_columns",
      requested = missing,
      supported = names(data)
    )
  }
  if (!is.numeric(data[[target]])) {
    zuk_abort_capability(
      "Target {.val {target}} must be numeric.",
      capability = "target_type",
      requested = class(data[[target]]),
      supported = "numeric"
    )
  }
  data <- data[order(data[[key]], data[[index]]), , drop = FALSE]
  list(data = data, index = index, key = key, target = target)
}

# Generate `h` future index values per key by extending the observed index by
# its typical step (works for numeric, integer, and Date indices).
future_index_for <- function(spec, h) {
  by_key <- split(spec$data[[spec$index]], spec$data[[spec$key]], drop = TRUE)
  lapply(by_key, function(idx) extend_index(idx, h))
}

# Extend an observed index by `h` further steps of its own typical spacing.
#
# The step has to be measured numerically, but the extension must not be: a
# `yearmonth` panel extended in numeric space came back indexed `636, 637, 638`
# instead of `2023 Jan`, because tsibble's calendar types count periods under
# `as.numeric()` while storing days. A bare numeric index silently breaks the
# join back to the caller's data, the interval a tsibble reports, and any
# accuracy() call against held-out actuals. `seq()` already knows how to walk
# each index type in its own units, so the extension is delegated to it.
extend_index <- function(idx, h, call = rlang::caller_env()) {
  n <- length(idx)
  if (n == 0L) return(idx[0])
  if (!is.numeric(idx) &&
      !inherits(idx, c("Date", "POSIXct", "vctrs_vctr"))) {
    zuk_abort_capability(
      c(
        "Cannot extend an index of class {.cls {class(idx)}}.",
        "i" = "Supported index types are numeric, integer, {.cls Date}, {.cls POSIXct}, and tsibble's calendar types."
      ),
      capability = "index_type",
      requested = class(idx),
      supported = c("numeric", "integer", "Date", "POSIXct", "vctrs_vctr"),
      call = call
    )
  }
  step <- if (n >= 2L) stats::median(diff(as.numeric(idx))) else 1
  if (!is.finite(step) || step == 0) step <- 1
  extended <- tryCatch(
    seq(idx[n], by = step, length.out = h + 1L)[-1L],
    error = function(e) NULL
  )
  if (is.null(extended) || length(extended) != h) {
    zuk_abort_capability(
      c(
        "Could not extend a {.cls {class(idx)}} index by {h} step{?s} of {.val {step}}.",
        "i" = "Supply a regularly spaced index, or forecast from a plain data frame with a numeric index."
      ),
      capability = "index_type",
      requested = class(idx),
      supported = "a regularly spaced index seq() can extend",
      call = call
    )
  }
  # `is.object()` guard: seq.Date() and friends hand back classed vectors with
  # integer storage, and as.integer() would strip the class straight back off.
  if (is.integer(idx) && !is.object(idx)) as.integer(round(extended)) else extended
}