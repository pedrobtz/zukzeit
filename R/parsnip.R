# parsnip engine `zuk_reg` / engine "zukzeit".
#
# A parsnip spec makes a supported model usable inside a workflow() by passing
# its checkpoint identity as an engine argument. The spec's sole tunable main
# argument is `context_length`;
# everything model-specific (`model_id`, `revision`, the time `index`, the
# series `id`, `quantile_levels`) is passed through `set_engine()`.
#
# Registration is lazy (from .onLoad, only when parsnip is installed) and wrapped
# defensively so it can never block package load. The full fit/predict path is
# validated with the weight-free stub in the parsnip-gated tests.

#' A parsnip specification for time-series foundation models
#'
#' @param mode Only `"regression"` is supported.
#' @param engine The engine (only `"zukzeit"`).
#' @param context_length Optional context length (tunable).
#' @details
#' Model-specific settings are supplied through `set_engine("zukzeit", ...)`:
#' `model_id` (required; use `"stub"` for the executable test fixture),
#' `revision`, `index`
#' (time column), `id` (series column), and `quantile_levels`.
#'
#' # Inside a workflow
#'
#' `index` and `id` name columns the engine reads from the data it is handed,
#' so both have to survive the workflow's preprocessor. `add_formula()` keeps
#' only the terms in the formula, which silently drops the series id and makes
#' the fit fail on a missing column. Use `add_variables()`, which passes the
#' named columns through untouched:
#'
#' ```r
#' workflow() |>
#'   add_variables(outcomes = sales, predictors = c(month, store)) |>
#'   add_model(spec)
#' ```
#'
#' A formula listing every needed column (`sales ~ month + store`) works too.
#'
#' @examplesIf requireNamespace("parsnip", quietly = TRUE)
#' spec <- zuk_reg(context_length = 512) |>
#'   parsnip::set_engine("zukzeit", model_id = "stub",
#'                       index = "day", id = "store")
#' spec
#' @return A parsnip `model_spec`.
#' @export
zuk_reg <- function(mode = "regression", engine = "zukzeit",
                     context_length = NULL) {
  zuk_require_namespace(
    "parsnip",
    reason = "It provides the tidymodels specification interface (the engine itself does not need it)."
  )
  args <- list(context_length = rlang::enquo(context_length))
  parsnip::new_model_spec(
    "zuk_reg",
    args     = args,
    eng_args = NULL,
    mode     = mode,
    method   = NULL,
    engine   = engine
  )
}

#' Update a zukzeit model specification
#'
#' The [stats::update()] method every `parsnip` specification is expected to
#' provide. `tune` reaches it through `tune::finalize_workflow()` when it
#' substitutes a chosen `context_length` back into the spec, so without it the
#' whole tuning path fails with `update.default()`'s "need an object with call
#' component".
#'
#' @param object A [zuk_reg()] specification.
#' @param parameters A one-row tibble or named list of parameter values, as
#'   supplied by `tune::finalize_workflow()`.
#' @param context_length New context length, or `NULL` to leave it unchanged.
#' @param fresh If `TRUE`, replace the argument set rather than modifying it.
#' @param ... Not used.
#' @return An updated [zuk_reg()] specification.
#' @export
#' @examplesIf requireNamespace("parsnip", quietly = TRUE)
#' spec <- zuk_reg(context_length = 512L)
#' update(spec, context_length = 128L)
update.zuk_reg <- function(object, parameters = NULL, context_length = NULL,
                            fresh = FALSE, ...) {
  zuk_require_namespace(
    "parsnip",
    reason = "It provides the tidymodels specification interface (the engine itself does not need it)."
  )
  args <- list(context_length = rlang::enquo(context_length))
  parsnip::update_spec(
    object = object,
    parameters = parameters,
    args_enquo_list = args,
    fresh = fresh,
    cls = "zuk_reg",
    ...
  )
}

#' Fit bridge used by the parsnip engine
#'
#' This exported bridge is an implementation detail required by parsnip's
#' package-qualified engine registration. Users normally call [zuk_reg()] and
#' `parsnip::fit()` rather than invoking it directly.
#'
#' @param formula,data Passed by parsnip's formula fit interface.
#' @param model_id,revision,device,reuse Checkpoint identity and lifecycle
#'   arguments passed to [zuk_pretrained()].
#' @param index,id Time-index and optional series-id columns.
#' @param quantile_levels Quantiles retained by the fitted engine.
#' @param context_length Optional maximum history used at inference.
#' @param ... Load-affecting architecture options forwarded to
#'   [zuk_pretrained()]. They become part of the resident-cache key and are
#'   available to the constructor as `config$load_options`.
#' @return A `zuk_fit` object.
#' @keywords internal
#' @export
#' @examplesIf requireNamespace("parsnip", quietly = TRUE)
#' train <- data.frame(store = "a", day = 1:40,
#'                     sales = cumsum(rep(2, 40)) + 100)
#'
#' # Normally reached through zuk_reg() and parsnip::fit(); called directly
#' # here to show what the engine registration wires up.
#' fit <- zuk_parsnip_fit(sales ~ ., data = train, model_id = "stub",
#'                         index = "day", id = "store")
#' predict(fit, new_data = data.frame(store = "a", day = 41:43))
#'
#' zuk_unload("stub")
zuk_parsnip_fit <- function(formula, data, model_id, revision = NULL,
                             device = NULL, reuse = TRUE,
                             index, id = NULL,
                             quantile_levels = c(0.1, 0.5, 0.9),
                             context_length = NULL, ...) {
  model <- zuk_pretrained(
    model_id,
    revision = revision,
    device = device,
    reuse = reuse,
    ...
  )
  if (!is.null(context_length)) {
    check_context_length(model$capabilities, context_length)
    model$capabilities$max_context <- as.integer(context_length)
  }
  zuk_fit(formula, data = data, model = model, index = index, id = id,
           quantile_levels = quantile_levels)
}

# Register the model with parsnip. Idempotent and best-effort.
make_zuk_reg <- function() {
  if ("zuk_reg" %in% parsnip::get_from_env("models")) {
    return(invisible())
  }
  parsnip::set_new_model("zuk_reg")
  parsnip::set_model_mode("zuk_reg", "regression")
  parsnip::set_model_engine("zuk_reg", "regression", "zukzeit")
  parsnip::set_dependency("zuk_reg", "zukzeit", "zukzeit")

  parsnip::set_model_arg(
    model = "zuk_reg", eng = "zukzeit",
    parsnip = "context_length", original = "context_length",
    func = list(pkg = "zukzeit", fun = "context_length"),
    has_submodel = FALSE
  )

  parsnip::set_fit(
    model = "zuk_reg", eng = "zukzeit", mode = "regression",
    value = list(
      interface = "formula",
      protect   = c("formula", "data"),
      func      = c(pkg = "zukzeit", fun = "zuk_parsnip_fit"),
      defaults  = list()
    )
  )

  parsnip::set_encoding(
    model = "zuk_reg", eng = "zukzeit", mode = "regression",
    options = list(
      predictor_indicators = "none",
      compute_intercept    = FALSE,
      remove_intercept     = FALSE,
      allow_sparse_x       = FALSE
    )
  )

  parsnip::set_pred(
    model = "zuk_reg", eng = "zukzeit", mode = "regression", type = "numeric",
    value = list(
      pre  = NULL,
      post = function(results, object) {
        data.frame(.pred = results$.pred, check.names = FALSE)
      },
      func = c(fun = "predict"),
      args = list(
        object   = rlang::expr(object$fit),
        new_data = rlang::expr(new_data)
      )
    )
  )

  # The tunable() generic is re-exported across the tidymodels stack from a
  # single home (historically `generics`); register against whichever installed
  # package actually owns it, so dispatch works regardless of version.
  for (pkg in c("generics", "hardhat", "tune", "parsnip")) {
    if (requireNamespace(pkg, quietly = TRUE) &&
        exists("tunable", envir = asNamespace(pkg), inherits = FALSE)) {
      vctrs::s3_register(paste0(pkg, "::tunable"), "zuk_reg", tunable_zuk_reg)
      break
    }
  }
  invisible()
}

tunable_zuk_reg <- function(x, ...) {
  data.frame(
    name         = "context_length",
    call_info    = I(list(list(pkg = "zukzeit", fun = "context_length"))),
    source       = "model_spec",
    component    = "zuk_reg",
    component_id = "main",
    stringsAsFactors = FALSE
  )
}

#' Context length parameter (for tuning)
#'
#' A \pkg{dials} parameter so `context_length` can be tuned with
#' `tune::tune_grid()` over `rsample` resamples.
#'
#' @param range Integer range of context lengths.
#' @param trans A transformation, or `NULL`.
#' @return A `dials` quantitative parameter.
#' @export
#' @examplesIf requireNamespace("dials", quietly = TRUE)
#' context_length()
#'
#' # Narrow the search space, then draw a grid from it.
#' dials::grid_regular(context_length(range = c(64L, 512L)), levels = 4)
context_length <- function(range = c(64L, 2048L), trans = NULL) {
  zuk_require_namespace(
    "dials",
    reason = "It provides the tunable parameter objects used by the tidymodels adapter."
  )
  dials::new_quant_param(
    type      = "integer",
    range     = range,
    inclusive = c(TRUE, TRUE),
    trans     = trans,
    label     = c(context_length = "Context Length"),
    finalize  = NULL
  )
}
