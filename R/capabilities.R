#' Capability metadata for a time-series foundation model
#'
#' A `zuk_capabilities` object is a small record describing what a model can
#' and cannot do. It is attached to every model returned by
#' [zuk_pretrained()] and is used for *pre-flight validation*: a fit or
#' forecast call is rejected with an informative error **before** any tensor
#' work if the request exceeds the model's declared capabilities.
#'
#' @param architecture Character scalar, the architecture key (e.g. `"ttm"`).
#' @param max_context Integer, the maximum supported context length.
#' @param max_horizon Numeric scalar, the maximum supported forecast horizon,
#'   or `Inf` when the architecture has no smaller declared limit.
#' @param quantiles One of `"native"` (the model emits quantiles directly) or
#'   `"none"`.
#' @param quantile_levels Numeric vector of quantile levels the checkpoint can
#'   emit, or `NULL` when a native-quantile architecture accepts arbitrary
#'   levels.
#' @param multivariate,past_covariates,future_covariates Grouped-input
#'   channels. They may be `TRUE` only for an architecture written against
#'   contract 1.1 or later; [new_zuk_model()] refuses the combination
#'   otherwise. See `?`[zuk-architecture-contract].
#' @param samples,static_covariates,fine_tunable Reserved logical fields. They
#'   must remain `FALSE`: no contract version has an execution channel for
#'   those inputs or outputs, and declaring one would describe behaviour that
#'   cannot run.
#' @param license Character scalar, the SPDX identifier of the *weight* licence.
#'
#' @return A `zuk_capabilities` object.
#' @export
#' @examples
#' new_zuk_capabilities("demo", max_context = 512L)
#'
#' # A checkpoint with fixed trained quantile levels declares them, and the
#' # engine then refuses any other level before running the forward pass.
#' new_zuk_capabilities(
#'   "demo",
#'   max_context = 512L,
#'   max_horizon = 64,
#'   quantile_levels = c(0.1, 0.5, 0.9),
#'   license = "Apache-2.0"
#' )
new_zuk_capabilities <- function(architecture,
                                  max_context,
                                  max_horizon = Inf,
                                  quantiles = c("native", "none"),
                                  quantile_levels = NULL,
                                  multivariate = FALSE,
                                  samples = FALSE,
                                  past_covariates = FALSE,
                                  future_covariates = FALSE,
                                  static_covariates = FALSE,
                                  fine_tunable = FALSE,
                                  license = NA_character_) {
  architecture <- as.character(architecture)
  max_context <- as.integer(max_context)
  max_horizon <- as.numeric(max_horizon)
  quantiles <- match.arg(quantiles)
  if (length(architecture) != 1L || is.na(architecture) || !nzchar(architecture)) {
    zuk_abort_contract(
      "{.arg architecture} must be one non-empty string.",
      contract = "capability declaration",
      expected = "one non-empty architecture key",
      actual = architecture
    )
  }
  if (length(max_context) != 1L || is.na(max_context) || max_context <= 0L) {
    zuk_abort_contract(
      "{.arg max_context} must be one positive integer.",
      architecture = architecture,
      contract = "capability declaration",
      expected = "one positive integer",
      actual = max_context
    )
  }
  if (length(max_horizon) != 1L || is.na(max_horizon) || max_horizon <= 0) {
    zuk_abort_contract(
      "{.arg max_horizon} must be positive or {.val Inf}.",
      architecture = architecture,
      contract = "capability declaration",
      expected = "one positive number or Inf",
      actual = max_horizon
    )
  }

  if (is.null(quantile_levels)) {
    quantile_levels <- if (identical(quantiles, "none")) numeric() else NULL
  } else {
    quantile_levels <- sort(unique(as.numeric(quantile_levels)))
    if (anyNA(quantile_levels) || any(!is.finite(quantile_levels)) ||
        any(quantile_levels <= 0 | quantile_levels >= 1)) {
      zuk_abort_contract(
        "{.arg quantile_levels} must contain unique finite values strictly between 0 and 1.",
        architecture = architecture,
        contract = "capability declaration",
        expected = "finite probabilities in (0, 1)",
        actual = quantile_levels
      )
    }
  }
  if (identical(quantiles, "none") && length(quantile_levels)) {
    zuk_abort_contract(
      "A point-only model cannot declare supported quantile levels.",
      architecture = architecture,
      contract = "capability declaration",
      expected = numeric(),
      actual = quantile_levels
    )
  }

  grouped <- list(
    multivariate = multivariate,
    past_covariates = past_covariates,
    future_covariates = future_covariates
  )
  reserved <- c(
    grouped,
    list(
      samples = samples,
      static_covariates = static_covariates,
      fine_tunable = fine_tunable
    )
  )
  # Checked as a list, not a vector: c() would coerce a stray numeric to a
  # common type, and isTRUE(1) is FALSE, so a non-logical declaration would slip
  # through the gate below and be silently recorded as disabled.
  malformed <- names(reserved)[!vapply(
    reserved,
    function(flag) is.logical(flag) && length(flag) == 1L && !is.na(flag),
    logical(1)
  )]
  if (length(malformed)) {
    zuk_abort_contract(
      c(
        "Capability flags must each be one non-missing logical value.",
        "x" = "Malformed: {.val {malformed}}."
      ),
      architecture = architecture,
      contract = "capability declaration",
      expected = "TRUE or FALSE",
      actual = reserved[malformed]
    )
  }
  # Only the channels that genuinely have no implementation are refused here.
  # The grouped-input flags are checked against the architecture's declared
  # contract version instead, which new_zuk_model() knows and this does not.
  never <- reserved[c("samples", "static_covariates", "fine_tunable")]
  if (any(vapply(never, isTRUE, logical(1)))) {
    enabled <- names(never)[vapply(never, isTRUE, logical(1))]
    zuk_abort_contract(
      c(
        "No contract version has an execution channel for the declared capability.",
        "x" = "Disable: {.val {enabled}}."
      ),
      architecture = architecture,
      contract = "capability declaration",
      expected = FALSE,
      actual = enabled
    )
  }

  structure(
    list(
      architecture      = architecture,
      max_context       = max_context,
      max_horizon       = max_horizon,
      quantiles         = quantiles,
      quantile_levels   = quantile_levels,
      multivariate      = isTRUE(multivariate),
      samples           = FALSE,
      past_covariates   = isTRUE(past_covariates),
      future_covariates = isTRUE(future_covariates),
      static_covariates = FALSE,
      fine_tunable      = FALSE,
      license           = as.character(license)
    ),
    class = "zuk_capabilities"
  )
}

#' Report the capabilities of a model
#'
#' @param x A `zuk_model` (or another object carrying capability metadata).
#' @param ... Unused, for future methods.
#' @return A [new_zuk_capabilities()] object.
#' @export
#' @examples
#' model <- zuk_pretrained("stub")
#' zuk_capabilities(model)
#'
#' zuk_capabilities(model)$max_context
#' zuk_unload("stub")
zuk_capabilities <- function(x, ...) {
  UseMethod("zuk_capabilities")
}

#' @export
zuk_capabilities.zuk_capabilities <- function(x, ...) {
  x
}

#' @export
format.zuk_capabilities <- function(x, ...) {
  yn <- function(v) if (isTRUE(v)) "TRUE" else "FALSE"
  levels <- if (is.null(x$quantile_levels)) {
    "arbitrary"
  } else if (!length(x$quantile_levels)) {
    "none"
  } else {
    paste(x$quantile_levels, collapse = ", ")
  }
  c(
    "<zuk_capabilities>",
    sprintf("  architecture:      %s", x$architecture),
    sprintf("  max_context:       %s        max_horizon: %s", x$max_context, x$max_horizon),
    sprintf("  quantiles:         %s        levels: %s", x$quantiles, levels),
    sprintf("  multivariate:      %s        samples:   %s", yn(x$multivariate), yn(x$samples)),
    sprintf("  past_covariates:   %s        future_covariates: %s",
            yn(x$past_covariates), yn(x$future_covariates)),
    sprintf("  static_covariates: %s        fine_tunable: %s",
            yn(x$static_covariates), yn(x$fine_tunable)),
    sprintf("  license:           %s", x$license)
  )
}

#' @export
print.zuk_capabilities <- function(x, ...) {
  cat(format(x, ...), sep = "\n")
  cat("\n")
  invisible(x)
}

# ---- Pre-flight validation --------------------------------------------------

# Assert a requested context length fits the model. `call` is the calling
# environment for cli's error attribution.
check_context_length <- function(caps, context_length,
                                 model_id = NA_character_,
                                 revision = NA_character_,
                                 call = rlang::caller_env()) {
  if (!is.null(context_length) && context_length > caps$max_context) {
    zuk_abort_context_length(
      c(
        "Requested context length exceeds the model's capability.",
        "x" = "Requested {.val {context_length}} observations of context.",
        "i" = "Architecture {.val {caps$architecture}} supports at most {.val {caps$max_context}}."
      ),
      model_id = model_id,
      revision = revision,
      requested = context_length,
      supported = caps$max_context,
      call = call
    )
  }
  invisible(caps)
}

check_horizon <- function(caps, h,
                          model_id = NA_character_,
                          revision = NA_character_,
                          call = rlang::caller_env()) {
  valid <- is.numeric(h) && length(h) > 0L && !anyNA(h) && all(is.finite(h)) &&
    all(h > 0) && all(h == floor(h)) && all(h <= .Machine$integer.max)
  if (!valid) {
    zuk_abort_capability(
      "Forecast horizons must be positive integers.",
      model_id = model_id,
      revision = revision,
      capability = "horizon",
      requested = h,
      supported = sprintf("positive integers up to %s", caps$max_horizon),
      call = call
    )
  }
  if (any(h > caps$max_horizon)) {
    zuk_abort_capability(
      c(
        "Requested horizon exceeds the model's capability.",
        "x" = "Requested a maximum horizon of {.val {max(h)}}.",
        "i" = "This checkpoint supports at most {.val {caps$max_horizon}}."
      ),
      model_id = model_id,
      revision = revision,
      capability = "horizon",
      requested = h,
      supported = caps$max_horizon,
      call = call
    )
  }
  invisible(caps)
}

# Match requested levels against the levels a checkpoint was trained on.
#
# Doubles that print identically need not be bit-identical: seq(0.1, 0.9, by =
# 0.1) differs from the literals parsed out of config.json at 0.3 and 0.7. Every
# native-quantile architecture selects its output channel *by position*, so the
# match is tolerant and the engine carries the matched supported value forward
# rather than whichever spelling the caller happened to use. Returns positions
# into `supported`, `NA` where a level has no counterpart.
zuk_match_quantile_levels <- function(levels, supported) {
  vapply(
    as.numeric(levels),
    function(level) {
      hit <- which(abs(supported - level) <= sqrt(.Machine$double.eps))
      if (length(hit)) as.integer(hit[[1L]]) else NA_integer_
    },
    integer(1)
  )
}

check_quantile_levels <- function(caps, quantile_levels,
                                  model_id = NA_character_,
                                  revision = NA_character_,
                                  call = rlang::caller_env()) {
  if (!is.numeric(quantile_levels)) {
    zuk_abort_quantile_levels(
      "Quantile levels must be supplied as a numeric vector.",
      model_id = model_id,
      revision = revision,
      requested = quantile_levels,
      supported = caps$quantile_levels,
      call = call
    )
  }
  levels <- as.numeric(quantile_levels)
  valid <- length(levels) > 0L && !anyNA(levels) && all(is.finite(levels)) &&
    all(levels > 0 & levels < 1)
  if (!valid || anyDuplicated(levels)) {
    zuk_abort_quantile_levels(
      "Quantile levels must be unique finite probabilities strictly between 0 and 1.",
      model_id = model_id,
      revision = revision,
      requested = quantile_levels,
      supported = caps$quantile_levels,
      call = call
    )
  }
  if (identical(caps$quantiles, "none")) {
    zuk_abort_quantile_levels(
      "This model does not emit predictive quantiles.",
      model_id = model_id,
      revision = revision,
      requested = levels,
      supported = numeric(),
      call = call
    )
  }
  supported <- caps$quantile_levels
  if (!is.null(supported)) {
    matched <- zuk_match_quantile_levels(levels, supported)
    if (anyNA(matched)) {
      zuk_abort_quantile_levels(
        c(
          "The checkpoint does not emit every requested quantile level.",
          "x" = "Unsupported: {.val {levels[is.na(matched)]}}.",
          "i" = "Supported levels: {.val {supported}}."
        ),
        model_id = model_id,
        revision = revision,
        requested = levels,
        supported = supported,
        call = call
      )
    }
    # Hand back the checkpoint's own values, so architectures selecting an
    # output channel never have to re-derive a match the engine already made.
    levels <- supported[matched]
    if (anyDuplicated(levels)) {
      zuk_abort_quantile_levels(
        c(
          "Several requested quantile levels resolve to the same trained level.",
          "x" = "Duplicated after matching: {.val {unique(levels[duplicated(levels)])}}.",
          "i" = "Supported levels: {.val {supported}}."
        ),
        model_id = model_id,
        revision = revision,
        requested = quantile_levels,
        supported = supported,
        call = call
      )
    }
  }
  invisible(sort(levels))
}

# Assert the request only asks for capabilities the model declares.
check_capabilities <- function(caps,
                               multivariate = FALSE,
                               past_covariates = FALSE,
                               future_covariates = FALSE,
                               static_covariates = FALSE,
                               call = rlang::caller_env()) {
  problems <- character()
  flag <- function(requested, supported, label) {
    if (isTRUE(requested) && !isTRUE(supported)) {
      problems <<- c(problems, sprintf("%s is not supported by this model.", label))
    }
  }
  flag(multivariate,      caps$multivariate,      "Multivariate targets")
  flag(past_covariates,   caps$past_covariates,   "Past covariates")
  flag(future_covariates, caps$future_covariates, "Future covariates")
  flag(static_covariates, caps$static_covariates, "Static covariates")

  if (length(problems)) {
    zuk_abort_capability(
      c(
        "The request exceeds architecture {.val {caps$architecture}}'s capabilities.",
        stats::setNames(problems, rep("x", length(problems)))
      ),
      capability = "input_channels",
      requested = c(
        multivariate = isTRUE(multivariate),
        past_covariates = isTRUE(past_covariates),
        future_covariates = isTRUE(future_covariates),
        static_covariates = isTRUE(static_covariates)
      ),
      supported = FALSE,
      call = call
    )
  }
  invisible(caps)
}
