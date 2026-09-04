# Conformance harness for the architecture contract.
#
# An engine is only as good as its guarantee to architecture authors. This file
# turns ?`zuk-architecture-contract` from prose into an executable gate: every
# built-in architecture runs it, and external implementations can run the same
# check without needing to read the engine's internals.

# One check outcome. `ok = NA` means the check could not be run (for example a
# batch check on an architecture that supplies no predict_batch_fn) --- which is
# a pass, not a failure.
new_check_result <- function(name, ok, message = NULL) {
  list(name = name, ok = ok, message = message)
}

# Run one invariant, converting any error into a failed check so a broken
# architecture yields a report rather than a stack trace. `fn` is a zero-argument
# function returning NULL to pass, or a message describing the violation.
run_check <- function(name, fn) {
  tryCatch(
    {
      msg <- fn()
      if (is.null(msg)) new_check_result(name, TRUE) else new_check_result(name, FALSE, msg)
    },
    error = function(e) new_check_result(name, FALSE, conditionMessage(e))
  )
}

# Evaluate something that may fail, keeping the value on success and the error
# message on failure. Used for the two steps later checks depend on (building
# the model, and the probe forward pass).
try_value <- function(fn) {
  tryCatch(
    list(ok = TRUE, value = fn(), message = NULL),
    error = function(e) list(ok = FALSE, value = NULL, message = conditionMessage(e))
  )
}

# A deterministic, well-behaved context: trend + seasonality + mild noise. Fixed
# seed so failures are reproducible and diffable across runs --- but restored on
# exit, because this runs as a default argument of an exported function and
# leaving `set.seed()` in place would silently reseed the caller's session.
default_check_context <- function(n = 64L) {
  with_fixed_seed(20260813L, {
    seq_len(n) * 0.5 + 10 * sin(seq_len(n) * 2 * pi / 12) +
      stats::rnorm(n, sd = 0.5) + 100
  })
}

with_fixed_seed <- function(seed, expr) {
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    previous <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
    on.exit(assign(".Random.seed", previous, envir = globalenv()), add = TRUE)
  } else {
    # No RNG state to restore: leave the session as it was found, unseeded.
    on.exit(
      suppressWarnings(rm(".Random.seed", envir = globalenv())),
      add = TRUE
    )
  }
  set.seed(seed)
  expr
}

#' Verify an architecture against the zukzeit contract
#'
#' Runs the invariants in `?`[zuk-architecture-contract] against an
#' architecture constructor and reports which hold. Use it while implementing a
#' new architecture, and in that architecture's test suite as a regression gate.
#'
#' This checks *contract conformance*, not *numerical correctness*. An
#' architecture can pass every check here and still forecast badly; matching the
#' reference implementation is a separate gate, covered by golden parity
#' fixtures (see `tests/testthat/fixtures/README.md`).
#'
#' @param constructor A function `(config, weights)` returning a
#'   [new_zuk_model()], as passed to [zuk_register_arch()].
#' @param config A config list handed to `constructor`. Supply whatever the
#'   architecture needs; the default is empty.
#' @param weights Weights handed to `constructor`. Default `NULL`.
#' @param context Numeric context vector used for the probe forecasts. Defaults
#'   to a deterministic trend-plus-seasonality series of length 64.
#' @param h Probe horizon.
#' @param quantile_levels Probe quantile levels.
#' @param tolerance Numerical tolerance for the batch-agreement check.
#' @param crossing_tolerance How far quantiles may cross before the
#'   monotonicity check fails. Defaults to `tolerance`.
#' @param error If `TRUE` (the default when called non-interactively), fail on
#'   the first violated invariant. If `FALSE`, return the full report so every
#'   problem is visible at once.
#'
#' @return Invisibly, a data frame with one row per check: `name`, `ok`
#'   (`TRUE`, `FALSE`, or `NA` for not applicable), and `message`.
#'
#' @seealso `?`[zuk-architecture-contract]
#' @export
#' @examples
#' # A minimal conforming architecture: last value, with a Gaussian spread.
#' demo_arch <- function(config, weights) {
#'   new_zuk_model(
#'     architecture = "demo",
#'     config = config,
#'     capabilities = new_zuk_capabilities("demo", max_context = 128L),
#'     predict_fn = function(context, h, quantile_levels) {
#'       if (length(context) == 0L) stop("empty context")
#'       last <- context[length(context)]
#'       outer(rep(last, h), stats::qnorm(quantile_levels), `+`)
#'     }
#'   )
#' }
#'
#' report <- zuk_check_architecture(demo_arch, error = FALSE)
#' report
zuk_check_architecture <- function(constructor,
                                    config = list(),
                                    weights = NULL,
                                    context = default_check_context(),
                                    h = 6L,
                                    quantile_levels = c(0.1, 0.5, 0.9),
                                    tolerance = 1e-8,
                                    crossing_tolerance = tolerance,
                                    error = !interactive()) {
  if (!is.function(constructor)) {
    zuk_abort_contract(
      "{.arg constructor} must be a function of {.code (config, weights)}.",
      contract = "conformance input",
      expected = "function(config, weights)",
      actual = class(constructor)
    )
  }
  h <- as.integer(h)
  quantile_levels <- sort(as.numeric(quantile_levels))
  q <- length(quantile_levels)
  checks <- list()

  # --- construction --------------------------------------------------------
  built <- try_value(function() constructor(config, weights))
  model <- built$value
  checks$constructor <- run_check("constructor returns a zuk_model", function() {
    if (!built$ok) {
      sprintf("Constructor raised an error: %s", built$message)
    } else if (!inherits(model, "zuk_model")) {
      sprintf("Got a <%s>; expected a <zuk_model> from new_zuk_model().",
              paste(class(model), collapse = "/"))
    } else {
      NULL
    }
  })
  if (!isTRUE(checks$constructor$ok)) {
    return(finish_check(checks, error))
  }

  checks$capabilities <- run_check("declares capabilities", function() {
    caps <- model$capabilities
    if (!inherits(caps, "zuk_capabilities")) {
      "capabilities must be a <zuk_capabilities> from new_zuk_capabilities()."
    } else if (!is.finite(caps$max_context) || caps$max_context <= 0L) {
      sprintf("max_context must be a positive integer; got %s.", caps$max_context)
    } else {
      NULL
    }
  })

  checks$contract_version <- run_check("stamps a contract version", function() {
    v <- model$contract_version
    if (is.null(v)) {
      "No contract_version on the model handle; construct it with new_zuk_model()."
    } else if (package_version(v)$major != zuk_contract_version()$major) {
      sprintf("Architecture targets contract %s; this zukzeit implements %s.",
              format(v), format(zuk_contract_version()))
    } else {
      NULL
    }
  })

  # Probe context, truncated the way the engine would truncate it.
  max_context <- model$capabilities$max_context %||% length(context)
  probe <- utils::tail(as.numeric(context), max_context)

  # --- the forward pass ----------------------------------------------------
  probed <- try_value(function() model$predict_fn(probe, h, quantile_levels))
  pred <- probed$value
  checks$shape <- run_check("predict_fn returns an h x q numeric matrix", function() {
    if (!probed$ok) {
      sprintf("predict_fn raised an error: %s", probed$message)
    } else if (!is.matrix(pred) || !is.numeric(pred)) {
      sprintf("Expected a numeric matrix; got a <%s>.",
              paste(class(pred), collapse = "/"))
    } else if (!identical(dim(pred), c(h, q))) {
      sprintf("Expected dim c(%d, %d); got c(%s).",
              h, q, paste(dim(pred), collapse = ", "))
    } else {
      NULL
    }
  })

  if (isTRUE(checks$shape$ok)) {
    checks$finite <- run_check("forecasts are finite", function() {
      bad <- sum(!is.finite(pred))
      if (bad > 0L) sprintf("%d of %d values are NA, NaN, or Inf.", bad, length(pred)) else NULL
    })

    checks$monotone <- run_check("quantiles are non-decreasing across levels", function() {
      if (q < 2L) {
        return(NULL)
      }
      crossed <- which(apply(
        pred, 1L, function(row) any(diff(row) < -crossing_tolerance)
      ))
      if (length(crossed)) {
        sprintf(
          "Quantiles cross at horizon step(s) %s. Sort across levels before returning.",
          paste(crossed, collapse = ", ")
        )
      } else {
        NULL
      }
    })
  } else {
    checks$finite <- new_check_result("forecasts are finite", NA)
    checks$monotone <- new_check_result("quantiles are non-decreasing across levels", NA)
  }

  # --- edge cases ----------------------------------------------------------
  checks$empty_context <- run_check("empty context errors rather than returning NA", function() {
    out <- tryCatch(model$predict_fn(numeric(0), h, quantile_levels),
                    error = function(e) structure(list(), class = "expected_error"))
    if (inherits(out, "expected_error")) NULL else "Returned a value; expected an error."
  })

  checks$short_context <- run_check("a single observation is handled or refused cleanly", function() {
    out <- tryCatch(model$predict_fn(probe[length(probe)], h, quantile_levels),
                    error = function(e) structure(list(), class = "expected_error"))
    if (inherits(out, "expected_error")) {
      NULL  # refusing is a valid answer
    } else if (!is.matrix(out) || !identical(dim(out), c(h, q))) {
      "Returned a value of the wrong shape; either forecast correctly or error."
    } else if (any(!is.finite(out))) {
      "Returned non-finite values; either forecast correctly or error."
    } else {
      NULL
    }
  })

  # A probe shorter than max_context cannot reach the limit. Report that as
  # not-applicable: silently passing an unexercised check would overstate what
  # the harness verified, and for a long-context model the default probe never
  # comes close.
  over <- c(rep(probe[1], 8L), probe)
  checks$context_limit <- if (length(over) <= max_context) {
    new_check_result(
      "respects its declared max_context", NA,
      sprintf(
        paste0("Not exercised: the probe context (%d values) is shorter than ",
               "max_context (%d). Pass a longer `context` to check the limit."),
        length(over), max_context
      )
    )
  } else {
    run_check("respects its declared max_context", function() {
      out <- model$predict_fn(utils::tail(over, max_context), h, quantile_levels)
      if (!is.matrix(out) || !identical(dim(out), c(h, q))) {
        "A context at exactly max_context did not produce a valid forecast."
      } else {
        NULL
      }
    })
  }

  # --- batch path ----------------------------------------------------------
  if (is.function(model$predict_batch_fn)) {
    checks$batch_agrees <- run_check("predict_batch_fn agrees with predict_fn", function() {
      if (!isTRUE(checks$shape$ok)) {
        return("Cannot compare: the predict_fn probe did not produce a valid forecast.")
      }
      device <- zuk_resolve_device(model$device %||% NULL)
      batched <- model$predict_batch_fn(list(probe, probe), c(h, h),
                                        quantile_levels, device = device)
      if (!is.list(batched) || length(batched) != 2L) {
        return(sprintf("Expected a list of 2 matrices; got %s of length %d.",
                       class(batched)[1], length(batched)))
      }
      # Both entries, not just the first: a batch path that returns the right
      # number of matrices with the wrong shapes in the tail would otherwise
      # pass, and the whole point of the check is that the two paths are the
      # same model.
      shapes <- vapply(
        batched,
        function(m) is.matrix(m) && is.numeric(m) && identical(dim(m), c(h, q)),
        logical(1)
      )
      if (!all(shapes)) {
        return(sprintf(
          "Batch element(s) %s are not %d x %d numeric matrices.",
          paste(which(!shapes), collapse = ", "), h, q
        ))
      }
      diffs <- vapply(
        batched,
        function(m) max(abs(as.numeric(m) - as.numeric(pred))),
        numeric(1)
      )
      worst <- max(diffs)
      if (!is.finite(worst) || worst > tolerance) {
        sprintf("Batch and loop paths differ by %.3g (tolerance %.3g).",
                worst, tolerance)
      } else {
        NULL
      }
    })
  } else {
    checks$batch_agrees <- new_check_result(
      "predict_batch_fn agrees with predict_fn", NA,
      "No predict_batch_fn supplied; the engine will loop predict_fn."
    )
  }

  finish_check(checks, error)
}

# Assemble the report, print it, and stop if any invariant was violated.
finish_check <- function(checks, error) {
  report <- data.frame(
    name    = vapply(checks, `[[`, character(1), "name"),
    ok      = vapply(checks, function(x) as.logical(x$ok), logical(1)),
    message = vapply(checks, function(x) x$message %||% NA_character_, character(1)),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
  failed <- report[!is.na(report$ok) & !report$ok, , drop = FALSE]

  if (nrow(failed) == 0L) {
    cli::cli_alert_success(
      "Architecture satisfies the zukzeit contract ({sum(report$ok, na.rm = TRUE)} check{?s} passed)."
    )
    return(invisible(report))
  }

  bullets <- stats::setNames(
    sprintf("%s: %s", failed$name, failed$message),
    rep("x", nrow(failed))
  )
  msg <- c(
    "Architecture violates the zukzeit contract ({nrow(failed)} of {nrow(report)} check{?s} failed).",
    bullets,
    "i" = "See {.code ?`zuk-architecture-contract`}."
  )
  if (isTRUE(error)) {
    zuk_abort_contract(
      msg,
      contract = "architecture conformance",
      expected = "all checks pass",
      actual = failed
    )
  } else {
    cli::cli_warn(msg)
  }
  invisible(report)
}
