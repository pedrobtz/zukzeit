test_that("capabilities constructor coerces and defaults", {
  caps <- new_zuk_capabilities(
    "timesfm",
    max_context = 1536,
    max_horizon = 1024,
    quantiles = "native",
    quantile_levels = c(0.9, 0.1, 0.5),
    license = "Apache-2.0"
  )
  expect_s3_class(caps, "zuk_capabilities")
  expect_identical(caps$max_context, 1536L)
  expect_identical(caps$max_horizon, 1024)
  expect_equal(caps$quantile_levels, c(0.1, 0.5, 0.9))
  expect_false(caps$multivariate)
  expect_false(caps$samples)
  expect_identical(caps$license, "Apache-2.0")
})

test_that("declarations with no execution channel are refused", {
  # Sample paths and fine-tuning have no channel at any contract version, so
  # the capability record itself refuses them.
  error <- expect_error(
    new_zuk_capabilities("bad", 128L, samples = TRUE),
    class = "zuk_error_contract"
  )
  expect_s3_class(error, "zuk_error_internal")
  expect_identical(error$contract, "capability declaration")

  # The grouped-input flags do have a channel, but only from contract 1.1. The
  # capability record cannot judge that -- it does not know which version the
  # architecture targets -- so new_zuk_model() enforces it instead.
  expect_s3_class(
    new_zuk_capabilities("ok", 128L, multivariate = TRUE), "zuk_capabilities"
  )
  expect_error(
    new_zuk_model("bad", list(), new_zuk_capabilities("bad", 128L, multivariate = TRUE),
                  function(context, h, quantile_levels) matrix(0, h, 1L)),
    class = "zuk_error_contract"
  )
})

test_that("capabilities print is informative", {
  caps <- new_zuk_capabilities("stub", max_context = 512)
  out <- capture.output(print(caps))
  expect_true(any(grepl("architecture:", out)))
  expect_true(any(grepl("license:", out)))
})

test_that("pre-flight rejects requests beyond capability", {
  caps <- new_zuk_capabilities("stub", max_context = 100, multivariate = FALSE)
  expect_error(check_context_length(caps, 200), "context length")
  expect_error(check_capabilities(caps, multivariate = TRUE), "Multivariate")
  expect_error(check_capabilities(caps, future_covariates = TRUE), "covariates")
  expect_silent(check_capabilities(caps, multivariate = FALSE))
  expect_silent(check_context_length(caps, 100))
})

test_that("explicit quantile levels and horizon are enforced", {
  caps <- new_zuk_capabilities(
    "timesfm", 16384L,
    max_horizon = 1024L,
    quantile_levels = c(0.1, 0.5, 0.9)
  )
  expect_equal(check_quantile_levels(caps, c(0.9, 0.1)), c(0.1, 0.9))
  expect_error(
    check_quantile_levels(caps, c(0.05, 0.5)),
    class = "zuk_error_quantile_levels"
  )
  expect_error(check_horizon(caps, 0), class = "zuk_error_capability")
  expect_error(check_horizon(caps, 1025), class = "zuk_error_capability")
  expect_silent(check_horizon(caps, 1024))
})

# seq(0.1, 0.9, by = 0.1) accumulates rounding error, so its 3rd and 7th values
# are not the doubles a JSON config parses for 0.3 and 0.7. Both spellings name
# the same trained levels, and the catalogue and config.json each use a
# different one, so the engine must reconcile them rather than compare bits.
test_that("the two spellings of the trained levels really do differ", {
  literal <- jsonlite::fromJSON("[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]")
  generated <- seq(0.1, 0.9, by = 0.1)
  expect_false(identical(literal, generated))
  expect_equal(literal, generated)
  expect_identical(which(literal != generated), c(3L, 7L))
})

test_that("requested quantile levels are matched tolerantly and canonicalised", {
  literal <- jsonlite::fromJSON("[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]")
  caps <- new_zuk_capabilities("timesfm", 16384L, quantile_levels = literal)

  resolved <- check_quantile_levels(caps, seq(0.1, 0.9, by = 0.1))
  # The checkpoint's own values come back, not the caller's spelling: an
  # architecture selecting an output channel by position must not have to
  # re-derive a match the engine already made.
  expect_identical(resolved, literal)

  expect_identical(zuk_match_quantile_levels(seq(0.1, 0.9, by = 0.1), literal), 1:9)
  expect_identical(zuk_match_quantile_levels(c(0.05, 0.5), literal), c(NA_integer_, 5L))
})

test_that("levels collapsing onto one trained level are rejected", {
  literal <- jsonlite::fromJSON("[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]")
  caps <- new_zuk_capabilities("timesfm", 16384L, quantile_levels = literal)
  error <- expect_error(
    check_quantile_levels(caps, c(0.3, 0.30000000000000004, 0.5)),
    class = "zuk_error_quantile_levels"
  )
  expect_s3_class(error, "zuk_error_recoverable")
})

test_that("the median is resolved in the checkpoint's own spelling", {
  # seq(0.1, 0.9, by = 0.1) and the literals parsed from a config.json differ in
  # the last bits at 0.3 and 0.7. Appending a raw 0.5 to already-reconciled
  # levels risked carrying two spellings of one trained level into the engine,
  # which check_quantile_levels() then rejects as a duplicate.
  trained <- c(0.1, 0.25, 0.5, 0.75, 0.9)
  caps <- new_zuk_capabilities("demo", max_context = 64L,
                                quantile_levels = trained)

  resolved <- zukzeit:::resolve_median_level(caps, c(0.1, 0.9))
  expect_identical(resolved, trained[[3]])

  # A checkpoint with no trained median falls back to the requested level
  # nearest it, rather than failing an otherwise supported request.
  tailed <- new_zuk_capabilities("demo", max_context = 64L,
                                  quantile_levels = c(0.05, 0.4, 0.95))
  expect_identical(zukzeit:::resolve_median_level(tailed, c(0.05, 0.4)), 0.4)

  # An architecture that accepts arbitrary levels takes the exact median.
  free <- new_zuk_capabilities("demo", max_context = 64L)
  expect_identical(zukzeit:::resolve_median_level(free, c(0.1, 0.9)), 0.5)
})

test_that("a checkpoint without a trained median still forecasts", {
  skip_if_not_installed("distributional")

  levels <- c(0.05, 0.4, 0.95)
  arch <- function(config, weights) {
    new_zuk_model(
      architecture = "tailed",
      config = config,
      capabilities = new_zuk_capabilities("tailed", max_context = 64L,
                                           quantile_levels = levels),
      predict_fn = function(context, h, quantile_levels) {
        outer(rep(context[length(context)], h), stats::qnorm(quantile_levels), `+`)
      }
    )
  }
  model <- arch(list(), NULL)
  history <- data.frame(t = 1:30, y = as.numeric(1:30))

  fc <- forecast(model, history, h = 2, index = "t", target = "y",
                 quantile_levels = c(0.05, 0.4))
  expect_s3_class(fc, "zuk_forecast")
  expect_identical(attr(fc, "quantile_levels"), c(0.05, 0.4))
})

test_that("capability flags must be logical, not merely not-TRUE", {
  # c(multivariate = 1, samples = FALSE) coerces to numeric, and isTRUE(1) is
  # FALSE, so a non-logical declaration used to pass the reserved-field gate and
  # then be recorded as disabled.
  expect_error(
    new_zuk_capabilities("demo", max_context = 64L, multivariate = 1),
    class = "zuk_error_contract"
  )
  expect_error(
    new_zuk_capabilities("demo", max_context = 64L, samples = NA),
    class = "zuk_error_contract"
  )
  expect_error(
    new_zuk_capabilities("demo", max_context = 64L, fine_tunable = "no"),
    class = "zuk_error_contract"
  )
})
