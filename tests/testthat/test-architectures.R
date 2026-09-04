test_that("all built-in architectures are registered", {
  expect_setequal(zuk_registry_archs(), c("stub", "ttm", "timesfm", "toto2"))
  expect_false(zuk_registry_has("chronos2"))
})

test_that("architecture keys are normalised from Hub config conventions", {
  expect_identical(normalize_architecture(list(model_type = "TinyTimeMixer")), "ttm")
  expect_identical(
    normalize_architecture(list(architectures = "TinyTimeMixerForPrediction")),
    "ttm"
  )
  expect_identical(normalize_architecture(list(model_type = "TimesFM")), "timesfm")
  expect_identical(
    normalize_architecture(list(architectures = "PatchedTimeSeriesDecoder")),
    "timesfm"
  )
  expect_identical(normalize_architecture(list(model_type = "ChronosBolt")), "chronos2")
  expect_identical(normalize_architecture(list(model_type = "PatchTST")), "patchtst")
})

test_that("Chronos-2 is rejected before network or adapter work", {
  expect_true(is_chronos2_id("amazon/chronos-2"))
  expect_true(is_chronos2_id("amazon/chronos2"))
  expect_false(is_chronos2_id("google/timesfm-2.5-200m-pytorch"))

  expect_error(
    zuk_pretrained("amazon/chronos-2"),
    "not a supported model in zukzeit 0.1.0",
    fixed = TRUE
  )
})

test_that("TTM scaffold advertises capabilities but defers the forward pass", {
  model <- ttm_constructor(list(context_length = 1536L))
  expect_identical(model$architecture, "ttm")
  caps <- zuk_capabilities(model)
  expect_identical(caps$max_context, 1536L)
  expect_identical(caps$quantiles, "none")
  expect_false(caps$multivariate)
  expect_false(caps$past_covariates)
  expect_false(caps$future_covariates)
  expect_false(caps$fine_tunable)
  expect_identical(caps$license, "Apache-2.0")
  # Numerical port not done: forward pass must error, not fabricate output.
  expect_error(model$predict_fn(1:10, 3, c(0.1, 0.5, 0.9)), "not implemented")
})

test_that("TimesFM advertises only executable contract-v1 capabilities", {
  config <- timesfm_test_config()
  caps <- timesfm_capabilities(config)
  expect_identical(caps$max_context, 16256L)
  expect_identical(caps$max_horizon, 1024)
  expect_false(caps$multivariate)
  expect_false(caps$past_covariates)
  expect_false(caps$future_covariates)
  expect_false(caps$fine_tunable)
  expect_identical(caps$quantiles, "native")
  expect_identical(caps$license, "Apache-2.0")
})

test_that("TimesFM output channels are derived, not hardcoded", {
  layout <- timesfm_channel_layout(timesfm_test_config())
  # Channel 1 is the point forecast; 2..10 are the trained quantiles ascending,
  # so the median (0.5) is channel 6 and the crossing repair works outwards.
  expect_identical(layout$outputs, 10L)
  expect_identical(layout$quantile_channels, 2:10)
  expect_identical(layout$median_channel, 6L)
  expect_identical(layout$lower_channels, 2:5)
  expect_identical(layout$upper_channels, 7:10)
})

test_that("usable context shrinks as the horizon claims output blocks", {
  config <- timesfm_test_config()
  # 16,384 positions cover context and forecast together; the horizon rounds up
  # to whole 128-value blocks. max_context reports the best case only.
  expect_identical(timesfm_usable_context(1L, config), 16256L)
  expect_identical(timesfm_usable_context(128L, config), 16256L)
  expect_identical(timesfm_usable_context(129L, config), 16128L)
  expect_identical(timesfm_usable_context(1024L, config), 15360L)
  expect_identical(
    timesfm_capabilities(config)$max_context,
    timesfm_usable_context(1L, config)
  )
})

test_that("context truncation is per series, not per batch", {
  skip_if_no_torch()
  config <- timesfm_test_config()

  # A long series forecast at h = 6 keeps 16,000 observations. Batching it
  # beside a h = 1024 request must not shorten it: a forecast may not depend on
  # which other series happen to share its batch.
  long <- as.numeric(seq_len(16000L))
  alone <- timesfm_prepare_batch(list(long), 6L, config, "cpu")
  together <- timesfm_prepare_batch(list(long, 1:10), c(6L, 1024L), config, "cpu")

  observed_alone <- as.numeric(alone$inputs[1, ][!as.logical(alone$masks[1, ])])
  observed_together <- as.numeric(
    together$inputs[1, ][!as.logical(together$masks[1, ])]
  )
  expect_identical(length(observed_alone), 16000L)
  expect_equal(observed_together, observed_alone)

  # The horizon that does bind is the series' own.
  clipped <- timesfm_prepare_batch(list(long), 1024L, config, "cpu")
  expect_identical(
    sum(!as.logical(clipped$masks[1, ])),
    timesfm_usable_context(1024L, config)
  )
})

test_that("a horizon leaving no room for context is refused", {
  config <- timesfm_test_config()
  config$context_length <- 256L
  error <- expect_error(
    timesfm_prepare_batch(list(1:10), 256L, config, "cpu"),
    class = "zuk_error_capability"
  )
  expect_identical(error$capability, "horizon")
})

test_that("TimesFM rejects missing weights and incompatible variants", {
  config <- timesfm_test_config()
  expect_error(timesfm_constructor(config), class = "zuk_error_checkpoint")

  config$hidden_size <- 64L
  error <- expect_error(
    validate_timesfm_config(config),
    class = "zuk_error_checkpoint"
  )
  expect_identical(error$tensor, "config.json")
})
