# Structural invariants of the native Toto port, on synthetic weights so they
# run without the checkpoint. Numerical parity against the pinned reference is
# a separate gate; see test-parity-toto.R.

test_that("the synthetic weight map covers the module exactly", {
  skip_if_no_torch()
  config <- toto_synthetic_config()
  module <- toto_module(config)
  map <- toto_weight_map(config)

  expect_length(map, 48L)
  expect_setequal(unname(map), names(module$state_dict()))
  expect_identical(anyDuplicated(names(map)), 0L)
})

test_that("the constructor yields a conforming architecture", {
  skip_if_no_torch()
  model <- toto_synthetic_model()

  expect_s3_class(model, "zuk_model")
  expect_identical(model$architecture, "toto2")
  expect_identical(model$capabilities$max_context, 4096L)
  expect_equal(model$capabilities$quantile_levels, seq(0.1, 0.9, by = 0.1))
  expect_false(model$capabilities$multivariate)
  expect_identical(model$capabilities$license, "Apache-2.0")
})

test_that("Toto passes the architecture contract", {
  skip_if_no_torch()
  config <- toto_synthetic_config()
  weights <- toto_synthetic_weights(config)

  report <- zuk_check_architecture(
    function(cfg, w) toto_constructor(config, weights),
    weights = weights,
    context = 100 + 0.5 * seq_len(320) + 10 * sin(seq_len(320) * 2 * pi / 24),
    h = 24L,
    quantile_levels = c(0.1, 0.5, 0.9),
    error = FALSE
  )
  expect_false(any(!is.na(report$ok) & !report$ok))
})

test_that("a forecast does not depend on which series share its batch", {
  skip_if_no_torch()
  model <- toto_synthetic_model()
  levels <- seq(0.1, 0.9, by = 0.1)
  long <- 100 + 0.5 * seq_len(200) + 10 * sin(seq_len(200) * 2 * pi / 24)
  short <- 50 - 0.2 * seq_len(73) + 4 * cos(seq_len(73) * 2 * pi / 12)

  # Unequal lengths are the point: the shorter series is left-padded to the
  # batch width, creating fully unobserved leading patches that must be
  # excluded from attention. Without that exclusion this forecast moves.
  batched <- model$predict_batch_fn(list(long, short), c(12L, 12L), levels,
                                    device = "cpu")
  alone <- lapply(list(long, short), function(series) {
    model$predict_fn(series, 12L, levels)
  })
  for (i in seq_along(alone)) {
    expect_equal(batched[[i]], alone[[i]], tolerance = 1e-5)
  }
})

test_that("inference is deterministic, quiet, and monotone", {
  skip_if_no_torch()
  model <- toto_synthetic_model()
  levels <- seq(0.1, 0.9, by = 0.1)
  series <- 100 + cumsum(rep(0.5, 160))

  expect_silent(first <- model$predict_fn(series, 16L, levels))
  second <- model$predict_fn(series, 16L, levels)
  expect_identical(first, second)

  expect_identical(dim(first), c(16L, 9L))
  expect_true(all(is.finite(first)))
  expect_true(all(apply(first, 1L, function(row) all(diff(row) >= 0))))
})

test_that("unsupported requests are refused before any tensor work", {
  skip_if_no_torch()
  model <- toto_synthetic_model()

  expect_error(
    zuk_run_batches(model, list(rnorm(100)), 12L, c(0.15, 0.5)),
    class = "zuk_error_quantile_levels"
  )
  expect_error(
    zuk_run_batches(model, list(rnorm(100)), 5000L, c(0.1, 0.5, 0.9)),
    class = "zuk_error_capability"
  )
})

test_that("an incompatible checkpoint variant is rejected", {
  skip_if_no_torch()
  config <- toto_synthetic_config()
  weights <- toto_synthetic_weights(config)

  wrong_ratio <- config
  wrong_ratio$residual_attn_ratio <- 4.0
  expect_error(toto_constructor(wrong_ratio, weights),
               class = "zuk_error_checkpoint")

  wrong_flag <- config
  wrong_flag$use_xpos <- FALSE
  expect_error(toto_constructor(wrong_flag, weights),
               class = "zuk_error_checkpoint")

  expect_error(toto_constructor(config, NULL), class = "zuk_error_checkpoint")
})
