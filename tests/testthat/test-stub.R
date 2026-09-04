test_that("zuk_pretrained('stub') loads without network or torch", {
  model <- zuk_pretrained("stub")
  expect_s3_class(model, "zuk_model")
  expect_identical(model$architecture, "stub")
  expect_identical(zuk_capabilities(model)$quantiles, "native")
})

test_that("stub forecasts are monotone quantiles centred on the last value", {
  set.seed(1)
  ctx <- cumsum(rnorm(50)) + 100
  q <- stub_predict_batch(ctx, h = 5, quantile_levels = c(0.1, 0.5, 0.9))
  expect_equal(dim(q), c(5L, 3L))
  expect_true(all(q[, 1] <= q[, 2]) && all(q[, 2] <= q[, 3]))
  expect_equal(q[, 2], rep(ctx[length(ctx)], 5))
  expect_true(all(diff(q[, 3] - q[, 1]) > 0))  # spread grows with horizon
})

test_that("stub handles constant and degenerate context", {
  q <- stub_predict_batch(rep(7, 10), h = 3, quantile_levels = c(0.1, 0.5, 0.9))
  expect_true(all(is.finite(q)))
  expect_equal(q[, 2], rep(7, 3))
  expect_error(stub_predict_batch(numeric(0), h = 2, quantile_levels = 0.5),
               "at least one")
})
