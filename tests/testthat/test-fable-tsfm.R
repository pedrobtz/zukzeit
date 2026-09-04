# The tidyverts model definition. Stage 4's composable user journey:
# fabletools::model(TSFM(...)) must work like any other tidyverts model, and
# must carry the checkpoint's identity through to the fitted object.

skip_if_no_fable <- function() {
  skip_if_not_installed("fabletools")
  skip_if_not_installed("tsibble")
}

fable_panel <- function(n = 24L, keys = c("a", "b")) {
  set.seed(20260813L)
  rows <- lapply(keys, function(k) {
    data.frame(
      t = seq_len(n),
      v = as.numeric(100 + cumsum(stats::rnorm(n))),
      g = k,
      stringsAsFactors = FALSE
    )
  })
  tsibble::as_tsibble(do.call(rbind, rows), key = "g", index = "t")
}

test_that("TSFM composes inside fabletools::model() and forecasts a fable", {
  skip_if_no_fable()
  fits <- fabletools::model(fable_panel(), zukzeit = TSFM(v, model_id = "stub"))

  expect_s3_class(fits, "mdl_df")
  expect_identical(nrow(fits), 2L)
  expect_true("zukzeit" %in% names(fits))

  fc <- fabletools::forecast(fits, h = 6L)
  expect_s3_class(fc, "fbl_ts")
  # One row per key per horizon step, and a real distribution column.
  expect_identical(nrow(fc), 12L)
  expect_true(all(is.finite(fc$.mean)))
  expect_s3_class(fc$v, "distribution")
})

test_that("the fitted model carries checkpoint and column identity", {
  skip_if_no_fable()
  fits <- fabletools::model(
    fable_panel(),
    zukzeit = TSFM(v, model_id = "stub", quantile_levels = c(0.2, 0.5, 0.8))
  )
  fit <- fits$zukzeit[[1]]$fit

  expect_s3_class(fit, "model_tsfm")
  expect_identical(fit$model$model_id, "stub")
  expect_identical(fit$model$revision, "main")
  expect_identical(fit$model$architecture, "stub")
  expect_identical(fit$response, "v")
  expect_identical(fit$index, "t")
  expect_equal(fit$quantile_levels, c(0.2, 0.5, 0.8))
  expect_identical(length(fit$history), 24L)
  # The key is fabletools' to carry, and it must survive into the fable.
  fc <- fabletools::forecast(fits, h = 3L)
  expect_true("g" %in% names(fc))
  expect_setequal(unique(fc$g), c("a", "b"))
})

test_that("an explicit revision is carried onto the handle", {
  skip_if_no_fable()
  fits <- fabletools::model(
    fable_panel(keys = "a"),
    zukzeit = TSFM(v, model_id = "stub", revision = "abc123")
  )
  expect_identical(fits$zukzeit[[1]]$fit$model$revision, "abc123")
})

test_that("TSFM binds each key's own history", {
  skip_if_no_fable()
  panel <- fable_panel()
  fits <- fabletools::model(panel, zukzeit = TSFM(v, model_id = "stub"))
  histories <- lapply(fits$zukzeit, function(m) m$fit$history)

  expect_length(histories, 2L)
  expect_false(isTRUE(all.equal(histories[[1]], histories[[2]])))
  # Each fit holds exactly the observations of its own key, in order.
  by_key <- split(panel$v, panel$g)
  expect_equal(histories[[1]], as.numeric(by_key[["a"]]))
  expect_equal(histories[[2]], as.numeric(by_key[["b"]]))
})

test_that("keys share one constructed handle rather than reloading per key", {
  skip_if_no_fable()
  # fabletools trains per key, so without resident reuse a wide panel would
  # rebuild the checkpoint once per series. Every fit must be the same handle.
  old <- options(zuk.max_loaded_models = 1L)
  on.exit(options(old), add = TRUE)
  zuk_unload()

  fits <- fabletools::model(
    fable_panel(keys = c("a", "b", "c")),
    zukzeit = TSFM(v, model_id = "stub")
  )
  handles <- lapply(fits$zukzeit, function(m) m$fit$model)
  expect_length(handles, 3L)
  expect_true(identical(handles[[1]], handles[[2]]))
  expect_true(identical(handles[[2]], handles[[3]]))
})

test_that("a TSFM fable flows into fabletools::accuracy()", {
  skip_if_no_fable()
  panel <- fable_panel(n = 30L, keys = "a")
  train <- utils::head(panel, 24L)
  fits <- fabletools::model(train, zukzeit = TSFM(v, model_id = "stub"))
  fc <- fabletools::forecast(fits, h = 6L)

  acc <- fabletools::accuracy(fc, panel)
  expect_identical(nrow(acc), 1L)
  expect_identical(acc$.type, "Test")
  expect_true(is.finite(acc$RMSE))
  expect_true(is.finite(acc$MAE))
})

test_that("a zero-shot fit reports no in-sample values rather than inventing them", {
  skip_if_no_fable()
  fits <- fabletools::model(fable_panel(keys = "a"), zukzeit = TSFM(v, model_id = "stub"))
  aug <- fabletools::augment(fits)
  expect_true(all(is.na(aug$.fitted)))
  expect_true(all(is.na(aug$.resid)))
})

test_that("TSFM raises typed conditions for a bad response or levels", {
  skip_if_no_fable()
  chr <- tsibble::as_tsibble(
    data.frame(t = 1:10, v = letters[1:10], stringsAsFactors = FALSE),
    index = "t"
  )
  # Asserted at the point of failure: fabletools captures per-key training
  # errors (see below), so the mable is not where the class survives.
  expect_error(
    train_tsfm(chr, specials = NULL, model_id = "stub"),
    class = "zuk_error_capability"
  )
  expect_error(
    train_tsfm(
      fable_panel(keys = "a"),
      specials = NULL, model_id = "stub", quantile_levels = c(0, 0.5)
    ),
    class = "zuk_error_quantile_levels"
  )
})

test_that("a failing key degrades to a null model rather than killing the panel", {
  skip_if_no_fable()
  # fabletools converts a training error into a warning and a null model, so one
  # unusable series does not discard the whole panel. Our message still reaches
  # the user; only the condition class is absorbed.
  chr <- tsibble::as_tsibble(
    data.frame(t = 1:10, v = letters[1:10], stringsAsFactors = FALSE),
    index = "t"
  )
  expect_warning(
    fits <- fabletools::model(chr, zukzeit = TSFM(v, model_id = "stub")),
    "must be numeric"
  )
  expect_true(fabletools::is_null_model(fits$zukzeit[[1]]))
})

test_that("the batched panel route and TSFM agree on the same history", {
  skip_if_no_fable()
  panel <- fable_panel(keys = "a")
  levels <- c(0.1, 0.5, 0.9)

  model <- zuk_pretrained("stub")
  batched <- forecast(model, panel, h = 6L, quantile_levels = levels)
  composed <- fabletools::forecast(
    fabletools::model(panel, zukzeit = TSFM(v, model_id = "stub",
                                         quantile_levels = levels)),
    h = 6L
  )
  # Same engine, same context, same horizon: the composable route must not be a
  # different model from the batched one.
  expect_equal(as.numeric(composed$.mean), as.numeric(batched$.mean))
})
