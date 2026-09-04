# Stage 4 exit gate: the four required 0.1.0 user journeys, driven by the real
# pinned checkpoint rather than the stub.
#
# Opt-in. These need the 925 MB checkpoint in the hfhub cache; the deterministic
# suite proves the same paths against the stub. Handles are pinned to CPU: the
# default "auto" resolves to CUDA or MPS on an accelerator host, and this gate is
# about the documented portable baseline.

skip_unless_checkpoint <- function() {
  skip_if_not(identical(Sys.getenv("ZUK_RUN_CHECKPOINT_TEST"), "true"))
  skip_if_no_torch()
  record <- zuk_catalogue_get("google/timesfm-2.5-200m-pytorch")
  skip_if_not(identical(record$state, "supported"), "TimesFM is not supported.")
  invisible(record)
}

smoke_panel <- function(n = 96L, keys = c("store_a", "store_b")) {
  set.seed(20260813L)
  rows <- lapply(keys, function(k) {
    data.frame(
      date = as.Date("2024-01-01") + seq_len(n) - 1L,
      sales = as.numeric(
        100 + seq_len(n) * 0.3 + 8 * sin(seq_len(n) * 2 * pi / 7) +
          stats::rnorm(n, sd = 1.5)
      ),
      store = k,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

test_that("journey 1: plain-R panel forecasting from a data frame", {
  record <- skip_unless_checkpoint()
  model <- zuk_pretrained(record$model_id, revision = record$revision,
                           device = "cpu")
  panel <- smoke_panel()

  fc <- forecast(model, panel, h = 14L, index = "date", key = "store",
                 target = "sales", quantile_levels = c(0.1, 0.5, 0.9))

  expect_s3_class(fc, "zuk_forecast")
  expect_identical(nrow(fc), 28L)
  # Stable ordering: keys in sorted order, dates ascending within each key.
  expect_identical(unique(fc$store), c("store_a", "store_b"))
  for (k in unique(fc$store)) {
    dates <- fc$date[fc$store == k]
    expect_identical(dates, sort(dates))
    expect_identical(length(dates), 14L)
    expect_identical(min(dates), max(panel$date) + 1L)
  }
  expect_true(all(is.finite(fc$.mean)))

  df <- as.data.frame(fc)
  expect_true(all(c("store", "date", ".mean") %in% names(df)))
  q <- attr(fc, "quantiles")
  expect_identical(dim(q), c(28L, 3L))
  expect_true(all(apply(q, 1L, function(row) all(diff(row) >= 0))))
})

test_that("journey 2: the batched fable route", {
  record <- skip_unless_checkpoint()
  skip_if_not_installed("fabletools")
  skip_if_not_installed("tsibble")
  model <- zuk_pretrained(record$model_id, revision = record$revision,
                           device = "cpu")

  fc <- forecast(model, smoke_panel(), h = 7L, index = "date", key = "store",
                 target = "sales")
  fbl <- fabletools::as_fable(fc)

  expect_s3_class(fbl, "fbl_ts")
  expect_identical(nrow(fbl), 14L)
  expect_true(".mean" %in% names(fbl))
  expect_true(all(is.finite(fbl$.mean)))
  # The point forecast is the engine's exact median, and the distribution
  # travelling beside it agrees.
  expect_equal(as.numeric(fbl$.mean), as.numeric(fc$.mean))
  expect_equal(as.numeric(stats::median(fbl$sales)), as.numeric(fc$.mean))
})

test_that("journey 3: TSFM() composed inside fabletools::model()", {
  record <- skip_unless_checkpoint()
  skip_if_not_installed("fabletools")
  skip_if_not_installed("tsibble")
  old <- options(zuk.max_loaded_models = 1L)
  on.exit(options(old), add = TRUE)

  panel <- tsibble::as_tsibble(smoke_panel(), key = "store", index = "date")
  fits <- fabletools::model(
    panel,
    timesfm = TSFM(sales, model_id = record$model_id,
                   revision = record$revision, device = "cpu")
  )
  expect_identical(nrow(fits), 2L)

  # Both keys must be served by one constructed handle: reloading a 925 MB
  # checkpoint per series is the failure this guards against.
  handles <- lapply(fits$timesfm, function(m) m$fit$model)
  expect_true(identical(handles[[1]], handles[[2]]))

  fc <- fabletools::forecast(fits, h = 7L)
  expect_s3_class(fc, "fbl_ts")
  expect_identical(nrow(fc), 14L)
  expect_true(all(is.finite(fc$.mean)))
})

test_that("journey 4: parsnip fit() and predict()", {
  record <- skip_unless_checkpoint()
  skip_if_not_installed("parsnip")
  skip_if_not_installed("hardhat")

  panel <- smoke_panel(keys = "store_a")
  spec <- parsnip::set_engine(
    zuk_reg(),
    "zukzeit",
    model_id = record$model_id,
    revision = record$revision,
    device = "cpu",
    index = "date"
  )
  fitted <- parsnip::fit(spec, sales ~ 1, data = panel)
  expect_s3_class(fitted, "model_fit")

  future <- data.frame(date = max(panel$date) + 1:10)
  preds <- stats::predict(fitted, new_data = future)
  expect_identical(nrow(preds), 10L)
  expect_true(all(is.finite(preds$.pred)))
})

test_that("forecasts are antisymmetric under sign flip", {
  record <- skip_unless_checkpoint()
  model <- zuk_pretrained(record$model_id, revision = record$revision,
                           device = "cpu")
  # The port symmetrises a forward and a flipped pass (force_flip_invariance),
  # so on a mixed-sign series --- where the positivity clamp applies to neither
  # direction --- f(-x) must be -f(x) with the quantile channels reversed. This
  # needs no reference fixture, and it fails loudly if timesfm_channel_layout()
  # ever reverses the wrong range.
  set.seed(11L)
  mixed <- as.numeric(sin(seq_len(96) / 6) * 10 + stats::rnorm(96))
  levels <- c(0.1, 0.5, 0.9)

  q <- model$predict_fn(mixed, 12L, levels)
  flipped <- model$predict_fn(-mixed, 12L, levels)
  expect_equal(flipped, -q[, rev(seq_along(levels)), drop = FALSE])

  # A series carrying negatives must not be clamped at zero.
  expect_true(any(q < 0) || any(flipped < 0))
})

test_that("accelerator inference agrees with the CPU baseline", {
  record <- skip_unless_checkpoint()
  device <- if (isTRUE(tryCatch(torch::backends_mps_is_available(),
                               error = function(e) FALSE))) {
    "mps"
  } else if (isTRUE(tryCatch(torch::cuda_is_available(),
                             error = function(e) FALSE))) {
    "cuda"
  } else {
    NA_character_
  }
  skip_if(is.na(device), "No accelerator available on this host.")

  set.seed(12L)
  context <- as.numeric(100 + cumsum(stats::rnorm(96)))
  levels <- c(0.1, 0.5, 0.9)

  cpu <- zuk_pretrained(record$model_id, revision = record$revision,
                         device = "cpu")$predict_fn(context, 12L, levels)
  accel <- zuk_pretrained(record$model_id, revision = record$revision,
                           device = device)$predict_fn(context, 12L, levels)

  expect_true(all(is.finite(accel)))
  # Deliberately looser than the CPU fixture budget: cross-device reassociation
  # runs to a few tens of ulps, above PyTorch's float32 rtol default.
  expect_close_f32(accel, cpu, atol = 1e-3, rtol = 1e-5)
})

test_that("the composed and batched routes agree on one series", {
  record <- skip_unless_checkpoint()
  skip_if_not_installed("fabletools")
  skip_if_not_installed("tsibble")
  panel <- smoke_panel(keys = "store_a")
  levels <- c(0.1, 0.5, 0.9)

  model <- zuk_pretrained(record$model_id, revision = record$revision,
                           device = "cpu")
  batched <- forecast(model, panel, h = 7L, index = "date", key = "store",
                      target = "sales", quantile_levels = levels)
  composed <- fabletools::forecast(
    fabletools::model(
      tsibble::as_tsibble(panel, key = "store", index = "date"),
      timesfm = TSFM(sales, model_id = record$model_id,
                     revision = record$revision, device = "cpu",
                     quantile_levels = levels)
    ),
    h = 7L
  )
  # Two public routes, one engine: the predictive distributions must be
  # identical. Compare medians, which the engine evaluates exactly.
  #
  # `.mean` is deliberately not the comparison. fabletools derives it from the
  # distribution, so the composed route reports the distribution's mean, while
  # the plain-R route reports the engine's median. Same distribution, different
  # summary; see ?TSFM.
  expect_equal(
    as.numeric(stats::median(composed$sales)),
    as.numeric(batched$.mean)
  )
  expect_false(isTRUE(all.equal(
    as.numeric(composed$.mean), as.numeric(batched$.mean)
  )))
})
