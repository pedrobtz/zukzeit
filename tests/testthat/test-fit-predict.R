# Stage 0 exit gate (1/2): fit -> predict -> yardstick::rmse().

test_that("fit -> predict -> yardstick::rmse works on the stub", {
  skip_if_not_installed("distributional")
  skip_if_not_installed("yardstick")

  set.seed(1)
  train <- data.frame(store = "a", t = 1:40, y = cumsum(rnorm(40)) + 100)
  future <- data.frame(store = "a", t = 41:45, y = cumsum(rnorm(5)) + 100)

  model <- zuk_pretrained("stub")
  fit <- zuk_fit(y ~ 1, data = train, model = model, index = "t", id = "store")
  expect_s3_class(fit, "zuk_fit")

  preds <- predict(fit, new_data = future)
  expect_identical(nrow(preds), 5L)
  expect_true(all(c(".pred", ".pred_lower", ".pred_upper") %in% names(preds)))
  expect_true(all(preds$.pred_lower <= preds$.pred))
  expect_true(all(preds$.pred <= preds$.pred_upper))

  score <- yardstick::rmse_vec(truth = future$y, estimate = preds$.pred)
  expect_true(is.finite(score))
})

test_that("predictions are row-aligned to new_data across multiple series", {
  skip_if_not_installed("distributional")

  set.seed(2)
  train <- rbind(
    data.frame(store = "a", t = 1:20, y = cumsum(rnorm(20)) + 100),
    data.frame(store = "b", t = 1:20, y = cumsum(rnorm(20)) + 50)
  )
  # Interleaved series order, to prove the scatter-back is correct.
  future <- data.frame(
    store = c("b", "a", "b", "a"),
    t     = c(21L, 21L, 22L, 22L)
  )

  model <- zuk_pretrained("stub")
  fit <- zuk_fit(y ~ 1, data = train, model = model, index = "t", id = "store")
  preds <- predict(fit, new_data = future)

  expect_identical(nrow(preds), nrow(future))
  # Series "a" starts near 100+, "b" near 50 — the point forecast is the last
  # observed value, so row order must follow `future`, not sorted order.
  a_last <- tail(train$y[train$store == "a"], 1)
  b_last <- tail(train$y[train$store == "b"], 1)
  expect_equal(preds$.pred[future$store == "a"], rep(a_last, 2))
  expect_equal(preds$.pred[future$store == "b"], rep(b_last, 2))
})

test_that("unknown series in new_data are rejected", {
  skip_if_not_installed("distributional")

  train <- data.frame(store = "a", t = 1:10, y = as.numeric(1:10))
  model <- zuk_pretrained("stub")
  fit <- zuk_fit(y ~ 1, data = train, model = model, index = "t", id = "store")
  future <- data.frame(store = "z", t = 11:12)
  expect_error(predict(fit, new_data = future), "No fitted history")
})

# --- regression: series keys are labels, not factor level codes --------------

test_that("a factor series id selects history by label, not by level code", {
  skip_if_not_installed("distributional")

  train <- rbind(
    data.frame(store = "a", t = 1:30, y = rep(1, 30)),
    data.frame(store = "b", t = 1:30, y = rep(50, 30)),
    data.frame(store = "c", t = 1:30, y = rep(900, 30))
  )
  model <- zuk_pretrained("stub")
  on.exit(zuk_unload("stub"), add = TRUE)
  fit <- zuk_fit(y ~ ., data = train, model = model, index = "t", id = "store")

  # A single-level factor -- what droplevels() or subset() leaves behind --
  # previously resolved to level code 1, i.e. series "a".
  dropped <- data.frame(store = factor("c"), t = 31L)
  expect_equal(predict(fit, dropped)$.pred, 900)

  # Level order must not change which series is forecast, or how rows align.
  reordered <- data.frame(
    store = factor(c("c", "a"), levels = c("c", "a", "b")),
    t     = c(31L, 31L)
  )
  expect_equal(predict(fit, reordered)$.pred, c(900, 1))

  # Character ids keep working unchanged.
  expect_equal(
    predict(fit, data.frame(store = c("b", "a"), t = 31L))$.pred,
    c(50, 1)
  )
})

test_that("a factor id at fit time is keyed the same way as at predict time", {
  skip_if_not_installed("distributional")

  train <- data.frame(
    store = factor(rep(c("b", "a"), each = 30), levels = c("b", "a")),
    t     = rep(1:30, 2),
    y     = c(rep(100, 30), rep(7, 30))
  )
  model <- zuk_pretrained("stub")
  on.exit(zuk_unload("stub"), add = TRUE)
  fit <- zuk_fit(y ~ ., data = train, model = model, index = "t", id = "store")

  expect_identical(names(fit$histories), c("a", "b"))
  expect_equal(
    predict(fit, data.frame(store = c("a", "b"), t = 31L))$.pred,
    c(7, 100)
  )
})

# --- regression: one prediction column per requested quantile level ----------

test_that("every requested quantile level gets its own prediction column", {
  skip_if_not_installed("distributional")

  train <- data.frame(store = "a", t = 1:40, y = as.numeric(1:40))
  model <- zuk_pretrained("stub")
  on.exit(zuk_unload("stub"), add = TRUE)

  # Whole percents keep the familiar two-digit names.
  whole <- zuk_fit(y ~ ., data = train, model = model, index = "t", id = "store",
                    quantile_levels = c(0.1, 0.5, 0.9))
  whole_cols <- names(predict(whole, data.frame(store = "a", t = 41L)))
  expect_identical(
    whole_cols,
    c(".pred", ".pred_lower", ".pred_upper", ".pred_q10", ".pred_q50", ".pred_q90")
  )

  # Sub-percent levels used to collide: 0.02 and 0.025 both rounded to
  # `.pred_q02`, so one silently overwrote the other.
  levels <- c(0.02, 0.025, 0.5, 0.975, 0.98)
  fine <- zuk_fit(y ~ ., data = train, model = model, index = "t", id = "store",
                   quantile_levels = levels)
  fine_pred <- predict(fine, data.frame(store = "a", t = 41L))
  quantile_cols <- grep("^\\.pred_q", names(fine_pred), value = TRUE)

  expect_length(quantile_cols, length(levels))
  expect_identical(anyDuplicated(quantile_cols), 0L)
  expect_identical(
    quantile_cols,
    c(".pred_q02_0", ".pred_q02_5", ".pred_q50_0", ".pred_q97_5", ".pred_q98_0")
  )
  # The columns are still ordered by level, and still non-decreasing.
  expect_true(all(diff(unlist(fine_pred[1, quantile_cols])) >= 0))
})
