test_that("zuk_reg registers with parsnip and builds a spec", {
  skip_if_not_installed("parsnip")
  make_zuk_reg()  # idempotent

  expect_true("zuk_reg" %in% parsnip::get_from_env("models"))

  spec <- zuk_reg(context_length = 512)
  expect_s3_class(spec, "model_spec")
  expect_identical(spec$mode, "regression")
})

test_that("model identity is supplied through set_engine", {
  skip_if_not_installed("parsnip")
  make_zuk_reg()

  spec_stub <- parsnip::set_engine(
    zuk_reg(), "zukzeit",
    model_id = "stub", index = "date", id = "store"
  )
  spec_other <- parsnip::set_engine(
    zuk_reg(), "zukzeit",
    model_id = "stub-alternate", index = "date", id = "store"
  )

  expect_identical(spec_stub$engine, "zukzeit")
  expect_identical(
    rlang::eval_tidy(spec_stub$eng_args$model_id),
    "stub"
  )
  expect_identical(
    rlang::eval_tidy(spec_other$eng_args$model_id),
    "stub-alternate"
  )
  expect_identical(names(spec_stub$eng_args), names(spec_other$eng_args))
})

test_that("parsnip fits and predicts through the exported stub bridge", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("hardhat")
  skip_if_not_installed("distributional")
  make_zuk_reg()

  expect_true("zuk_parsnip_fit" %in% getNamespaceExports("zukzeit"))

  train <- data.frame(store = "a", date = 1:20, value = as.numeric(1:20))
  future <- data.frame(store = "a", date = 21:23)
  spec <- parsnip::set_engine(
    zuk_reg(context_length = 8L),
    "zukzeit",
    model_id = "stub",
    index = "date",
    id = "store"
  )

  fit <- parsnip::fit(spec, value ~ 1, data = train)
  expect_s3_class(fit$fit, "zuk_fit")
  expect_identical(fit$fit$model$capabilities$max_context, 8L)

  pred <- predict(fit, new_data = future)
  expect_identical(names(pred), ".pred")
  expect_equal(pred$.pred, rep(20, 3))
})

test_that("context_length is exposed as a tunable dials parameter", {
  skip_if_not_installed("dials")

  param <- context_length()
  expect_s3_class(param, "quant_param")

  # Exercise the tunable metadata directly (the generic's home package varies
  # across tidymodels versions; dispatch is wired in make_zuk_reg()).
  tun <- tunable_zuk_reg(zuk_reg())
  expect_true("context_length" %in% tun$name)
  expect_identical(tun$call_info[[1]]$fun, "context_length")
})

# --- regression: the spec must be updatable, or tuning cannot finalize -------

test_that("the spec provides an update() method", {
  skip_if_not_installed("parsnip")

  spec <- zuk_reg(context_length = 512L)
  updated <- update(spec, context_length = 128L)

  expect_s3_class(updated, "zuk_reg")
  expect_identical(rlang::eval_tidy(updated$args$context_length), 128L)
  # The original is untouched.
  expect_identical(rlang::eval_tidy(spec$args$context_length), 512L)

  # `fresh = TRUE` replaces the argument set rather than modifying it.
  expect_s3_class(update(spec, context_length = 64L, fresh = TRUE), "zuk_reg")
})

test_that("a tuned context_length can be finalized back into a workflow", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("workflows")
  skip_if_not_installed("tune")
  skip_if_not_installed("distributional")
  make_zuk_reg()

  train <- data.frame(store = "a", day = 1:40, sales = as.numeric(1:40))
  spec <- parsnip::set_engine(
    zuk_reg(context_length = tune::tune()),
    "zukzeit", model_id = "stub", index = "day", id = "store"
  )
  wf <- workflows::add_model(
    workflows::add_variables(
      workflows::workflow(),
      outcomes = sales, predictors = c(day, store)
    ),
    spec
  )

  # Without update.zuk_reg() this failed inside update.default() with
  # "need an object with call component", so no tuning result could ever be
  # refit -- the whole dials/tune integration was unreachable.
  final <- tune::finalize_workflow(wf, data.frame(context_length = 16L))
  fitted <- parsnip::fit(final, data = train)

  engine_fit <- workflows::extract_fit_parsnip(fitted)$fit
  expect_s3_class(engine_fit, "zuk_fit")
  expect_identical(engine_fit$model$capabilities$max_context, 16L)
  expect_identical(
    nrow(predict(fitted, new_data = data.frame(store = "a", day = 41:43))),
    3L
  )
  zuk_unload("stub")
})

test_that("a workflow must pass the index and id columns to the engine", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("workflows")
  skip_if_not_installed("distributional")
  make_zuk_reg()

  train <- data.frame(store = "a", day = 1:40, sales = as.numeric(1:40))
  spec <- parsnip::set_engine(
    zuk_reg(context_length = 16L),
    "zukzeit", model_id = "stub", index = "day", id = "store"
  )

  # add_variables() passes the named columns through untouched.
  by_variables <- workflows::add_model(
    workflows::add_variables(
      workflows::workflow(),
      outcomes = sales, predictors = c(day, store)
    ),
    spec
  )
  expect_s3_class(parsnip::fit(by_variables, data = train), "workflow")

  # add_formula() keeps only the formula's terms, so the series id never
  # reaches the engine and the fit fails on a missing column.
  by_formula <- workflows::add_model(
    workflows::add_formula(workflows::workflow(), sales ~ day),
    spec
  )
  expect_error(
    parsnip::fit(by_formula, data = train),
    class = "zuk_error_capability"
  )
  zuk_unload("stub")
})
