# The caller-facing covariate surface: plain R and the tidyverts route.

grouped_capabilities <- function(architecture = "stub", ...) {
  new_zuk_capabilities(architecture, max_context = 512L, ...)
}

# Records what the engine hands the architecture, so a test can assert on the
# request rather than only on the shape of the answer.
recording_arch <- function(record, caps = grouped_capabilities(
    past_covariates = TRUE, future_covariates = TRUE)) {
  function(config, weights) {
    new_zuk_model(
      "stub", config, caps,
      predict_fn = function(context, h, quantile_levels) {
        matrix(context[length(context)], h, length(quantile_levels))
      },
      predict_batch_fn = function(contexts, horizons, quantile_levels, device, groups) {
        record(list(groups = groups, contexts = contexts))
        lapply(which(groups$target), function(i) {
          matrix(contexts[[i]][length(contexts[[i]])], horizons[[i]],
                 length(quantile_levels))
        })
      },
      contract_version = "1.1.0"
    )
  }
}

local_grouped_stub <- function(record, ..., env = parent.frame()) {
  zuk_register_arch("stub", recording_arch(record, ...), overwrite = TRUE)
  withr::defer(zuk_register_arch("stub", stub_constructor, overwrite = TRUE), envir = env)
}

sales_panel <- function() {
  data.frame(
    store   = rep(c("a", "b"), each = 30),
    region  = "north",
    day     = rep(1:30, 2),
    sales   = c(cumsum(rep(2, 30)) + 100, cumsum(rep(1, 30)) + 50),
    promo   = rep(c(0, 1), 30),
    traffic = as.numeric(rep(seq_len(30), 2))
  )
}

test_that("covariates must be named, not inferred from leftover columns", {
  # panel_spec() already derives an unspecified target as the first measured
  # variable, so treating extra columns as covariates would silently change
  # which column is forecast. Extra columns are therefore ignored by default.
  seen <- NULL
  local_grouped_stub(function(x) seen <<- x)
  model <- zuk_pretrained("stub")
  on.exit(zuk_unload("stub"), add = TRUE)

  forecast(model, sales_panel(), h = 3, index = "day", key = "store",
           target = "sales")
  expect_true(all(seen$groups$target))
  expect_length(seen$contexts, 2L)
})

test_that("a covariate is future-known when future values are supplied", {
  seen <- NULL
  local_grouped_stub(function(x) seen <<- x)
  model <- zuk_pretrained("stub")
  on.exit(zuk_unload("stub"), add = TRUE)

  future <- data.frame(store = rep(c("a", "b"), each = 3), day = rep(31:33, 2),
                       promo = c(1, 0, 1, 0, 1, 0))
  fc <- forecast(model, sales_panel(), h = 3, index = "day", key = "store",
                 target = "sales", covariates = c("promo", "traffic"),
                 future = future)

  # Each task is one target followed by its two covariate rows.
  expect_identical(seen$groups$target, c(TRUE, FALSE, FALSE, TRUE, FALSE, FALSE))
  expect_identical(length(unique(seen$groups$id)), 2L)
  # promo appears in `future`, traffic does not, so their kinds are inferred.
  expect_equal(seen$groups$future[[2]], c(1, 0, 1))
  expect_null(seen$groups$future[[3]])
  expect_identical(nrow(fc), 6L)
})

test_that("cross-series learning is opt-in", {
  tasks <- NULL
  # Several series in one task means several targets in it, which is what the
  # multivariate flag governs -- cross-learning across items and multivariate
  # forecasting are the same channel.
  local_grouped_stub(
    function(x) tasks <<- c(tasks, length(unique(x$groups$id))),
    caps = grouped_capabilities(multivariate = TRUE, past_covariates = TRUE,
                                future_covariates = TRUE)
  )
  model <- zuk_pretrained("stub")
  on.exit(zuk_unload("stub"), add = TRUE)
  panel <- sales_panel()

  forecast(model, panel, h = 3, index = "day", key = "store", target = "sales")
  expect_identical(tasks, 2L)

  tasks <- NULL
  forecast(model, panel, h = 3, index = "day", key = "store", target = "sales",
           group = "region")
  expect_identical(tasks, 1L)
})

test_that("grouping series together needs a multivariate checkpoint", {
  local_grouped_stub(function(x) NULL,
                     caps = grouped_capabilities(past_covariates = TRUE))
  model <- zuk_pretrained("stub")
  on.exit(zuk_unload("stub"), add = TRUE)
  expect_error(
    forecast(model, sales_panel(), h = 3, index = "day", key = "store",
             target = "sales", group = "region"),
    class = "zuk_error_capability"
  )
})

test_that("malformed covariate requests are refused", {
  local_grouped_stub(function(x) NULL)
  model <- zuk_pretrained("stub")
  on.exit(zuk_unload("stub"), add = TRUE)
  panel <- sales_panel()
  ask <- function(...) {
    forecast(model, panel, h = 3, index = "day", key = "store",
             target = "sales", ...)
  }
  future <- data.frame(store = rep(c("a", "b"), each = 3), day = rep(31:33, 2),
                       promo = c(1, 0, 1, 0, 1, 0))

  expect_error(ask(covariates = "absent"), class = "zuk_error_capability")
  expect_error(ask(covariates = "sales"), class = "zuk_error_capability")
  expect_error(ask(covariates = "promo", future = future[1:4, ]),
               class = "zuk_error_capability")
  expect_error(ask(covariates = "promo", future = as.list(future)),
               class = "zuk_error_capability")
  expect_error(ask(group = "promo"), class = "zuk_error_capability")
  expect_error(ask(group = "absent"), class = "zuk_error_capability")
})

test_that("a contract 1.0 checkpoint refuses covariates", {
  model <- zuk_pretrained("stub")
  on.exit(zuk_unload("stub"), add = TRUE)
  expect_error(
    forecast(model, sales_panel(), h = 3, index = "day", key = "store",
             target = "sales", covariates = "promo"),
    class = "zuk_error_contract"
  )
})

test_that("TSFM() carries regressors through the tidyverts specials", {
  skip_if_not_installed("fabletools")
  skip_if_not_installed("tsibble")
  seen <- NULL
  local_grouped_stub(function(x) seen <<- x)
  on.exit(zuk_unload(), add = TRUE)

  history <- tsibble::tsibble(
    day = 1:30, sales = as.numeric(cumsum(rep(2, 30))),
    promo = rep(c(0, 1), 15), traffic = as.numeric(1:30), index = day
  )
  ahead <- tsibble::tsibble(day = 31:33, promo = c(9, 8, 7), index = day)

  fits <- fabletools::model(
    history, m = TSFM(sales ~ xreg(promo) + past_xreg(traffic), model_id = "stub")
  )
  fc <- fabletools::forecast(fits, new_data = ahead)

  expect_identical(seen$groups$target, c(TRUE, FALSE, FALSE))
  # fabletools re-evaluates specials against new_data, so xreg() arrives as the
  # future values while past_xreg() correctly has none.
  expect_equal(seen$groups$future[[2]], c(9, 8, 7))
  expect_null(seen$groups$future[[3]])
  expect_identical(nrow(fc), 3L)
})

test_that("regressors are refused by a checkpoint that cannot use them", {
  skip_if_not_installed("fabletools")
  skip_if_not_installed("tsibble")
  history <- tsibble::tsibble(day = 1:30, sales = as.numeric(1:30),
                              promo = rep(c(0, 1), 15), index = day)
  # The built-in stub is contract 1.0 and declares no covariate channel.
  expect_warning(
    fits <- fabletools::model(history, m = TSFM(sales ~ xreg(promo),
                                                model_id = "stub")),
    "future-known"
  )
  zuk_unload("stub")
})
