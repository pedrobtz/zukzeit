test_that("batching loops predict_fn, aligned to inputs, when no batch fn", {
  set.seed(1)
  model <- zuk_pretrained("stub")
  contexts <- list(a = cumsum(rnorm(30)) + 100, b = cumsum(rnorm(40)) + 50)
  res <- zuk_run_batches(model, contexts, c(3, 5), c(0.1, 0.5, 0.9))

  expect_length(res, 2)
  expect_equal(dim(res[[1]]), c(3L, 3L))
  expect_equal(dim(res[[2]]), c(5L, 3L))
  expect_equal(res[[1]][, 2], rep(contexts$a[30], 3))  # median == last value
  expect_equal(res[[2]][, 2], rep(contexts$b[40], 5))
})

test_that("batching truncates context to max_context", {
  model <- zuk_pretrained("stub")
  model$capabilities$max_context <- 10L
  res <- zuk_run_batches(model, list(x = 1:100), 2, c(0.1, 0.5, 0.9))
  expect_equal(res[[1]][, 2], rep(100, 2))  # kept the most recent value
})

test_that("a vectorised predict_batch_fn is chunked by batch_size", {
  set.seed(2)
  base <- zuk_pretrained("stub")
  seen <- integer(0)
  batch <- function(contexts, horizons, quantile_levels, device) {
    seen <<- c(seen, length(contexts))
    lapply(seq_along(contexts), function(i) {
      base$predict_fn(contexts[[i]], horizons[[i]], quantile_levels)
    })
  }
  base$predict_batch_fn <- batch

  contexts <- stats::setNames(lapply(1:5, function(i) rnorm(20)), letters[1:5])
  res <- zuk_run_batches(base, contexts, rep(2, 5), c(0.25, 0.75), batch_size = 2)
  expect_length(res, 5)
  expect_equal(seen, c(2L, 2L, 1L))  # 5 series in batches of 2
})

test_that("engine-boundary request validation happens before inference", {
  model <- zuk_pretrained("stub")
  model$predict_fn <- function(...) stop("inference was reached")

  expect_error(
    zuk_run_batches(model, list(1:3), 0, 0.5),
    class = "zuk_error_capability"
  )
  expect_error(
    zuk_run_batches(model, list(1:3), 1.5, 0.5),
    class = "zuk_error_capability"
  )
  expect_error(
    zuk_run_batches(model, list(1:3), 1, 0),
    class = "zuk_error_quantile_levels"
  )
  expect_error(
    zuk_run_batches(model, list(numeric()), 1, 0.5),
    class = "zuk_error_capability"
  )
  expect_error(
    zuk_run_batches(model, list(1:3), 1, 0.5, batch_size = 0),
    class = "zuk_error_capability"
  )
})

test_that("architecture return matrices are validated at the engine boundary", {
  base <- zuk_pretrained("stub")
  base$predict_fn <- function(context, h, quantile_levels) {
    matrix(0, nrow = h + 1L, ncol = length(quantile_levels))
  }
  error <- expect_error(
    zuk_run_batches(base, list(1:3), 2, c(0.1, 0.5)),
    class = "zuk_error_contract"
  )
  expect_identical(error$contract, "predict return shape")
  expect_identical(error$expected, c(2L, 2L))
})

test_that("unsupported trained quantiles fail before a TimesFM forward pass", {
  config <- timesfm_test_config()
  model <- new_zuk_model(
    architecture = "timesfm",
    config = config,
    capabilities = timesfm_capabilities(config),
    predict_fn = function(...) stop("inference was reached")
  )
  error <- expect_error(
    zuk_run_batches(model, list(1:32), 2, c(0.05, 0.5)),
    class = "zuk_error_quantile_levels"
  )
  expect_equal(error$requested, c(0.05, 0.5))
  expect_equal(error$supported, seq(0.1, 0.9, by = 0.1))
})

test_that("architectures receive the checkpoint's own quantile levels", {
  # The catalogue advertises seq(0.1, 0.9, by = 0.1) while config.json parses
  # literals; they differ at 0.3 and 0.7. Reading the levels off the catalogue
  # and forecasting with them is the documented consumer path, so the engine
  # must hand the architecture values it can select channels with by position.
  literal <- jsonlite::fromJSON("[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]")
  seen <- NULL
  model <- new_zuk_model(
    architecture = "recorder",
    config = list(),
    capabilities = new_zuk_capabilities(
      "recorder", max_context = 128L, quantile_levels = literal
    ),
    predict_fn = function(context, h, quantile_levels) {
      seen <<- quantile_levels
      matrix(rep(stats::qnorm(quantile_levels), each = h), nrow = h)
    }
  )

  advertised <- zuk_models(state = NULL)$quantile_levels[[1]]
  out <- zuk_run_batches(model, list(1:32), 2L, advertised)
  expect_identical(seen, literal)
  expect_true(all(is.finite(out[[1]])))
  expect_identical(dim(out[[1]]), c(2L, 9L))
})

test_that("interrupts between batches propagate without a partial result", {
  model <- zuk_pretrained("stub")
  completed <- 0L
  model$predict_batch_fn <- function(contexts, horizons, quantile_levels, device) {
    completed <<- completed + length(contexts)
    lapply(seq_along(contexts), function(i) {
      model$predict_fn(contexts[[i]], horizons[[i]], quantile_levels)
    })
  }
  checks <- 0L
  old <- options(zuk.interrupt_check = function() {
    checks <<- checks + 1L
    if (checks == 2L) {
      stop(structure(
        list(message = "simulated user interrupt", call = NULL),
        class = c("interrupt", "error", "condition")
      ))
    }
  })
  on.exit(options(old), add = TRUE)
  result <- NULL
  expect_condition(
    result <- zuk_run_batches(
      model, rep(list(1:8), 3L), rep(2L, 3L), c(0.1, 0.5, 0.9),
      batch_size = 1L
    ),
    class = "interrupt"
  )
  expect_identical(completed, 1L)
  expect_null(result)
})
