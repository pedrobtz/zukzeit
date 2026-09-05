# Grouped inputs, contract 1.1. The engine half: what an architecture receives,
# what it must declare first, and that contract-1.0 architectures are untouched.

grouped_caps <- function(...) {
  new_zuk_capabilities("grouped", max_context = 512L, ...)
}

grouped_model <- function(caps = grouped_caps(multivariate = TRUE,
                                              past_covariates = TRUE,
                                              future_covariates = TRUE),
                          record = NULL) {
  new_zuk_model(
    architecture = "grouped", config = list(), capabilities = caps,
    predict_fn = function(context, h, quantile_levels) {
      matrix(context[length(context)], h, length(quantile_levels))
    },
    predict_batch_fn = function(contexts, horizons, quantile_levels, device, groups) {
      if (!is.null(record)) record(groups)
      lapply(which(groups$target), function(i) {
        matrix(contexts[[i]][length(contexts[[i]])], horizons[[i]],
               length(quantile_levels))
      })
    },
    contract_version = "1.1.0"
  )
}

three_series <- function() list(a = as.numeric(1:20), b = as.numeric(101:120),
                                c = as.numeric(201:220))
levels3 <- c(0.1, 0.5, 0.9)

test_that("a grouped request returns one matrix per target row", {
  seen <- NULL
  model <- grouped_model(record = function(g) seen <<- g)
  groups <- list(id = c(1, 1, 1), target = c(TRUE, FALSE, FALSE),
                 future = list(NULL, NULL, rep(9, 6)))

  out <- zuk_run_batches(model, three_series(), rep(6L, 3), levels3, groups = groups)

  expect_length(out, 1L)
  expect_identical(dim(out[[1]]), c(6L, 3L))
  # The architecture sees every row, including the covariates it conditions on.
  expect_length(seen$id, 3L)
  expect_identical(seen$target, c(TRUE, FALSE, FALSE))
  expect_equal(seen$future[[3]], rep(9, 6))
})

test_that("contract 1.0 architectures are untouched", {
  model <- zuk_pretrained("stub")
  on.exit(zuk_unload("stub"), add = TRUE)

  # The default is 1.0.0, not the engine's version: the stamp is a claim about
  # the architecture, and over-claiming would hand it an argument it cannot take.
  expect_identical(model$contract_version, package_version("1.0.0"))
  expect_false(zuk_supports_groups(model))

  out <- zuk_run_batches(model, three_series(), rep(6L, 3), levels3)
  expect_length(out, 3L)

  expect_error(
    zuk_run_batches(model, three_series(), rep(6L, 3), levels3,
                    groups = list(id = c(1, 1, 1), target = c(TRUE, FALSE, FALSE),
                                  future = vector("list", 3))),
    class = "zuk_error_contract"
  )
})

test_that("grouped capabilities require a contract 1.1 architecture", {
  expect_error(
    new_zuk_model("d", list(), grouped_caps(multivariate = TRUE),
                  function(context, h, quantile_levels) matrix(0, h, 1L)),
    class = "zuk_error_contract"
  )
  expect_s3_class(
    new_zuk_model("d", list(), grouped_caps(multivariate = TRUE),
                  function(context, h, quantile_levels) matrix(0, h, 1L),
                  contract_version = "1.1.0"),
    "zuk_model"
  )
})

test_that("channels with no execution path stay refused at every version", {
  expect_error(grouped_caps(samples = TRUE), class = "zuk_error_contract")
  expect_error(grouped_caps(static_covariates = TRUE), class = "zuk_error_contract")
  expect_error(grouped_caps(fine_tunable = TRUE), class = "zuk_error_contract")
})

test_that("the batch boundary refuses undeclared and malformed tasks", {
  refuse <- function(caps, groups, horizons = rep(6L, 3)) {
    expect_error(
      zuk_run_batches(grouped_model(caps), three_series(), horizons, levels3,
                      groups = groups),
      class = "zuk_error_capability"
    )
  }
  full <- grouped_caps(multivariate = TRUE, past_covariates = TRUE,
                       future_covariates = TRUE)
  univariate <- grouped_caps(past_covariates = TRUE)

  refuse(univariate, list(id = c(1, 1, 1), target = c(TRUE, TRUE, FALSE),
                          future = vector("list", 3)))
  refuse(univariate, list(id = c(1, 1, 1), target = c(TRUE, FALSE, FALSE),
                          future = list(NULL, NULL, rep(9, 6))))
  refuse(full, list(id = c(1, 2, 2), target = c(TRUE, FALSE, FALSE),
                    future = vector("list", 3)))
  refuse(full, list(id = c(1, 1, 1), target = c(TRUE, FALSE, FALSE),
                    future = list(NULL, NULL, rep(9, 3))))
  refuse(full, list(id = c(1, 1, 1), target = c(TRUE, FALSE, FALSE),
                    future = list(rep(9, 6), NULL, NULL)))
  refuse(full, list(id = c(1, 1, 1), target = c(TRUE, FALSE, FALSE),
                    future = vector("list", 3)),
         horizons = c(6L, 4L, 6L))
  refuse(full, list(id = c(1, 1), target = c(TRUE, FALSE), future = vector("list", 2)))
})

test_that("a task is never split across batches", {
  tasks <- NULL
  contexts <- stats::setNames(
    lapply(1:6, function(i) as.numeric(seq_len(20) + i * 10)), letters[1:6]
  )
  model <- grouped_model(record = function(g) tasks <<- c(tasks, length(unique(g$id))))
  groups <- list(id = c(1, 1, 1, 2, 2, 2),
                 target = c(TRUE, FALSE, FALSE, TRUE, FALSE, FALSE),
                 future = vector("list", 6))

  out <- zuk_run_batches(model, contexts, rep(6L, 6), levels3,
                         batch_size = 4L, groups = groups)

  # Two three-row tasks with a budget of four rows: one task per call, never a
  # split one, because a task's rows are attended together.
  expect_identical(tasks, c(1L, 1L))
  expect_length(out, 2L)
})

test_that("covariate rows need a batch forward pass", {
  loop_only <- new_zuk_model(
    "grouped", list(), grouped_caps(past_covariates = TRUE),
    predict_fn = function(context, h, quantile_levels) {
      matrix(0, h, length(quantile_levels))
    },
    contract_version = "1.1.0"
  )
  expect_error(
    zuk_run_batches(loop_only, three_series(), rep(6L, 3), levels3,
                    groups = list(id = c(1, 1, 1), target = c(TRUE, FALSE, FALSE),
                                  future = vector("list", 3))),
    class = "zuk_error_contract"
  )
})
