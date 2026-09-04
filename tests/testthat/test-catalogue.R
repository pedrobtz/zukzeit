test_that("the safe catalogue default only returns supported checkpoints", {
  local_mocked_bindings(
    zuk_probe_cached_file = function(...) FALSE
  )
  supported <- zuk_models()
  all <- zuk_models(state = NULL)

  expect_identical(nrow(supported), 1L)
  expect_identical(supported$model_id, "google/timesfm-2.5-200m-pytorch")
  expect_identical(supported$max_context, 16256L)
  expect_identical(nrow(all), 3L)
  expect_setequal(unique(all$state), c("supported", "experimental", "scaffold"))
  expect_true(all(grepl("^[0-9a-f]{40}$", all$revision)))
  expect_true(is.list(all$quantile_levels))
  expect_false(any(all$multivariate))
  expect_false(any(all$past_covariates))
  expect_false(any(all$future_covariates))
  expect_setequal(
    names(all),
    c(
      "model_id", "architecture", "revision", "state", "max_context",
      "quantile_levels", "multivariate", "past_covariates",
      "future_covariates", "n_params", "size_bytes", "cached", "license"
    )
  )
})

test_that("manifest cache state distinguishes complete, incomplete, and unknown", {
  record <- zuk_catalogue_records()[[1]]
  expect_true(zuk_manifest_cached(record, function(...) TRUE))

  probe <- function(model_id, revision, file) !identical(file, "model.safetensors")
  expect_false(zuk_manifest_cached(record, probe))

  record$manifest <- NULL
  expect_true(is.na(zuk_manifest_cached(record, function(...) stop("not called"))))
})

test_that("catalogue cache probes are local-only and reflected per manifest", {
  calls <- list()
  local_mocked_bindings(
    zuk_probe_cached_file = function(model_id, revision, file) {
      calls[[length(calls) + 1L]] <<- list(model_id, revision, file)
      !identical(file, "model.safetensors")
    }
  )
  out <- zuk_models(NULL)
  expect_false(any(out$cached))
  # Two manifest files probed for each catalogue record.
  expect_length(calls, 2L * nrow(out))
})

test_that("hfhub cache probes explicitly disable outgoing traffic", {
  seen <- NULL
  local_mocked_bindings(
    hub_download = function(...) {
      seen <<- list(...)
      tempfile()
    },
    .package = "hfhub"
  )
  expect_true(zuk_probe_cached_file("org/model", "abc", "config.json"))
  expect_true(seen$local_files_only)
  expect_identical(seen$repo_id, "org/model")
  expect_identical(seen$revision, "abc")
})

test_that("invalid catalogue states are structured recoverable errors", {
  error <- expect_error(zuk_models("planned"), class = "zuk_error_capability")
  expect_s3_class(error, "zuk_error_recoverable")
  expect_identical(error$capability, "catalogue_state")
})

test_that("scaffold checkpoints fail before download", {
  local_mocked_bindings(
    zuk_resolve_config = function(...) stop("download path was reached")
  )
  error <- expect_error(
    zuk_pretrained("ibm-granite/granite-timeseries-ttm-r2"),
    class = "zuk_error_capability"
  )
  expect_identical(error$capability, "model_state")
  expect_identical(error$requested, "scaffold")
})
