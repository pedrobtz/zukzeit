capture_zuk_error <- function(expr) {
  tryCatch(force(expr), zuk_error = identity)
}

expect_one_policy_parent <- function(error) {
  policies <- c(
    "zuk_error_recoverable",
    "zuk_error_external",
    "zuk_error_internal"
  )
  expect_identical(
    sum(vapply(policies, function(policy) inherits(error, policy), logical(1))),
    1L
  )
  expect_s3_class(error, "zuk_error")
}

test_that("every recoverable error family has its leaf, parent, and fields", {
  capability <- capture_zuk_error(zuk_models("invalid"))
  context <- capture_zuk_error(
    check_context_length(new_zuk_capabilities("x", 4L), 5L, "id", "rev")
  )
  quantile <- capture_zuk_error(
    check_quantile_levels(
      new_zuk_capabilities("x", 4L, quantile_levels = c(0.1, 0.5)),
      0.9,
      "id",
      "rev"
    )
  )
  device <- capture_zuk_error(zuk_resolve_device("tpu"))

  expect_s3_class(capability, "zuk_error_capability")
  expect_identical(capability$capability, "catalogue_state")
  expect_s3_class(context, "zuk_error_context_length")
  expect_identical(context$model_id, "id")
  expect_identical(context$requested, 5L)
  expect_identical(context$supported, 4L)
  expect_s3_class(quantile, "zuk_error_quantile_levels")
  expect_equal(quantile$requested, 0.9)
  expect_equal(quantile$supported, c(0.1, 0.5))
  expect_s3_class(device, "zuk_error_device")
  expect_identical(device$requested_device, "tpu")

  lapply(list(capability, context, quantile, device), expect_one_policy_parent)
})

test_that("external and internal families retain programmatic fields", {
  parent <- simpleError("offline")
  download <- capture_zuk_error(zuk_abort_download(
    "download failed",
    model_id = "org/model",
    revision = "abc",
    file = "config.json",
    parent = parent
  ))
  checkpoint <- capture_zuk_error(zuk_abort_checkpoint(
    "bad tensor",
    model_id = "org/model",
    revision = "abc",
    tensor = "head.weight",
    expected = c(3L, 2L),
    actual = c(2L, 3L)
  ))
  contract <- capture_zuk_error(zuk_abort_contract(
    "wrong shape",
    architecture = "test",
    model_id = "org/model",
    contract = "predict return shape",
    expected = c(3L, 2L),
    actual = c(2L, 3L)
  ))

  expect_s3_class(download, "zuk_error_download")
  expect_s3_class(download, "zuk_error_external")
  expect_identical(download$file, "config.json")
  expect_identical(download$parent, parent)
  expect_s3_class(checkpoint, "zuk_error_checkpoint")
  expect_s3_class(checkpoint, "zuk_error_internal")
  expect_identical(checkpoint$tensor, "head.weight")
  expect_s3_class(contract, "zuk_error_contract")
  expect_s3_class(contract, "zuk_error_internal")
  expect_identical(contract$contract, "predict return shape")

  lapply(list(download, checkpoint, contract), expect_one_policy_parent)
})
