test_that("device strings are validated", {
  expect_true(is_valid_device("auto"))
  expect_true(is_valid_device("cpu"))
  expect_true(is_valid_device("mps"))
  expect_true(is_valid_device("cuda"))
  expect_true(is_valid_device("cuda:1"))
  expect_false(is_valid_device("gpu"))
  expect_false(is_valid_device("cuda:x"))
  expect_false(is_valid_device(c("cpu", "cuda")))
  expect_error(zuk_resolve_device("gpu"), "Invalid")
})

test_that("resolve falls back to cpu without torch", {
  skip_if(requireNamespace("torch", quietly = TRUE),
          "torch installed; resolution is backend-dependent")
  expect_identical(zuk_resolve_device("auto"), "cpu")
  expect_identical(zuk_resolve_device("cpu"), "cpu")
  # explicit accelerator with no torch backend is passed through unchanged
  expect_identical(zuk_resolve_device("cuda"), "cuda")
})

test_that("zuk_set_device round-trips the option", {
  old <- getOption("zuk.device", "auto")
  on.exit(options(zuk.device = old), add = TRUE)

  prev <- zuk_set_device("cpu")
  expect_identical(getOption("zuk.device"), "cpu")
  expect_identical(prev, old)
  expect_error(zuk_set_device("tpu"), "Invalid")
})

test_that("zuk_to_device is a no-op without torch", {
  skip_if(requireNamespace("torch", quietly = TRUE), "torch installed")
  x <- 1:5
  expect_identical(zuk_to_device(x, "cpu"), x)
})
