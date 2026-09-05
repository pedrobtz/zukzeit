# Numerical-parity harness for the native TimesFM port (roadmap Stage 2 exit
# gate). Identical in spirit to test-parity-ttm.R: fixed inputs vs. the
# reference implementation's outputs, committed as data so CI needs neither
# Python nor the Hub. Skips until the numerical port lands and fixtures exist.

test_that("native TimesFM matches the reference on golden fixtures", {
  skip_unless_checkpoint("timesfm")
  skip_if_not_installed("torch")
  skip_if_not_installed("jsonlite")
  dir <- testthat::test_path("fixtures", "timesfm")
  skip_if_not(dir.exists(dir) && length(list.files(dir, pattern = "\\.json$")) > 0,
              "No TimesFM golden fixtures yet; generate them per fixtures/README.md.")
  record <- zuk_catalogue_get("google/timesfm-2.5-200m-pytorch")
  skip_if_not(
    identical(record$state, "supported"),
    "TimesFM fixtures exist, but native inference has not passed the support gates."
  )

  # The fixtures are CPU reference outputs, so pin the handle to CPU rather than
  # inheriting the host's device resolution: on a CUDA or MPS machine the
  # default "auto" loads elsewhere and this gate cannot run.
  model <- zuk_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    revision = record$revision,
    device = "cpu"
  )

  # Thread count is deliberately not pinned here. LibTorch's native parallel
  # backend refuses to change intraop threads once parallel work has started or
  # after one set_num_threads call, so setting it mid-suite is a one-way door
  # that warns on restore. Export OMP_NUM_THREADS=1 before starting R if you want
  # single-threaded reduction. It is not needed for correctness: repeated
  # inference is asserted bit-identical below under default threading, and
  # cross-build variation is what the atol/rtol budget covers.

  files <- list.files(dir, pattern = "\\.json$", full.names = TRUE)
  for (f in files) {
    fx <- jsonlite::fromJSON(f, simplifyVector = FALSE)
    contexts <- lapply(fx$context_files, function(file) {
      path <- file.path(dir, file)
      readBin(
        path, "numeric", n = unname(file.info(path)$size) %/% 4L,
        size = 4L, endian = "little"
      )
    })
    levels <- as.numeric(fx$quantile_levels)
    h <- as.integer(fx$horizon)
    actual <- model$predict_batch_fn(
      contexts, rep(h, length(contexts)), levels, device = "cpu"
    )
    expected <- array(
      unlist(fx$expected_quantiles),
      dim = c(length(levels), h, length(contexts))
    )
    expected <- aperm(expected, c(3, 2, 1))
    for (i in seq_along(actual)) {
      expect_close_f32(
        unname(actual[[i]]), unname(expected[i, , ]),
        atol = fx$atol, rtol = fx$rtol
      )
    }
  }
})

test_that("real TimesFM is conforming, deterministic, and silent", {
  skip_unless_checkpoint("timesfm")
  model <- zuk_pretrained(
    "google/timesfm-2.5-200m-pytorch", device = "cpu"
  )
  context <- readBin(
    testthat::test_path("fixtures", "timesfm", "typical-context-1.f32"),
    "numeric", n = 96L, size = 4L, endian = "little"
  )
  first <- expect_silent(model$predict_fn(context, 6L, c(0.1, 0.5, 0.9)))
  second <- expect_silent(model$predict_fn(context, 6L, c(0.1, 0.5, 0.9)))
  expect_identical(first, second)
  multi_block <- expect_silent(
    model$predict_fn(context, 129L, c(0.1, 0.5, 0.9))
  )
  expect_identical(dim(multi_block), c(129L, 3L))
  expect_true(all(is.finite(multi_block)))
  expect_true(all(apply(multi_block, 1L, function(row) all(diff(row) >= 0))))
  device_error <- expect_error(
    model$predict_batch_fn(list(context), 2L, c(0.1, 0.5, 0.9), "cuda"),
    class = "zuk_error_device"
  )
  expect_identical(device_error$resolved_device, "cpu")

  # Every trained level at once, spelled the way the catalogue advertises them.
  # seq() and the config's literals differ at 0.3 and 0.7, so an exact match
  # here would silently produce NA columns.
  advertised <- zuk_models()$quantile_levels[[1]]
  all_levels <- expect_silent(model$predict_fn(context, 6L, advertised))
  expect_identical(dim(all_levels), c(6L, 9L))
  expect_true(all(is.finite(all_levels)))
  expect_true(all(apply(all_levels, 1L, function(row) all(diff(row) >= 0))))
  # The median column must agree with the three-level request that shares it.
  expect_equal(all_levels[, 5L], first[, 2L])

  unsupported <- expect_error(
    model$predict_fn(context, 6L, c(0.05, 0.5)),
    class = "zuk_error_quantile_levels"
  )
  expect_equal(unsupported$supported, seq(0.1, 0.9, by = 0.1))

  # A forecast may not depend on which series share its batch. Asserted within
  # the float32 budget, not bit-exactly: the batch dimension changes the shapes
  # BLAS reduces over, which moves results by a couple of ulps on some
  # architectures. The regression this guards against --- truncating every series
  # to the batch's longest horizon --- would drop hundreds of observations of
  # context and move the forecast by orders of magnitude more than that.
  solo <- model$predict_batch_fn(list(context), 6L, c(0.1, 0.5, 0.9), "cpu")
  mixed <- model$predict_batch_fn(
    list(context, context), c(6L, 512L), c(0.1, 0.5, 0.9), "cpu"
  )
  expect_close_f32(mixed[[1]], solo[[1]])
  report <- zuk_check_architecture(
    function(config, weights) model,
    context = context,
    tolerance = 1e-4,
    error = FALSE
  )
  expect_true(all(report$ok, na.rm = TRUE))
})
