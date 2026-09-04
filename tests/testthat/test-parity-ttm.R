# Numerical-parity harness for the native TTM port (roadmap Stage 1 exit gate).
#
# The acceptance gate for every native architecture is: same inputs produce
# forecasts within float tolerance of the reference Python implementation.
# Fixtures are committed as data (see tests/testthat/fixtures/README.md) so CI
# needs neither Python nor the Hub. Until the numerical port lands and fixtures
# are generated, this test skips rather than fails --- it is the executable
# contract the port must satisfy.

parity_fixture_dir <- function() {
  testthat::test_path("fixtures", "ttm")
}

read_parity_fixtures <- function(dir) {
  files <- list.files(dir, pattern = "\\.json$", full.names = TRUE)
  lapply(files, function(f) jsonlite::fromJSON(f, simplifyVector = TRUE))
}

test_that("native TTM matches the reference implementation on golden fixtures", {
  skip_if_not_installed("torch")
  skip_if_not_installed("jsonlite")
  dir <- parity_fixture_dir()
  skip_if_not(dir.exists(dir) && length(list.files(dir, pattern = "\\.json$")) > 0,
              "No TTM golden fixtures yet; generate them per fixtures/README.md.")

  model <- zuk_pretrained("ibm-granite/granite-timeseries-ttm-r2")

  for (fx in read_parity_fixtures(dir)) {
    context <- as.numeric(fx$context)
    h <- length(fx$expected_median)
    levels <- as.numeric(fx$quantile_levels)

    q <- model$predict_fn(context, h, levels)
    median_col <- match(0.5, levels)

    expect_close_f32(q[, median_col], as.numeric(fx$expected_median),
                     atol = fx$atol %||% 1e-4, rtol = fx$rtol %||% 1e-5)
    if (!is.null(fx$expected_quantiles)) {
      expected <- matrix(as.numeric(unlist(fx$expected_quantiles)),
                         nrow = h, ncol = length(levels), byrow = TRUE)
      expect_close_f32(unname(q), expected,
                       atol = fx$atol %||% 1e-4, rtol = fx$rtol %||% 1e-5)
    }
  }
})
