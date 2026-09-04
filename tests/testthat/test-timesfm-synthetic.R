# Structural invariants of the native TimesFM port, checked on a reduced-config
# module with synthetic weights.
#
# Numerical parity against the pinned reference needs the 925 MB checkpoint and
# stays opt-in. These invariants do not: they follow from the *structure* of the
# forward pass rather than from any particular weights, so they need no download
# and run in the ordinary deterministic suite wherever the LibTorch runtime is
# present. On CI that is the one job which installs it, which makes these the
# part of the native port that every push actually exercises.

test_that("the synthetic module builds and maps its weights exactly", {
  skip_if_no_torch()
  config <- timesfm_synthetic_config()
  spec <- timesfm_expected_state_spec(config)
  module <- timesfm_synthetic_module(config)

  expect_identical(length(spec), 34L)
  expect_setequal(names(module$state_dict()), names(spec))
  # Every tensor adopted the shape the checkpoint layout declares.
  for (name in names(spec)) {
    expect_identical(
      as.integer(module$state_dict()[[name]]$shape),
      as.integer(spec[[name]]),
      info = name
    )
  }
})

test_that("forecasts are antisymmetric under a sign flip", {
  skip_if_no_torch()
  config <- timesfm_synthetic_config()
  module <- timesfm_synthetic_module(config)
  levels <- c(0.1, 0.5, 0.9)
  context <- as.numeric(sin(seq_len(40) / 3) * 5)

  q <- timesfm_predict_batch(module, list(context), 4L, levels, config, "cpu")[[1]]
  flipped <- timesfm_predict_batch(module, list(-context), 4L, levels, config, "cpu")[[1]]

  # The decoder averages a forward and a flipped pass, so on a mixed-sign series
  # --- where the positivity clamp applies in neither direction --- this is exact,
  # not approximate. It fails immediately if timesfm_channel_layout() ever
  # reverses the wrong channel range.
  expect_equal(flipped, -q[, rev(seq_along(levels)), drop = FALSE])
})

test_that("the positivity clamp follows the sign of the observed context", {
  skip_if_no_torch()
  config <- timesfm_synthetic_config()
  module <- timesfm_synthetic_module(config)
  levels <- c(0.1, 0.5, 0.9)

  positive <- as.numeric(50 + sin(seq_len(40) / 3) * 5)
  mixed <- as.numeric(sin(seq_len(40) / 3) * 5)

  from_positive <- timesfm_predict_batch(
    module, list(positive), 4L, levels, config, "cpu"
  )[[1]]
  from_mixed <- timesfm_predict_batch(
    module, list(mixed), 4L, levels, config, "cpu"
  )[[1]]

  # An all-positive history is clamped at zero; a series that dips below zero
  # must not be. Every committed parity fixture but one is strictly positive, so
  # this is the cheap standing check on the other branch.
  expect_true(all(from_positive >= 0))
  expect_true(any(from_mixed < 0))
})

test_that("repeated inference is bit-identical and silent", {
  skip_if_no_torch()
  config <- timesfm_synthetic_config()
  module <- timesfm_synthetic_module(config)
  context <- as.numeric(10 + cumsum(sin(seq_len(40))))

  first <- expect_silent(
    timesfm_predict_batch(module, list(context), 6L, c(0.1, 0.5, 0.9), config, "cpu")
  )
  second <- expect_silent(
    timesfm_predict_batch(module, list(context), 6L, c(0.1, 0.5, 0.9), config, "cpu")
  )
  expect_identical(first, second)
})

test_that("the batch path equals the loop path", {
  skip_if_no_torch()
  config <- timesfm_synthetic_config()
  module <- timesfm_synthetic_module(config)
  levels <- c(0.1, 0.5, 0.9)
  contexts <- list(
    as.numeric(10 + sin(seq_len(40) / 2) * 3),
    as.numeric(-4 + cos(seq_len(24) / 5) * 2)
  )

  batched <- timesfm_predict_batch(module, contexts, c(6L, 6L), levels, config, "cpu")
  looped <- lapply(contexts, function(ctx) {
    timesfm_predict_batch(module, list(ctx), 6L, levels, config, "cpu")[[1]]
  })
  # The contract says the batch path is an optimisation, never a different
  # model. Padding differs between the two calls, so this also proves the mask
  # keeps padded positions out of the result. Compared within the float32 budget
  # rather than bit-exactly: the batch dimension alone changes reduction shapes.
  for (i in seq_along(contexts)) {
    expect_close_f32(batched[[i]], looped[[i]])
  }
})

test_that("a series is unaffected by the horizons of its batch neighbours", {
  skip_if_no_torch()
  config <- timesfm_synthetic_config()
  module <- timesfm_synthetic_module(config)
  levels <- c(0.1, 0.5, 0.9)
  # 50 observations fit the h = 4 budget (56) but not the h = 20 budget (40),
  # so a shared truncation would visibly shorten this series.
  long <- as.numeric(20 + sin(seq_len(50) / 4) * 6)
  short <- as.numeric(5 + cos(seq_len(16) / 3))

  alone <- timesfm_predict_batch(module, list(long), 4L, levels, config, "cpu")[[1]]
  together <- timesfm_predict_batch(
    module, list(long, short), c(4L, 20L), levels, config, "cpu"
  )[[1]]
  # Within the float32 budget, not bit-exactly. Sharing the batch's longest
  # horizon would truncate this series from 50 observations to 40 and move the
  # forecast far beyond any rounding difference.
  expect_close_f32(together, alone)
})

test_that("multi-block horizons decode autoregressively and stay well formed", {
  skip_if_no_torch()
  config <- timesfm_synthetic_config()
  module <- timesfm_synthetic_module(config)
  levels <- seq(0.1, 0.9, by = 0.1)
  context <- as.numeric(30 + sin(seq_len(40) / 3) * 4)

  # horizon_length is 8, so 20 spans three output blocks and exercises the
  # cached decode loop rather than a single forward pass.
  q <- timesfm_predict_batch(module, list(context), 20L, levels, config, "cpu")[[1]]

  expect_identical(dim(q), c(20L, 9L))
  expect_true(all(is.finite(q)))
  expect_true(all(apply(q, 1L, function(row) all(diff(row) >= 0))))
})

test_that("all nine trained levels resolve, and an untrained one is refused", {
  skip_if_no_torch()
  config <- timesfm_synthetic_config()
  module <- timesfm_synthetic_module(config)
  context <- as.numeric(12 + sin(seq_len(40) / 3))

  # Spelled the way the catalogue advertises them; seq() and the literals a
  # config parses differ at 0.3 and 0.7.
  q <- timesfm_predict_batch(
    module, list(context), 4L, seq(0.1, 0.9, by = 0.1), config, "cpu"
  )[[1]]
  expect_identical(dim(q), c(4L, 9L))
  expect_true(all(is.finite(q)))

  error <- expect_error(
    timesfm_predict_batch(module, list(context), 4L, c(0.05, 0.5), config, "cpu"),
    class = "zuk_error_quantile_levels"
  )
  expect_s3_class(error, "zuk_error_recoverable")
})
