# The conformance harness is the engine's guarantee to architecture authors, so
# it needs to be trustworthy in both directions: every built-in architecture
# must pass, and each invariant must actually catch its violation. The negative
# cases below are deliberately broken architectures, one per invariant.

# A conforming baseline the negative cases mutate.
conforming_arch <- function(max_context = 128L,
                            predict_fn = NULL,
                            predict_batch_fn = NULL,
                            contract_version = zuk_contract_version()) {
  predict_fn <- predict_fn %||% function(context, h, quantile_levels) {
    if (length(context) == 0L) stop("empty context")
    last <- context[length(context)]
    outer(rep(last, h), stats::qnorm(quantile_levels), `+`)
  }
  function(config, weights) {
    new_zuk_model(
      architecture     = "conforming",
      config           = config,
      capabilities     = new_zuk_capabilities("conforming", max_context = max_context),
      predict_fn       = predict_fn,
      predict_batch_fn = predict_batch_fn,
      contract_version = contract_version
    )
  }
}

# Run the harness in reporting mode. `error = FALSE` warns by design, which is
# right for a user at the console but noise in a suite, so swallow it here --- the
# warn-vs-abort behaviour itself is asserted in its own test at the bottom.
check_report <- function(constructor, ...) {
  suppressWarnings(zuk_check_architecture(constructor, ..., error = FALSE))
}

# Names of the checks that failed, for terse assertions below.
failures <- function(report) {
  report$name[!is.na(report$ok) & !report$ok]
}

test_that("a conforming architecture passes every check", {
  report <- check_report(conforming_arch())
  expect_length(failures(report), 0L)
  expect_true(all(report$ok[!is.na(report$ok)]))
})

test_that("the built-in stub satisfies the contract", {
  # error = TRUE: the stub failing the contract should break the build.
  expect_no_error(
    zuk_check_architecture(
      function(config, weights) stub_constructor(config, weights),
      error = TRUE
    )
  )
})

test_that("the TTM scaffold does not yet satisfy the contract", {
  # Their forward passes abort by design until the numerical port lands. The
  # harness must report that as a failure, not mistake it for a clean refusal:
  # this is the gate those ports have to turn green.
  ttm <- check_report(function(config, weights) ttm_constructor(config))
  expect_true("predict_fn returns an h x q numeric matrix" %in% failures(ttm))

})

test_that("a non-zuk_model constructor fails fast", {
  report <- check_report(function(config, weights) list())
  expect_true("constructor returns a zuk_model" %in% failures(report))
  # Construction failing short-circuits the rest rather than cascading.
  expect_identical(nrow(report), 1L)
})

test_that("a constructor that errors is reported, not propagated", {
  report <- check_report(function(config, weights) stop("boom"))
  expect_true("constructor returns a zuk_model" %in% failures(report))
  expect_match(report$message[1], "boom")
})

test_that("crossing quantiles are caught", {
  crossing <- function(context, h, quantile_levels) {
    if (length(context) == 0L) stop("empty context")
    # Reversed: high levels get low values.
    outer(rep(context[length(context)], h), rev(stats::qnorm(quantile_levels)), `+`)
  }
  report <- check_report(conforming_arch(predict_fn = crossing))
  expect_true("quantiles are non-decreasing across levels" %in% failures(report))
})

test_that("non-finite forecasts are caught", {
  nonfinite <- function(context, h, quantile_levels) {
    m <- outer(rep(context[length(context)], h), stats::qnorm(quantile_levels), `+`)
    m[1, 1] <- NA_real_
    m
  }
  report <- check_report(conforming_arch(predict_fn = nonfinite))
  expect_true("forecasts are finite" %in% failures(report))
})

test_that("a wrong-shaped result is caught", {
  wrong_shape <- function(context, h, quantile_levels) {
    matrix(0, nrow = h + 1L, ncol = length(quantile_levels))
  }
  report <- check_report(conforming_arch(predict_fn = wrong_shape))
  expect_true("predict_fn returns an h x q numeric matrix" %in% failures(report))
  # Shape failing marks the dependent checks not-applicable rather than erroring.
  expect_true(is.na(report$ok[report$name == "forecasts are finite"]))
  expect_true(is.na(report$ok[report$name == "quantiles are non-decreasing across levels"]))
})

test_that("silently accepting an empty context is caught", {
  permissive <- function(context, h, quantile_levels) {
    last <- if (length(context)) context[length(context)] else 0
    outer(rep(last, h), stats::qnorm(quantile_levels), `+`)
  }
  report <- check_report(conforming_arch(predict_fn = permissive))
  expect_true("empty context errors rather than returning NA" %in% failures(report))
})

test_that("a disagreeing batch path is caught", {
  offset_batch <- function(contexts, horizons, quantile_levels, device) {
    lapply(seq_along(contexts), function(i) {
      ctx <- contexts[[i]]
      outer(rep(ctx[length(ctx)] + 1, horizons[[i]]), stats::qnorm(quantile_levels), `+`)
    })
  }
  report <- check_report(conforming_arch(predict_batch_fn = offset_batch))
  expect_true("predict_batch_fn agrees with predict_fn" %in% failures(report))
})

test_that("an agreeing batch path passes", {
  agreeing_batch <- function(contexts, horizons, quantile_levels, device) {
    lapply(seq_along(contexts), function(i) {
      ctx <- contexts[[i]]
      outer(rep(ctx[length(ctx)], horizons[[i]]), stats::qnorm(quantile_levels), `+`)
    })
  }
  report <- check_report(conforming_arch(predict_batch_fn = agreeing_batch))
  expect_length(failures(report), 0L)
})

test_that("no batch path is not-applicable rather than a failure", {
  report <- check_report(conforming_arch())
  expect_true(is.na(report$ok[report$name == "predict_batch_fn agrees with predict_fn"]))
})

test_that("an unexercised max_context is not-applicable rather than a pass", {
  # The default 64-value probe cannot reach a 128-value limit, let alone
  # TimesFM's 16,256. Reporting a pass there would overstate what ran.
  short <- check_report(conforming_arch(max_context = 128L))
  row <- short$ok[short$name == "respects its declared max_context"]
  expect_true(is.na(row))
  expect_match(
    short$message[short$name == "respects its declared max_context"],
    "Not exercised"
  )

  # Given a context long enough to truncate, the check actually runs.
  exercised <- check_report(
    conforming_arch(max_context = 32L),
    context = default_check_context(128L)
  )
  expect_true(exercised$ok[exercised$name == "respects its declared max_context"])
})

test_that("a mismatched major contract version is caught", {
  older <- conforming_arch(contract_version = package_version("0.9.0"))
  report <- check_report(older)
  expect_true("stamps a contract version" %in% failures(report))
})

test_that("error = TRUE aborts, error = FALSE reports every problem at once", {
  broken <- conforming_arch(predict_fn = function(context, h, quantile_levels) {
    matrix(0, nrow = h + 1L, ncol = length(quantile_levels))
  })
  expect_error(zuk_check_architecture(broken, error = TRUE), "contract")
  expect_warning(report <- zuk_check_architecture(broken, error = FALSE), "contract")
  expect_gt(length(failures(report)), 1L)
})

test_that("new_zuk_model stamps the current contract version", {
  model <- conforming_arch()(list(), NULL)
  expect_identical(model$contract_version, zuk_contract_version())
  expect_identical(zuk_contract_version()$major, 1L)
})

test_that("the conformance harness leaves the caller's RNG state alone", {
  # The default probe context is seeded for reproducibility, but it runs as a
  # default argument of an exported function: leaving set.seed() in place
  # silently reseeded whatever session called the gate.
  arch <- function(config, weights) {
    new_zuk_model(
      architecture = "seedless",
      config = config,
      capabilities = new_zuk_capabilities("seedless", max_context = 128L),
      predict_fn = function(context, h, quantile_levels) {
        if (length(context) == 0L) stop("empty context")
        outer(rep(context[length(context)], h), stats::qnorm(quantile_levels), `+`)
      }
    )
  }

  set.seed(42)
  expected <- runif(3)

  set.seed(42)
  before <- get(".Random.seed", envir = globalenv())
  suppressMessages(zuk_check_architecture(arch, error = FALSE))

  expect_identical(get(".Random.seed", envir = globalenv()), before)
  expect_equal(runif(3), expected)
})

test_that("an unseeded session is left unseeded", {
  arch <- function(config, weights) {
    new_zuk_model(
      architecture = "seedless",
      config = config,
      capabilities = new_zuk_capabilities("seedless", max_context = 128L),
      predict_fn = function(context, h, quantile_levels) {
        if (length(context) == 0L) stop("empty context")
        outer(rep(context[length(context)], h), stats::qnorm(quantile_levels), `+`)
      }
    )
  }

  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    previous <- get(".Random.seed", envir = globalenv())
    on.exit(assign(".Random.seed", previous, envir = globalenv()), add = TRUE)
    rm(".Random.seed", envir = globalenv())
  }

  suppressMessages(zuk_check_architecture(arch, error = FALSE))
  expect_false(exists(".Random.seed", envir = globalenv(), inherits = FALSE))
})
