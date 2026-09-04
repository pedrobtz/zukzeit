test_that("the stub architecture is registered at load", {
  expect_true(zuk_registry_has("stub"))
  expect_true("stub" %in% zuk_registry_archs())
  expect_type(zuk_registry_get("stub"), "closure")
})

test_that("registration guards against silent clobbering", {
  ctor <- function(config, weights) NULL
  zuk_register_arch("test-arch", ctor)
  on.exit(rm("test-arch", envir = zukzeit:::.zuk_registry), add = TRUE)
  expect_error(zuk_register_arch("test-arch", ctor), "already registered")
  expect_silent(zuk_register_arch("test-arch", ctor, overwrite = TRUE))
})

test_that("constructor registration does not extend the curated catalogue", {
  before <- zuk_models(state = NULL)
  reached <- FALSE
  ctor <- function(config, weights) {
    reached <<- TRUE
    NULL
  }
  zuk_register_arch("external-test", ctor)
  on.exit(rm("external-test", envir = zukzeit:::.zuk_registry), add = TRUE)

  expect_identical(zuk_models(state = NULL), before)
  error <- expect_error(
    zuk_pretrained("external/model"),
    class = "zuk_error_capability"
  )
  expect_identical(error$capability, "model_id")
  expect_false(reached)
})

test_that("unknown architectures error helpfully", {
  expect_error(zuk_registry_get("no-such-arch"), "No constructor registered")
})
