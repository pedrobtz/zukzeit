cache_test_record <- function(model_id) {
  list(
    model_id = model_id,
    architecture = "cache-test",
    revision = paste0(strrep(substr(model_id, nchar(model_id), nchar(model_id)), 40)),
    state = "supported",
    max_context = 32L,
    quantile_levels = NULL,
    multivariate = FALSE,
    past_covariates = FALSE,
    future_covariates = FALSE,
    n_params = 0,
    size_bytes = 0,
    license = "MIT",
    manifest = c("config.json", "model.safetensors")
  )
}

with_cache_test_loader <- function(code) {
  records <- list(cache_test_record("fixture/a"), cache_test_record("fixture/b"))
  builds <- 0L
  constructor <- function(config, weights) {
    builds <<- builds + 1L
    new_zuk_model(
      architecture = "cache-test",
      config = config,
      capabilities = new_zuk_capabilities("cache-test", 32L),
      predict_fn = function(context, h, quantile_levels) {
        matrix(0, nrow = h, ncol = length(quantile_levels))
      },
      model_id = config$model_id,
      revision = config$revision,
      device = config$device,
      params = list(build = builds, identity = new.env(parent = emptyenv()))
    )
  }
  zuk_register_arch("cache-test", constructor, overwrite = TRUE)
  on.exit(rm("cache-test", envir = zukzeit:::.zuk_registry), add = TRUE)
  on.exit(zuk_unload(), add = TRUE)
  zuk_unload()

  local_mocked_bindings(
    zuk_catalogue_records = function() records,
    zuk_download = function(model_id, revision, progress) {
      c("config.json" = "config.json", "model.safetensors" = "model.safetensors")
    },
    zuk_resolve_config = function(model_id, revision, paths, record, device,
                                   load_options) {
      list(
        config = list(
          architecture = "cache-test",
          model_id = model_id,
          revision = revision,
          device = device,
          load_options = load_options
        ),
        weights = NULL,
        paths = paths
      )
    },
    zuk_resolve_device = function(device = NULL) device %||% "cpu",
    zuk_manifest_cached = function(...) FALSE
  )
  force(code)
}

test_that("identical loads reuse a handle and the default LRU evicts", {
  old <- getOption("zuk.max_loaded_models")
  on.exit(options(zuk.max_loaded_models = old), add = TRUE)
  options(zuk.max_loaded_models = 1L)

  with_cache_test_loader({
    first <- zuk_pretrained("fixture/a", device = "cpu")
    again <- zuk_pretrained("fixture/a", device = "cpu")
    expect_identical(first$params$identity, again$params$identity)
    expect_identical(first$params$build, 1L)

    second <- zuk_pretrained("fixture/b", device = "cpu")
    expect_identical(second$params$build, 2L)
    status <- zuk_cache_status()
    expect_identical(status$model_id[status$resident], "fixture/b")

    rebuilt <- zuk_pretrained("fixture/a", device = "cpu")
    expect_identical(rebuilt$params$build, 3L)
    expect_false(identical(first$params$identity, rebuilt$params$identity))
  })
})

test_that("device and load-affecting options form distinct resident keys", {
  old <- getOption("zuk.max_loaded_models")
  on.exit(options(zuk.max_loaded_models = old), add = TRUE)
  options(zuk.max_loaded_models = 4L)

  with_cache_test_loader({
    cpu <- zuk_pretrained("fixture/a", device = "cpu", precision = "float32")
    mps <- zuk_pretrained("fixture/a", device = "mps", precision = "float32")
    alternate <- zuk_pretrained("fixture/a", device = "cpu", precision = "float64")
    reordered <- zuk_pretrained(
      "fixture/a", device = "cpu", second = 2, first = 1
    )
    reordered_again <- zuk_pretrained(
      "fixture/a", device = "cpu", first = 1, second = 2
    )

    expect_false(identical(cpu$params$identity, mps$params$identity))
    expect_false(identical(cpu$params$identity, alternate$params$identity))
    expect_identical(reordered$params$identity, reordered_again$params$identity)
    status <- zuk_cache_status()
    expect_setequal(status$device[status$resident], c("cpu", "mps"))
    expect_identical(sum(status$resident), 4L)
  })
})

test_that("reuse FALSE bypasses construction reuse and zero disables storage", {
  old <- getOption("zuk.max_loaded_models")
  on.exit(options(zuk.max_loaded_models = old), add = TRUE)
  options(zuk.max_loaded_models = 1L)

  with_cache_test_loader({
    cached <- zuk_pretrained("fixture/a")
    fresh <- zuk_pretrained("fixture/a", reuse = FALSE)
    expect_false(identical(cached$params$identity, fresh$params$identity))
    expect_identical(zuk_pretrained("fixture/a")$params$identity, cached$params$identity)

    zuk_unload()
    options(zuk.max_loaded_models = 0L)
    one <- zuk_pretrained("fixture/a")
    two <- zuk_pretrained("fixture/a")
    expect_false(identical(one$params$identity, two$params$identity))
    expect_false(any(zuk_cache_status()$resident))
  })
})

test_that("unload filters resident handles and never changes disk state", {
  old <- getOption("zuk.max_loaded_models")
  on.exit(options(zuk.max_loaded_models = old), add = TRUE)
  options(zuk.max_loaded_models = 3L)

  with_cache_test_loader({
    zuk_pretrained("fixture/a", device = "cpu")
    zuk_pretrained("fixture/a", device = "mps")
    zuk_pretrained("fixture/b", device = "cpu")
    expect_identical(zuk_unload("fixture/a", device = "cpu"), 1L)
    status <- zuk_cache_status()
    expect_false(any(status$resident & status$model_id == "fixture/a" & status$device == "cpu"))
    expect_true(any(status$resident & status$model_id == "fixture/a" & status$device == "mps"))
    expect_false(any(status$disk_cached, na.rm = TRUE))
  })
})

test_that("explicit prefetch downloads the manifest without construction", {
  skip_if_no_torch()
  config_path <- tempfile(fileext = ".json")
  writeLines('{"model_type":"fixture"}', config_path)
  weight_path <- tempfile(fileext = ".safetensors")
  safetensors::safe_save_file(list(weight = torch::torch_ones(1)), weight_path)
  calls <- character()
  record <- cache_test_record("fixture/a")
  record$state <- "scaffold"
  record$size_bytes <- unname(file.info(weight_path)$size)

  local_mocked_bindings(zuk_catalogue_records = function() list(record))
  local_mocked_bindings(
    hub_download = function(repo_id, filename, revision, ...) {
      calls <<- c(calls, filename)
      if (identical(filename, "config.json")) config_path else weight_path
    },
    .package = "hfhub"
  )
  paths <- zuk_download("fixture/a", progress = FALSE)
  expect_setequal(names(paths), record$manifest)
  expect_setequal(calls, record$manifest)
  expect_identical(ls(envir = zukzeit:::.zuk_handle_cache), character())
})

test_that("safetensors readers close their file connections deterministically", {
  skip_if_no_torch()
  path <- tempfile(fileext = ".safetensors")
  safetensors::safe_save_file(list(weight = torch::torch_ones(1)), path)
  gc()
  before <- rownames(showConnections())
  record <- cache_test_record("fixture/a")
  record$architecture <- "cache-test"
  record$size_bytes <- unname(file.info(path)$size)

  metadata <- zukzeit:::zuk_read_safetensors_metadata(path, record)
  state <- zukzeit:::zuk_load_state_dict(path, record, config = list())

  expect_named(metadata, "weight")
  expect_s3_class(state$weight, "torch_tensor")
  expect_setequal(rownames(showConnections()), before)
})

test_that("download failures retain external policy and checkpoint identity", {
  record <- cache_test_record("fixture/a")
  local_mocked_bindings(zuk_catalogue_records = function() list(record))
  local_mocked_bindings(
    hub_download = function(...) stop("offline"),
    .package = "hfhub"
  )
  error <- expect_error(
    zuk_download("fixture/a", progress = FALSE),
    class = "zuk_error_download"
  )
  expect_s3_class(error, "zuk_error_external")
  expect_identical(error$model_id, "fixture/a")
  expect_identical(error$file, "config.json")
})

test_that("unknown revisions and unregistered architectures fail before download", {
  record <- cache_test_record("fixture/a")
  reached_download <- FALSE
  local_mocked_bindings(
    zuk_catalogue_records = function() list(record),
    zuk_download = function(...) {
      reached_download <<- TRUE
      stop("download reached")
    }
  )
  revision_error <- expect_error(
    zuk_pretrained("fixture/a", revision = strrep("f", 40)),
    class = "zuk_error_capability"
  )
  expect_identical(revision_error$capability, "revision")
  expect_false(reached_download)

  record$architecture <- "not-registered"
  architecture_error <- expect_error(
    zuk_pretrained("fixture/a"),
    class = "zuk_error_capability"
  )
  expect_identical(architecture_error$capability, "architecture")
  expect_false(reached_download)
})

test_that("every supported checkpoint loads through the public loader", {
  # Regression: the port tests construct handles by calling each architecture's
  # constructor directly, which bypasses zuk_pretrained() entirely. Deriving the
  # architecture from config.json rather than from the curated record left two
  # of three supported checkpoints unloadable through the documented path --
  # Toto's config names no architecture, and Chronos-2 names a class no alias
  # covers -- and nothing caught it.
  # This one walks every supported row, so it needs all of them.
  for (architecture in zuk_models()$architecture) skip_unless_checkpoint(architecture)
  skip_if_no_torch()

  supported <- zuk_models()
  for (index in seq_len(nrow(supported))) {
    model <- zuk_pretrained(supported$model_id[[index]])
    expect_s3_class(model, "zuk_model")
    expect_identical(model$architecture, supported$architecture[[index]])
    expect_identical(model$model_id, supported$model_id[[index]])
    expect_identical(model$capabilities$max_context, supported$max_context[[index]])
    zuk_unload()
  }
})
