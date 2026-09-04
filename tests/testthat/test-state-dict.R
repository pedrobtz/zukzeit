test_that("the pinned TimesFM state layout captures all names and parameters", {
  spec <- timesfm_expected_state_spec(timesfm_test_config())
  expect_length(spec, 232L)
  expect_identical(spec[["stacked_xf.0.attn.qkv_proj.weight"]], c(3840L, 1280L))
  expect_identical(
    spec[["output_projection_quantiles.output_layer.weight"]],
    c(10240L, 1280L)
  )
  expect_identical(spec[["tokenizer.hidden_layer.weight"]], c(1280L, 64L))
  expect_equal(sum(vapply(spec, prod, numeric(1))), 231289280)
})

test_that("the native module has a complete identity weight map", {
  skip_if_no_torch()
  config <- timesfm_test_config()
  module <- timesfm_module(config)
  map <- timesfm_weight_map(config)
  expect_length(module$state_dict(), 232L)
  expect_identical(names(map), unname(map))
  expect_setequal(names(module$state_dict()), unname(map))
  expect_true(all(vapply(
    module$state_dict(), function(x) identical(x$device$type, "meta"), logical(1)
  )))
})

test_that("TimesFM header validation is exact and classed", {
  config <- timesfm_test_config()
  spec <- timesfm_expected_state_spec(config)
  metadata <- lapply(spec, function(shape) {
    list(shape = shape, dtype = "F32", data_offsets = c(0, 4))
  })
  expect_invisible(validate_timesfm_state_metadata(metadata, config, "id", "rev"))

  metadata[["stacked_xf.0.attn.qkv_proj.weight"]]$shape <- c(1280L, 3840L)
  error <- expect_error(
    validate_timesfm_state_metadata(metadata, config, "id", "rev"),
    class = "zuk_error_checkpoint"
  )
  expect_identical(error$tensor, "stacked_xf.0.attn.qkv_proj.weight")
  expect_s3_class(error, "zuk_error_internal")
})

test_that("a safetensors file loads to a complete named R torch state dict", {
  skip_if_no_torch()
  path <- tempfile(fileext = ".safetensors")
  safetensors::safe_save_file(
    list(
      "block.weight" = torch::torch_ones(c(2, 3)),
      "block.bias" = torch::torch_zeros(2)
    ),
    path
  )
  record <- list(model_id = "fixture/model", revision = "abc", architecture = "fixture")
  state <- zuk_load_state_dict(path, record, list())
  expect_setequal(names(state), c("block.weight", "block.bias"))
  expect_true(all(vapply(state, inherits, logical(1), what = "torch_tensor")))
})

test_that("missing and corrupt safetensors files are actionable checkpoint errors", {
  record <- list(model_id = "fixture/model", revision = "abc", architecture = "fixture")
  missing <- expect_error(
    zuk_read_safetensors_metadata(tempfile(), record),
    class = "zuk_error_checkpoint"
  )
  expect_identical(missing$model_id, "fixture/model")
  expect_identical(missing$tensor, "model.safetensors")

  corrupt <- tempfile(fileext = ".safetensors")
  writeBin(charToRaw("not safetensors"), corrupt)
  error <- expect_error(
    zuk_read_safetensors_metadata(corrupt, record),
    class = "zuk_error_checkpoint"
  )
  expect_match(conditionMessage(error), "header")
})

test_that("the real pinned checkpoint can load locally when explicitly enabled", {
  skip_if_not(identical(Sys.getenv("ZUK_RUN_CHECKPOINT_TEST"), "true"))
  skip_if_no_torch()
  record <- zuk_catalogue_get("google/timesfm-2.5-200m-pytorch")
  paths <- vapply(record$manifest, function(file) {
    hfhub::hub_download(
      record$model_id,
      file,
      revision = record$revision,
      local_files_only = TRUE
    )
  }, character(1))
  names(paths) <- record$manifest
  config <- jsonlite_read(paths[["config.json"]])
  config$architecture <- normalize_architecture(config)
  state <- zuk_load_state_dict(paths[["model.safetensors"]], record, config)
  expect_length(state, 232L)
  expect_setequal(names(state), names(timesfm_expected_state_spec(config)))
})
