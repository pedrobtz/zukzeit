#' Load a time-series foundation model from the Hugging Face Hub
#'
#' Resolves a model's configuration, looks up the matching architecture
#' constructor in the registry, and returns a uniform [new_zuk_model()] handle.
#'
#' The built-in `"stub"` model is fully wired: passing
#' `"stub"` (or any id beginning with `"stub"`) synthesises a config in memory
#' and skips download and torch. For a supported real checkpoint the function
#' resolves its immutable catalogue revision, downloads the required manifest,
#' validates and loads the safetensors state dict, dispatches to the registered
#' constructor, and optionally retains the constructed handle in a bounded
#' R-session LRU. Scaffold and experimental checkpoints fail before download;
#' use [zuk_download()] only when explicitly prefetching their files.
#'
#' @param model_id Character scalar. A Hub repo id such as
#'   `"ibm-granite/granite-timeseries-ttm-r2"`, or `"stub"` for the built-in
#'   Stage 0 model.
#' @param revision Character scalar. A branch, tag, or commit SHA to pin.
#'   `NULL` uses the immutable revision in the catalogue for curated models and
#'   `"main"` for the built-in stub.
#' @param device Device requested for model construction; resolved through
#'   [zuk_resolve_device()].
#' @param reuse Logical; reuse an identical constructed handle from the bounded
#'   R-session cache. `FALSE` constructs a fresh handle without deleting files.
#' @param ... Load-affecting architecture options included in the resident
#'   cache key and made available through `config$load_options`.
#' @return A [new_zuk_model()].
#' @export
#' @examples
#' model <- zuk_pretrained("stub")
#' zuk_capabilities(model)
#' zuk_unload("stub")
zuk_pretrained <- function(model_id, revision = NULL, device = NULL,
                            reuse = TRUE, ...) {
  if (length(model_id) != 1L || !is.character(model_id) ||
      is.na(model_id) || !nzchar(model_id)) {
    zuk_abort_capability(
      "{.arg model_id} must be one non-empty checkpoint identifier.",
      capability = "model_id",
      requested = model_id,
      supported = "one non-empty string"
    )
  }
  if (length(reuse) != 1L || !is.logical(reuse) || is.na(reuse)) {
    zuk_abort_capability(
      "{.arg reuse} must be one non-missing logical value.",
      model_id = model_id,
      capability = "resident_cache",
      requested = reuse,
      supported = c(TRUE, FALSE)
    )
  }
  record <- if (is_stub_id(model_id)) NULL else {
    zuk_resolve_catalogue_entry(model_id, revision)
  }
  if (!is.null(record) && !identical(record$state, "supported")) {
    zuk_abort_capability(
      c(
        "Checkpoint {.val {model_id}} is not supported for inference.",
        "x" = "Its catalogue state is {.val {record$state}}.",
        "i" = "Use {.fn zuk_models} to discover checkpoints that passed the release gates."
      ),
      model_id = model_id,
      revision = record$revision,
      capability = "model_state",
      requested = record$state,
      supported = "supported"
    )
  }
  if (!is.null(record) && !zuk_registry_has(record$architecture)) {
    zuk_abort_capability(
      "No constructor is registered for checkpoint architecture {.val {record$architecture}}.",
      model_id = record$model_id,
      revision = record$revision,
      capability = "architecture",
      requested = record$architecture,
      supported = zuk_registry_archs()
    )
  }
  revision <- if (is.null(record)) revision %||% "main" else record$revision
  if (length(revision) != 1L || !is.character(revision) ||
      is.na(revision) || !nzchar(revision)) {
    zuk_abort_capability(
      "{.arg revision} must be one non-empty revision string or {.code NULL}.",
      model_id = model_id,
      capability = "revision",
      requested = revision,
      supported = if (is.null(record)) "one revision string" else record$revision
    )
  }
  resolved_device <- zuk_resolve_device(device)
  load_options <- list(...)
  key <- zuk_handle_key(model_id, revision, resolved_device, load_options)
  maximum <- zuk_max_loaded_models()
  zuk_handle_cache_trim(maximum)
  if (isTRUE(reuse) && maximum > 0L) {
    cached <- zuk_handle_cache_get(key)
    if (!is.null(cached)) return(cached)
  }
  paths <- if (is.null(record)) NULL else {
    zuk_download(model_id, revision, progress = FALSE)
  }
  resolved <- zuk_resolve_config(
    model_id,
    revision,
    paths = paths,
    record = record,
    device = resolved_device,
    load_options = load_options
  )
  constructor <- zuk_registry_get(resolved$config$architecture)
  model <- constructor(resolved$config, resolved$weights)
  model$device <- resolved_device
  if (isTRUE(reuse)) {
    zuk_handle_cache_put(
      key,
      model,
      model_id,
      revision,
      resolved_device,
      load_options,
      size_bytes = record$size_bytes %||% 0
    )
  }
  model
}

# Distinguish the built-in stub from a real Hub id.
is_stub_id <- function(model_id) {
  identical(model_id, "stub") || grepl("^stub([-/]|$)", model_id)
}

# Return list(config = <parsed config, incl. `architecture`>, weights = <state
# dict or NULL>). The stub branch is self-contained; the real branch is the
# hfhub plumbing that Stage 1's native architectures build on.
zuk_resolve_config <- function(model_id, revision, paths = NULL, record = NULL,
                                device = "cpu", load_options = list(),
                                call = rlang::caller_env()) {
  if (is_stub_id(model_id)) {
    config <- list(
      architecture = "stub",
      model_id     = model_id,
      revision     = revision,
      max_context  = 512L,
      device       = device,
      load_options = load_options
    )
    return(list(config = config, weights = NULL))
  }

  record <- record %||% zuk_resolve_catalogue_entry(model_id, revision, call)
  paths <- paths %||% zuk_download(model_id, revision, progress = FALSE)
  config_path <- paths[["config.json"]]
  config <- tryCatch(
    jsonlite_read(config_path),
    error = function(e) {
      zuk_abort_checkpoint(
        c(
          "The checkpoint configuration is not valid JSON.",
          "x" = conditionMessage(e)
        ),
        model_id = model_id,
        revision = revision,
        tensor = "config.json",
        expected = "valid checkpoint configuration",
        actual = conditionMessage(e),
        call = call
      )
    }
  )
  # The curated catalogue is package-owned and names the architecture for every
  # entry it publishes, so it decides. Sniffing `config.json` is a fallback for
  # configs the catalogue does not describe: Toto's declares no architecture at
  # all, and Chronos-2 declares a class name no alias covers, so deriving it
  # from the file made two of three supported checkpoints unloadable.
  config$architecture <- record$architecture %||% normalize_architecture(config)
  config$model_id <- model_id
  config$revision <- revision
  config$device <- device
  config$load_options <- load_options
  weight_path <- paths[["model.safetensors"]]
  weights <- zuk_load_state_dict(weight_path, record, config)
  list(config = config, weights = weights, paths = paths)
}

jsonlite_read <- function(path) {
  jsonlite::fromJSON(path, simplifyVector = TRUE)
}

# Map a Hub config to our architecture key. Hugging Face configs expose either a
# top-level `architectures` array (transformers convention) or a `model_type`
# field; different TSFMs use different conventions, so both are handled, then
# normalised through a small alias table to the keys used in the registry.
normalize_architecture <- function(config) {
  arch <- config$architecture %||% config$model_type %||% config$architectures[1]
  if (is.null(arch) || is.na(arch)) {
    zuk_abort_checkpoint(
      "Could not determine the architecture from the checkpoint config.",
      tensor = "config.json",
      expected = "architecture, model_type, or architectures",
      actual = names(config)
    )
  }
  arch <- tolower(as.character(arch)[1])
  aliases <- c(
    tinytimemixer              = "ttm",
    tinytimemixerforprediction = "ttm",
    ttm                        = "ttm",
    timesfm                    = "timesfm",
    timesfmmodelforprediction  = "timesfm",
    patchedtimeseriesdecoder   = "timesfm",
    chronos                    = "chronos2",
    chronosbolt                = "chronos2",
    chronos2                   = "chronos2"
  )
  mapped <- aliases[arch]
  if (is.na(mapped)) arch else unname(mapped)
}
