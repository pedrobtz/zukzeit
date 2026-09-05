# Download, state-dict, and resident model-handle lifecycle.

.zuk_handle_cache <- new.env(parent = emptyenv())
.zuk_lifecycle_state <- new.env(parent = emptyenv())
.zuk_lifecycle_state$clock <- 0

zuk_resolve_catalogue_entry <- function(model_id, revision = NULL,
                                         call = rlang::caller_env()) {
  record <- zuk_catalogue_get(model_id)
  if (is.null(record)) {
    zuk_abort_capability(
      c(
        "Checkpoint {.val {model_id}} is not in the curated catalogue.",
        "i" = "Use {.fn zuk_models} to inspect known checkpoint IDs.",
        "i" = "Architecture registration does not add checkpoint entries."
      ),
      model_id = model_id,
      revision = revision %||% NA_character_,
      capability = "model_id",
      requested = model_id,
      supported = vapply(zuk_catalogue_records(), `[[`, character(1), "model_id"),
      call = call
    )
  }
  resolved_revision <- revision %||% record$revision
  if (!identical(as.character(resolved_revision), record$revision)) {
    zuk_abort_capability(
      c(
        "Revision {.val {resolved_revision}} is not registered for checkpoint {.val {model_id}}.",
        "i" = "The curated immutable revision is {.val {record$revision}}."
      ),
      model_id = model_id,
      revision = resolved_revision,
      capability = "revision",
      requested = resolved_revision,
      supported = record$revision,
      call = call
    )
  }
  record
}

zuk_hub_download_file <- function(record, file, progress) {
  old <- getOption("cli.progress_show_after")
  on.exit(options(cli.progress_show_after = old), add = TRUE)
  if (!isTRUE(progress)) options(cli.progress_show_after = Inf)
  tryCatch(
    hfhub::hub_download(
      repo_id = record$model_id,
      filename = file,
      revision = record$revision
    ),
    error = function(e) {
      zuk_abort_download(
        c(
          "Could not download checkpoint file {.file {file}}.",
          "x" = conditionMessage(e)
        ),
        model_id = record$model_id,
        revision = record$revision,
        file = file,
        parent = e
      )
    }
  )
}

validate_downloaded_manifest <- function(record, paths) {
  missing <- names(paths)[!file.exists(paths)]
  if (length(missing)) {
    zuk_abort_download(
      "Downloaded checkpoint file{?s} {.file {missing}} {?is/are} not available locally.",
      model_id = record$model_id,
      revision = record$revision,
      file = missing
    )
  }
  weight <- paths[["model.safetensors"]]
  if (!is.null(weight) && is.finite(record$size_bytes)) {
    actual <- unname(file.info(weight)$size)
    if (!identical(as.numeric(actual), as.numeric(record$size_bytes))) {
      zuk_abort_checkpoint(
        "Checkpoint file {.file model.safetensors} has an unexpected size.",
        model_id = record$model_id,
        revision = record$revision,
        tensor = "model.safetensors",
        expected = record$size_bytes,
        actual = actual
      )
    }
  }
  invisible(paths)
}

#' Download a curated checkpoint manifest
#'
#' Downloads every file in a checkpoint's static manifest through `hfhub` and
#' returns their local paths without constructing a model. Repeated calls reuse
#' the Hub cache. Disk deletion remains the responsibility of `hfhub`.
#'
#' @param model_id A checkpoint ID returned by [zuk_models()].
#' @param revision `NULL` to use the catalogue's immutable revision, or that
#'   exact revision string.
#' @param progress Logical; show expected-size and Hub download progress.
#' @return Invisibly, a named character vector of local manifest paths.
#' @export
#' @examples
#' # Which checkpoints can be prefetched, and how large they are.
#' zuk_models()[, c("model_id", "size_bytes", "cached")]
#'
#' # Downloading the pinned TimesFM checkpoint transfers about 925 MB.
#' \dontrun{
#' paths <- zuk_download("google/timesfm-2.5-200m-pytorch")
#' names(paths)
#' }
zuk_download <- function(model_id, revision = NULL, progress = interactive()) {
  if (length(progress) != 1L || !is.logical(progress) || is.na(progress)) {
    zuk_abort_capability(
      "{.arg progress} must be one non-missing logical value.",
      model_id = model_id,
      capability = "progress",
      requested = progress,
      supported = c(TRUE, FALSE)
    )
  }
  record <- zuk_resolve_catalogue_entry(model_id, revision)
  if (is.null(record$manifest)) {
    zuk_abort_checkpoint(
      "Checkpoint {.val {model_id}} has no required-file manifest.",
      model_id = record$model_id,
      revision = record$revision,
      expected = "a non-empty file manifest",
      actual = NULL
    )
  }
  if (isTRUE(progress)) {
    cli::cli_alert_info(
      "Resolving {.val {record$model_id}} ({format(structure(record$size_bytes, class = 'object_size'), units = 'auto')})."
    )
  }
  paths <- vapply(
    record$manifest,
    function(file) zuk_hub_download_file(record, file, progress),
    character(1)
  )
  names(paths) <- record$manifest
  validate_downloaded_manifest(record, paths)
  invisible(paths)
}

zuk_read_safetensors_metadata <- function(path, record = NULL) {
  if (length(path) != 1L || !is.character(path) || is.na(path) ||
      !file.exists(path)) {
    zuk_abort_checkpoint(
      "Safetensors checkpoint file {.file {path}} does not exist.",
      model_id = record$model_id %||% NA_character_,
      revision = record$revision %||% NA_character_,
      tensor = "model.safetensors",
      expected = "an existing safetensors file",
      actual = path
    )
  }
  connection <- NULL
  on.exit({
    if (!is.null(connection) && isOpen(connection)) close(connection)
  }, add = TRUE)
  tryCatch(
    {
      connection <- file(path, open = "rb")
      reader <- safetensors::safetensors$new(connection, framework = "torch")
      # `__metadata__` is the safetensors format's reserved header block, not a
      # tensor. Some checkpoints carry one and some do not, so leaving it in
      # makes an otherwise valid state dict look like it is missing an entry.
      reader$metadata[names(reader$metadata) != "__metadata__"]
    },
    error = function(e) {
      zuk_abort_checkpoint(
        c(
          "Could not read the safetensors checkpoint header.",
          "x" = conditionMessage(e)
        ),
        model_id = record$model_id %||% NA_character_,
        revision = record$revision %||% NA_character_,
        tensor = "model.safetensors",
        expected = "a valid safetensors header",
        actual = conditionMessage(e)
      )
    }
  )
}

zuk_load_state_dict <- function(path, record, config) {
  metadata <- zuk_read_safetensors_metadata(path, record)
  if (identical(record$architecture, "timesfm")) {
    validate_timesfm_state_metadata(
      metadata, config, record$model_id, record$revision
    )
  }
  connection <- NULL
  on.exit({
    if (!is.null(connection) && isOpen(connection)) close(connection)
  }, add = TRUE)
  state <- tryCatch(
    {
      connection <- file(path, open = "rb")
      safetensors::safe_load_file(connection, framework = "torch")
    },
    error = function(e) {
      zuk_abort_checkpoint(
        c(
          "Could not load the checkpoint tensors.",
          "x" = conditionMessage(e)
        ),
        model_id = record$model_id,
        revision = record$revision,
        tensor = "model.safetensors",
        expected = "a named R torch state dict",
        actual = conditionMessage(e)
      )
    }
  )
  if (!is.list(state) || is.null(names(state)) ||
      !identical(sort(names(state)), sort(names(metadata))) ||
      !all(vapply(state, inherits, logical(1), what = "torch_tensor"))) {
    zuk_abort_checkpoint(
      "The checkpoint did not decode to a complete named R torch state dict.",
      model_id = record$model_id,
      revision = record$revision,
      tensor = "model.safetensors",
      expected = names(metadata),
      actual = names(state)
    )
  }
  state
}

zuk_canonicalize_options <- function(x) {
  if (is.list(x)) {
    if (!is.null(names(x))) x <- x[order(names(x))]
    return(lapply(x, zuk_canonicalize_options))
  }
  x
}

zuk_handle_key <- function(model_id, revision, device, options) {
  payload <- serialize(
    list(
      model_id = model_id,
      revision = revision,
      device = device,
      options = zuk_canonicalize_options(options)
    ),
    connection = NULL,
    version = 2
  )
  paste(sprintf("%02x", as.integer(payload)), collapse = "")
}

zuk_max_loaded_models <- function() {
  value <- getOption("zuk.max_loaded_models", 1L)
  if (length(value) != 1L || !is.numeric(value) || is.na(value) ||
      !is.finite(value) || value < 0 || value != floor(value)) {
    zuk_abort_capability(
      "Option {.code zuk.max_loaded_models} must be one non-negative integer.",
      capability = "resident_cache_size",
      requested = value,
      supported = "one non-negative integer"
    )
  }
  as.integer(value)
}

zuk_handle_cache_get <- function(key) {
  if (!exists(key, envir = .zuk_handle_cache, inherits = FALSE)) return(NULL)
  entry <- get(key, envir = .zuk_handle_cache, inherits = FALSE)
  .zuk_lifecycle_state$clock <- .zuk_lifecycle_state$clock + 1
  entry$sequence <- .zuk_lifecycle_state$clock
  entry$last_used <- Sys.time()
  assign(key, entry, envir = .zuk_handle_cache)
  zuk_handle_cache_trim(zuk_max_loaded_models())
  entry$handle
}

zuk_handle_cache_trim <- function(maximum = zuk_max_loaded_models()) {
  keys <- ls(envir = .zuk_handle_cache, all.names = TRUE)
  while (length(keys) > maximum) {
    sequences <- vapply(
      keys,
      function(cache_key) get(cache_key, .zuk_handle_cache)$sequence,
      numeric(1)
    )
    evict <- keys[[which.min(sequences)]]
    rm(list = evict, envir = .zuk_handle_cache)
    keys <- ls(envir = .zuk_handle_cache, all.names = TRUE)
  }
  invisible(maximum)
}

zuk_handle_cache_put <- function(key, handle, model_id, revision, device,
                                  options, size_bytes = NA_real_) {
  maximum <- zuk_max_loaded_models()
  if (maximum == 0L) return(invisible(handle))
  .zuk_lifecycle_state$clock <- .zuk_lifecycle_state$clock + 1
  assign(
    key,
    list(
      key = key,
      handle = handle,
      model_id = model_id,
      revision = revision,
      device = device,
      options = options,
      size_bytes = size_bytes,
      last_used = Sys.time(),
      sequence = .zuk_lifecycle_state$clock
    ),
    envir = .zuk_handle_cache
  )
  zuk_handle_cache_trim(maximum)
  invisible(handle)
}

zuk_resident_entries <- function() {
  keys <- ls(envir = .zuk_handle_cache, all.names = TRUE)
  lapply(keys, get, envir = .zuk_handle_cache, inherits = FALSE)
}

#' Inspect checkpoint disk and resident-handle cache state
#'
#' Returns one row per catalogue checkpoint and resident-device combination. A
#' checkpoint without a resident handle has one disk-only row.
#'
#' @return A data frame with columns:
#'
#' * `model_id`, `revision`: character checkpoint identity.
#' * `device`: character resolved device, or `NA` for a disk-only row.
#' * `disk_cached`: logical indicating whether every manifest file is local;
#'   `NA` when no manifest is defined.
#' * `resident`: logical indicating whether a constructed handle is loaded.
#' * `size_bytes`: numeric static checkpoint-size estimate.
#' * `last_used`: `POSIXct` cache-access time, or `NA` for a disk-only row.
#' @export
#' @examples
#' model <- zuk_pretrained("stub")
#' zuk_cache_status()
#' zuk_unload("stub")
zuk_cache_status <- function() {
  records <- zuk_catalogue_records()
  entries <- zuk_resident_entries()
  ids <- unique(c(
    vapply(records, `[[`, character(1), "model_id"),
    vapply(entries, `[[`, character(1), "model_id")
  ))
  rows <- list()
  for (model_id in ids) {
    record <- zuk_catalogue_get(model_id)
    resident <- Filter(function(x) identical(x$model_id, model_id), entries)
    if (!length(resident)) resident <- list(NULL)
    for (entry in resident) {
      rows[[length(rows) + 1L]] <- data.frame(
        model_id = model_id,
        revision = if (is.null(entry)) record$revision else entry$revision,
        device = if (is.null(entry)) NA_character_ else entry$device,
        disk_cached = if (is.null(record)) NA else zuk_manifest_cached(record),
        resident = !is.null(entry),
        size_bytes = if (is.null(record)) entry$size_bytes else record$size_bytes,
        last_used = if (is.null(entry)) as.POSIXct(NA) else entry$last_used,
        stringsAsFactors = FALSE
      )
    }
  }
  if (!length(rows)) {
    return(data.frame(
      model_id = character(), revision = character(), device = character(),
      disk_cached = logical(), resident = logical(), size_bytes = numeric(),
      last_used = as.POSIXct(character())
    ))
  }
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' Unload resident model handles
#'
#' Removes matching strong references from the current R session. Downloaded
#' files are not deleted.
#'
#' @param model_id,revision,device Optional filters; `NULL` matches every value.
#' @return Invisibly, the number of resident handles removed.
#' @export
#' @examples
#' model <- zuk_pretrained("stub")
#' zuk_cache_status()$resident
#'
#' # Evict one checkpoint, or every resident handle with no arguments.
#' zuk_unload("stub")
#' zuk_cache_status()$resident
zuk_unload <- function(model_id = NULL, revision = NULL, device = NULL) {
  if (!is.null(device)) device <- zuk_resolve_device(device)
  entries <- zuk_resident_entries()
  remove <- vapply(entries, function(entry) {
    (is.null(model_id) || identical(entry$model_id, model_id)) &&
      (is.null(revision) || identical(entry$revision, revision)) &&
      (is.null(device) || identical(entry$device, device))
  }, logical(1))
  keys <- vapply(entries[remove], `[[`, character(1), "key")
  if (length(keys)) rm(list = keys, envir = .zuk_handle_cache)
  invisible(length(keys))
}
