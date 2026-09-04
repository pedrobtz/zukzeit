# Static, offline checkpoint catalogue.
#
# Entries describe executable state, not roadmap intent. Cache state is the
# only dynamic column and is computed exclusively through local-only hfhub
# probes; listing models never fetches a config or a weight file.

zuk_catalogue_records <- function() {
  list(
    list(
      model_id = "google/timesfm-2.5-200m-pytorch",
      architecture = "timesfm",
      revision = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
      state = "supported",
      max_context = 16256L,
      quantile_levels = seq(0.1, 0.9, by = 0.1),
      multivariate = FALSE,
      past_covariates = FALSE,
      future_covariates = FALSE,
      n_params = 231289280,
      size_bytes = 925181104,
      license = "Apache-2.0",
      manifest = c("config.json", "model.safetensors")
    ),
    list(
      model_id = "ibm-granite/granite-timeseries-ttm-r2",
      architecture = "ttm",
      revision = "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f",
      state = "scaffold",
      max_context = 512L,
      quantile_levels = numeric(),
      multivariate = FALSE,
      past_covariates = FALSE,
      future_covariates = FALSE,
      n_params = 805280,
      size_bytes = 3240592,
      license = "Apache-2.0",
      manifest = c("config.json", "model.safetensors")
    )
  )
}

zuk_probe_cached_file <- function(model_id, revision, file) {
  isTRUE(tryCatch(
    {
      hfhub::hub_download(
        repo_id = model_id,
        filename = file,
        revision = revision,
        local_files_only = TRUE
      )
      TRUE
    },
    error = function(e) FALSE
  ))
}

zuk_manifest_cached <- function(record, probe = zuk_probe_cached_file) {
  manifest <- record$manifest
  if (is.null(manifest)) {
    return(NA)
  }
  all(vapply(
    manifest,
    function(file) probe(record$model_id, record$revision, file),
    logical(1)
  ))
}

zuk_catalogue_frame <- function(records = zuk_catalogue_records(),
                                 probe = zuk_probe_cached_file) {
  if (!length(records)) {
    out <- data.frame(
      model_id = character(),
      architecture = character(),
      revision = character(),
      state = character(),
      max_context = integer(),
      multivariate = logical(),
      past_covariates = logical(),
      future_covariates = logical(),
      n_params = numeric(),
      size_bytes = numeric(),
      cached = logical(),
      license = character(),
      stringsAsFactors = FALSE
    )
    out$quantile_levels <- I(list())
    return(out[c(
      "model_id", "architecture", "revision", "state", "max_context",
      "quantile_levels", "multivariate", "past_covariates",
      "future_covariates", "n_params", "size_bytes", "cached", "license"
    )])
  }

  out <- data.frame(
    model_id = vapply(records, `[[`, character(1), "model_id"),
    architecture = vapply(records, `[[`, character(1), "architecture"),
    revision = vapply(records, `[[`, character(1), "revision"),
    state = vapply(records, `[[`, character(1), "state"),
    max_context = vapply(records, `[[`, integer(1), "max_context"),
    multivariate = vapply(records, `[[`, logical(1), "multivariate"),
    past_covariates = vapply(records, `[[`, logical(1), "past_covariates"),
    future_covariates = vapply(records, `[[`, logical(1), "future_covariates"),
    n_params = vapply(records, `[[`, numeric(1), "n_params"),
    size_bytes = vapply(records, `[[`, numeric(1), "size_bytes"),
    cached = vapply(records, zuk_manifest_cached, logical(1), probe = probe),
    license = vapply(records, `[[`, character(1), "license"),
    stringsAsFactors = FALSE
  )
  out$quantile_levels <- I(lapply(records, `[[`, "quantile_levels"))
  out[c(
    "model_id", "architecture", "revision", "state", "max_context",
    "quantile_levels", "multivariate", "past_covariates",
    "future_covariates", "n_params", "size_bytes", "cached", "license"
  )]
}

#' List curated time-series foundation-model checkpoints
#'
#' Returns static checkpoint metadata without downloading configs, weights, or
#' constructing a model. Cache availability is checked using local-only
#' `hfhub` probes. The safe default lists only checkpoints that have passed both
#' architecture-conformance and numerical-parity gates; `state = NULL` also
#' includes experimental and scaffold records.
#'
#' @param state One of `"supported"`, `"experimental"`, or `"scaffold"`;
#'   `NULL` returns every catalogue entry.
#' @return A data frame with one row per checkpoint. `quantile_levels` is a
#'   list-column and `cached` is `TRUE`, `FALSE`, or `NA` when no manifest has
#'   been defined.
#' @export
#' @examples
#' zuk_models()
zuk_models <- function(state = "supported") {
  allowed <- c("supported", "experimental", "scaffold")
  if (!is.null(state) &&
      (length(state) != 1L || !is.character(state) || is.na(state) ||
       !state %in% allowed)) {
    zuk_abort_capability(
      "{.arg state} must be {.val supported}, {.val experimental}, {.val scaffold}, or {.code NULL}.",
      capability = "catalogue_state",
      requested = state,
      supported = allowed
    )
  }
  out <- zuk_catalogue_frame()
  if (!is.null(state)) {
    out <- out[out$state == state, , drop = FALSE]
  }
  rownames(out) <- NULL
  out
}

zuk_catalogue_get <- function(model_id) {
  records <- zuk_catalogue_records()
  matched <- vapply(
    records,
    function(record) identical(record$model_id, as.character(model_id)),
    logical(1)
  )
  if (!any(matched)) NULL else records[[which(matched)[[1]]]]
}
