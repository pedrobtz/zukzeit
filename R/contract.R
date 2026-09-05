# The public architecture contract.
#
# zukzeit is an engine: the surface below is what architecture implementations are
# written against, so it is versioned and changes only with a deprecation cycle.
# Everything else in the package (loaders, adapters, batching) is free to move.

#' The zukzeit architecture contract
#'
#' `zukzeit` is an inference engine: it owns the path from a pretrained checkpoint
#' to predictive quantiles, and nothing above it. An *architecture* is a
#' constructor and forward pass satisfying the contract described here.
#'
#' [zuk_register_arch()] associates a constructor with an architecture key in
#' the current R session. Registration does not add checkpoint metadata to the
#' package-owned catalogue and does not make arbitrary model IDs available to
#' [zuk_pretrained()] or model-ID-based adapters. A checkpoint is exposed by
#' those APIs only after it is curated in `zukzeit`, including an immutable
#' revision, manifest, licence metadata, contract conformance, and numerical
#' parity. Independently constructed [new_zuk_model()] handles can still use
#' the framework-neutral [forecast()] path.
#'
#' Verify an implementation with [zuk_check_architecture()].
#'
#' @section The constructor:
#' A constructor is a function of `(config, weights)` returning a
#' [new_zuk_model()]:
#'
#' * `config` — a named list parsed from the checkpoint's `config.json`, with
#'   `architecture`, `model_id`, and `revision` filled in by the loader.
#' * `weights` — a state dict, or `NULL` for weight-free architectures.
#'
#' The constructor must not perform inference, and must not download anything
#' beyond what the loader already resolved.
#'
#' @section The forward pass:
#' Every architecture supplies `predict_fn`, and may additionally supply
#' `predict_batch_fn` for true vectorised inference.
#'
#' ```
#' predict_fn(context, h, quantile_levels) -> matrix[h, q]
#' ```
#'
#' * `context` — a numeric vector of observed history, **oldest first**, already
#'   truncated to `max_context` and with `NA` removed by [zuk_run_batches()].
#' * `h` — a positive integer horizon.
#' * `quantile_levels` — a sorted numeric vector in `(0, 1)`.
#' * Returns a numeric matrix with `h` rows and `length(quantile_levels)`
#'   columns. Column `j` holds the forecasts at `quantile_levels[j]`.
#'
#' ```
#' predict_batch_fn(contexts, horizons, quantile_levels, device) -> list(matrix)
#' ```
#'
#' * `contexts` — a list of numeric context vectors.
#' * `horizons` — an integer vector of per-series horizons, aligned to
#'   `contexts`.
#' * `device` — a resolved device string (never `"auto"`); see
#'   [zuk_resolve_device()].
#' * Returns a list of quantile matrices, aligned to and the same length as
#'   `contexts`.
#'
#' When `predict_batch_fn` is `NULL`, [zuk_run_batches()] loops `predict_fn`.
#' When both are present they must agree: the batch path is an optimisation,
#' never a different model.
#'
#' @section Grouped inputs (contract 1.1):
#' An architecture written against contract `1.1.0` or later receives a fifth
#' argument:
#'
#' ```
#' predict_batch_fn(contexts, horizons, quantile_levels, device, groups)
#' ```
#'
#' Every series is a row of `contexts`. `groups` says what those rows mean, with
#' one entry per row:
#'
#' * `id` --- task membership. Rows sharing an id exchange information; rows in
#'   different tasks do not. Values are labels, not indices.
#' * `target` --- logical; whether the row is forecast and returned.
#' * `future` --- known future values for the row, or `NULL`. When present it is
#'   exactly as long as that row's horizon.
#'
#' Those two fields express all three row kinds without a separate vocabulary:
#' a **target** has no future, a **past-only covariate** is a non-target with no
#' future, and a **future-known covariate** is a non-target with future values.
#'
#' [zuk_run_batches()] returns one matrix per **target** row, in the order those
#' rows appear. Rows of one task are never split across batches, because a task
#' is attended together.
#'
#' The engine dispatches on the version an architecture declares: a `1.0.0`
#' architecture is called with four arguments and *cannot* receive `groups`, so
#' it can neither break on one nor silently ignore one. `predict_fn` is
#' unchanged in both versions and remains the single-series fallback.
#'
#' @section Invariants:
#' These hold for every architecture and are what [zuk_check_architecture()]
#' asserts:
#'
#' * **Shape.** The result is a numeric matrix, `h` by `length(quantile_levels)`.
#' * **Monotonicity.** Within a row, forecasts are non-decreasing across
#'   quantile levels. A model whose raw output can cross must sort before
#'   returning.
#' * **Finiteness.** No `NA`, `NaN`, or `Inf` for a finite, non-empty context.
#' * **Context limit.** The model never requires more history than the
#'   `max_context` it declares in its [new_zuk_capabilities()].
#' * **Return alignment.** One matrix per target row. Without `groups` every row
#'   is a target, so the result aligns to `contexts` --- the rule generalises
#'   rather than changes.
#' * **Empty context.** An empty context raises an error rather than returning
#'   `NA` — silent nonsense is worse than a stop.
#' * **Batch agreement.** If `predict_batch_fn` is supplied, it matches
#'   `predict_fn` within numerical tolerance.
#'
#' @section Capabilities:
#' Capabilities are declarations the engine enforces *before* inference, so an
#' unsupported request costs nothing. Declaring a capability the forward pass
#' does not honour is a bug in the architecture. Contract 1.0 has channels
#' only for univariate numeric context and predictive quantiles. Contract 1.1
#' adds grouped targets and covariates, declarable only by an architecture at
#' `1.1.0` or later and only for what its fixtures demonstrate. Sample-path and
#' fine-tuning declarations must remain `FALSE` at every version, because no
#' channel exists for them.
#' Native-quantile checkpoints declare their exact trained levels when those
#' levels are fixed, and the engine validates them together with horizon and
#' context limits before calling the architecture. See
#' [new_zuk_capabilities()].
#'
#' @section Versioning:
#' [new_zuk_model()] stamps every handle with [zuk_contract_version()]. The
#' contract follows semantic versioning: the major component changes only for a
#' breaking change to the signatures or invariants above, and never without a
#' deprecation cycle. Architectures may record the version they were written
#' against by passing `contract_version` explicitly.
#'
#' @seealso [zuk_register_arch()], [new_zuk_model()],
#'   [zuk_check_architecture()], [new_zuk_capabilities()]
#' @name zuk-architecture-contract
NULL

#' The architecture contract version
#'
#' The version of the contract described in `?`[zuk-architecture-contract] that
#' this installation of zukzeit implements. Stamped onto every [new_zuk_model()].
#'
#' @return A [package_version].
#' @export
#' @examples
#' zuk_contract_version()
#'
#' # Grouped inputs are available from 1.1.0 onwards.
#' zuk_contract_version() >= package_version("1.1.0")
zuk_contract_version <- function() {
  # 1.1.0, not 2.0.0. The grouped-input extension is purely additive: an
  # architecture written against 1.0 keeps its four-argument forward pass, is
  # never handed a `groups` record, and its return alignment is unchanged.
  # Semantic versioning makes that a minor bump, and the section above promises
  # the major component moves only for a breaking change. The roadmap calls the
  # feature "contract v2"; that is its name, not its version.
  package_version("1.1.0")
}

# Whether an architecture was written against the grouped-input extension.
zuk_supports_groups <- function(model) {
  !is.null(model$contract_version) &&
    package_version(model$contract_version) >= package_version("1.1.0")
}
