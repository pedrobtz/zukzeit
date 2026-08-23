# The public architecture contract.
#
# tsfm is an engine: the surface below is what architecture implementations are
# written against, so it is versioned and changes only with a deprecation cycle.
# Everything else in the package (loaders, adapters, batching) is free to move.

#' The tsfm architecture contract
#'
#' `tsfm` is an inference engine: it owns the path from a pretrained checkpoint
#' to predictive quantiles, and nothing above it. An *architecture* is a
#' constructor and forward pass satisfying the contract described here.
#'
#' [tsfm_register_arch()] associates a constructor with an architecture key in
#' the current R session. Registration does not add checkpoint metadata to the
#' package-owned catalogue and does not make arbitrary model IDs available to
#' [tsfm_pretrained()] or model-ID-based adapters. A checkpoint is exposed by
#' those APIs only after it is curated in `tsfm`, including an immutable
#' revision, manifest, licence metadata, contract conformance, and numerical
#' parity. Independently constructed [new_tsfm_model()] handles can still use
#' the framework-neutral [forecast()] path.
#'
#' Verify an implementation with [tsfm_check_architecture()].
#'
#' @section The constructor:
#' A constructor is a function of `(config, weights)` returning a
#' [new_tsfm_model()]:
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
#'   truncated to `max_context` and with `NA` removed by [tsfm_run_batches()].
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
#'   [tsfm_resolve_device()].
#' * Returns a list of quantile matrices, aligned to and the same length as
#'   `contexts`.
#'
#' When `predict_batch_fn` is `NULL`, [tsfm_run_batches()] loops `predict_fn`.
#' When both are present they must agree: the batch path is an optimisation,
#' never a different model.
#'
#' @section Invariants:
#' These hold for every architecture and are what [tsfm_check_architecture()]
#' asserts:
#'
#' * **Shape.** The result is a numeric matrix, `h` by `length(quantile_levels)`.
#' * **Monotonicity.** Within a row, forecasts are non-decreasing across
#'   quantile levels. A model whose raw output can cross must sort before
#'   returning.
#' * **Finiteness.** No `NA`, `NaN`, or `Inf` for a finite, non-empty context.
#' * **Context limit.** The model never requires more history than the
#'   `max_context` it declares in its [new_tsfm_capabilities()].
#' * **Empty context.** An empty context raises an error rather than returning
#'   `NA` — silent nonsense is worse than a stop.
#' * **Batch agreement.** If `predict_batch_fn` is supplied, it matches
#'   `predict_fn` within numerical tolerance.
#'
#' @section Capabilities:
#' Capabilities are declarations the engine enforces *before* inference, so an
#' unsupported request costs nothing. Declaring a capability the forward pass
#' does not honour is a bug in the architecture. Contract v1 has channels only
#' for univariate numeric context and predictive quantiles: multivariate,
#' covariate, sample-path, and fine-tuning declarations must remain `FALSE`.
#' Native-quantile checkpoints declare their exact trained levels when those
#' levels are fixed, and the engine validates them together with horizon and
#' context limits before calling the architecture. See
#' [new_tsfm_capabilities()].
#'
#' @section Versioning:
#' [new_tsfm_model()] stamps every handle with [tsfm_contract_version()]. The
#' contract follows semantic versioning: the major component changes only for a
#' breaking change to the signatures or invariants above, and never without a
#' deprecation cycle. Architectures may record the version they were written
#' against by passing `contract_version` explicitly.
#'
#' @seealso [tsfm_register_arch()], [new_tsfm_model()],
#'   [tsfm_check_architecture()], [new_tsfm_capabilities()]
#' @name tsfm-architecture-contract
NULL

#' The architecture contract version
#'
#' The version of the contract described in `?`[tsfm-architecture-contract] that
#' this installation of tsfm implements. Stamped onto every [new_tsfm_model()].
#'
#' @return A [package_version].
#' @export
#' @examples
#' tsfm_contract_version()
tsfm_contract_version <- function() {
  package_version("1.0.0")
}
