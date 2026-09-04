# zukzeit 0.1.0

Development release. The native TimesFM inference baseline and surrounding
model-loading and forecasting shell are implemented. Toto 2.0 4M and a native
Chronos-2 port remain required before `0.1.0` is released.

## Fixes

* The future index now comes back in the observed index's own type. Extending an
  index in numeric space returned bare numbers for every `tsibble` calendar
  type, so a `yearmonth` panel was forecast at `636, 637, 638` instead of
  `2023 Jan` onwards, and `POSIXct` came back as seconds. That silently broke
  the join back to the caller's data, the interval a `tsibble` reports, and any
  `fabletools::accuracy()` call against held-out actuals. Extension is delegated
  to `seq()`, which walks each index type in its own units; an index type that
  cannot be extended is refused rather than downgraded.
* Every requested quantile level now gets its own prediction column. Column
  names rounded the level to a whole percent, so `0.02` and `0.025` both became
  `.pred_q02` and one silently overwrote the other. Whole-percent levels keep
  the familiar `.pred_q10` form; finer levels carry the digits they need
  (`.pred_q02_5`).
* `zuk_reg()` specifications gained the `update()` method every `parsnip`
  specification is expected to provide. Without it `tune::finalize_workflow()`
  fell through to `update.default()` and failed with "need an object with call
  component", so a tuned `context_length` could never be refit --- the whole
  `dials`/`tune` integration was unreachable.
* `predict()` on a fitted model now keys series by label rather than by factor
  level code. A factor `id` column --- anything that has been through
  `droplevels()` or `subset()` --- indexed the stored histories by code, which
  forecast the wrong series or failed several frames down with an error about
  horizons.
* `zuk_check_architecture()` no longer reseeds the caller's session. Its
  default probe context is still deterministic, but the previous RNG state is
  restored on exit.
* The point forecast's quantile level is resolved against the checkpoint's own
  trained levels instead of appending a literal `0.5` to levels the engine had
  already reconciled, which risked carrying two spellings of one trained level
  into the batch boundary. A checkpoint with no trained median now forecasts
  using the requested level nearest it rather than refusing the request.
* `zuk_fit()` no longer molds through `hardhat`. The blueprint it built was
  never forged in `predict()`, so the dependency bought nothing; the fit
  interface now works with no optional packages installed at all.
* Capability flags are validated as logicals. `multivariate = 1` passed the
  reserved-field gate --- `isTRUE(1)` is `FALSE` --- and was silently recorded
  as disabled.
* Panel keys with unused factor levels no longer produce empty series that the
  horizon check then rejects.

## Documentation

* Every exported function now has a runnable example. The `stub` fixture needs
  no network, no checkpoint, and no `torch`, so the engine, forecast, fit,
  device, lifecycle, and tuning surfaces are all demonstrated executably;
  `zuk_download()` is the sole `\dontrun{}`.
* `vignette("zero-shot-workflow")` and `vignette("rolling-origin-tuning")` now
  evaluate their code. They previously showed chunks that referenced objects
  never defined in the vignette, so nothing could have run as written --- which
  is how the missing `update()` method and the `add_formula()` trap below went
  unnoticed.
* `?zuk_reg` documents that a workflow must pass the `index` and `id` columns
  through to the engine. `add_formula()` keeps only the formula's terms and
  drops the series id; `add_variables()` is the right preprocessor.

## Engine and models

* Recorded the Apache-2.0 provenance of the native TimesFM derivative files,
  preserved Google LLC's copyright notice, credited the upstream project, and
  included the complete upstream licence in the installed package.
* Removed the undeclared `harus` startup integration. Consumer packages can
  use the documented catalogue, lifecycle, condition, and `TSFM()` APIs without
  creating a reverse dependency from `zukzeit`.
* Documented the structured condition hierarchy and cache-status schema as
  installed help topics, and made the stub-backed `TSFM()` example executable.
* Safetensors readers now close package-owned file connections deterministically
  after metadata and state-dict reads.

## Engine contract

* `zukzeit` is positioned as a framework-neutral **inference engine**: it owns the
  path from a pretrained checkpoint to predictive quantiles, and interfaces to
  tidymodels, the tidyverts, and modeltime are optional adapters of equal
  standing. `hardhat` moved from `Imports` to `Suggests` accordingly.
* `` ?`zuk-architecture-contract` `` documents the public execution surface, and
  `zuk_contract_version()` versions it. `new_zuk_model()` stamps every handle.
* `zuk_check_architecture()` turns the contract into an executable gate:
  construction, forecast shape, quantile monotonicity, finiteness, context
  limits, empty-context handling, and batch/loop agreement. Built-in
  architectures run it in the test suite; external implementations can use the
  same check without reading the engine's internals. A `max_context` check the probe is too
  short to reach now reports not-applicable instead of passing, so the summary
  never counts an invariant that did not run.
* `zuk_check_architecture()`'s batch-agreement check now asserts
  `|batch - loop| <= tolerance + rtol * |loop|` rather than a bare absolute
  threshold, matching the criterion the parity fixtures already use. A batched
  matrix multiply is not obliged to reduce in the same order as an unbatched
  one, so float32 kernels differ by a few ulps *of the values*: the reference
  PyTorch implementation shows the same 5e-07 relative spread between batch
  sizes on identical input. The old fixed `1e-8` was unreachable for any
  checkpoint forecasting in the hundreds, and had only ever been exercised
  against small-magnitude synthetic weights.
* `forecast()` is now re-exported from `generics` rather than defined locally,
  and `as_fable()` is registered onto `fabletools`'s generic from `.onLoad()`
  instead of shadowing it. Attaching `zukzeit` alongside fabletools, modeltime, or
  forecast no longer risks masking either verb. Call `fabletools::as_fable()`.
* Fixed a latent bug where the parsnip `tunable()` registration called a
  non-existent `rlang::s3_register()`; it is `vctrs::s3_register()`, and the
  failure was silently swallowed by `.onLoad()`'s `tryCatch`, so the engine's
  tuning support never actually registered. Failures now warn.

## Core

* Added `zuk_download()` for explicit manifest prefetch, `zuk_cache_status()`
  for separate disk/resident state, and `zuk_unload()` for resident eviction
  without deleting the `hfhub` cache.
* `zuk_pretrained()` now keys constructed handles by checkpoint, immutable
  revision, resolved device, and load-affecting options in an R-session LRU
  bounded by `options(zuk.max_loaded_models = 1L)`. `reuse = FALSE` bypasses
  reuse and `0L` disables resident storage.
* Added safetensors header validation and named R `torch` state-dict loading,
  with download and checkpoint failures mapped to their structured policy
  families.
* Added the offline `zuk_models()` checkpoint catalogue. Its safe default
  returns only supported checkpoints, while `state = NULL` exposes pinned
  scaffold records and static cost, capability, manifest-cache, provenance,
  and weight-licence metadata without network access.
* The checkpoint catalogue is explicitly package-owned and curated.
  `zuk_register_arch()` registers only an in-session constructor; it does not
  publish model IDs or change what `zuk_models()` reports.
* Added a structured condition hierarchy rooted at `zuk_error`, with one
  recoverable, external, or internal policy parent and structured fields on
  capability, context, quantile, device, download, checkpoint, and contract
  failures.
* Contract v1 now records explicit trained quantile levels and a maximum
  horizon, rejects unsupported probabilities before inference, and refuses
  capability declarations for multivariate/covariate/sample/fine-tuning
  channels that do not yet exist.
* The batch boundary validates contexts, horizons, quantile levels, batch size,
  and architecture return shape/finiteness/monotonicity before assembling a
  forecast.
* `zuk_pretrained()` loads the self-contained `stub` fixture and pinned native
  TimesFM checkpoints through the constructor registered for each curated
  catalogue entry.
* `zuk_capabilities()` reports per-model capability metadata, with pre-flight
  validation that rejects unsupported requests before inference.
* `forecast()` forecasts a `tsibble` or data-frame panel with no optional
  dependencies; `zuk_fit()` / `predict()` provide a hardhat-based,
  tidymodels-conformant interface when `hardhat` is installed.
* `zuk_forecast` objects are backed by `distributional`, with a
  `fabletools::as_fable()` method and tidy prediction adapters.
* Added `TSFM()`, a `fabletools` model definition, so a checkpoint composes
  inside `fabletools::model()` next to any other tidyverts model. It evaluates
  one key at a time and reuses the resident handle across keys;
  `forecast(model, panel, h) |> fabletools::as_fable()` remains the batched
  route. `fabletools` derives `.mean` from the distribution, so a `TSFM()` fable
  reports the distribution's mean where the other routes report the engine's
  exact median; the distributions themselves are identical, and `median()`
  recovers the point forecast on any route.
* Added `vignette("electricity-demand")`, a worked comparison on Victorian daily
  electricity demand from the Monash archive: `TSFM()` in one mable beside
  `SNAIVE()`, `ETS()`, and `ARIMA()`, scored on 28 held-out days with
  `fabletools::accuracy()`. Zero-shot TimesFM reaches MASE 0.47 against 1.20 for
  the fitted ARIMA. The vignette cites the benchmark literature on both sides,
  including where foundation models lose to statistical baselines, and is
  explicit that one window is one observation.
* `fabletools::as_fable()` no longer drops `.mean`. It does not synthesise the
  column the way `fabletools::forecast()` does, so the converted fable
  previously had a silently `NULL` point forecast.
* Batched panel inference (`zuk_run_batches()`) with device resolution and
  validation across CPU/CUDA/MPS (`zuk_resolve_device()`, `zuk_set_device()`).

## tidymodels

* `zuk_reg()` parsnip specification and `"zukzeit"` engine, with an exported fit
  bridge exercised end to end against the stub.
* `context_length()` is a `dials` parameter and now actually bounds the history
  retained by the fitted engine.
* `zuk_reg()` and `context_length()` check for `parsnip` and `dials` up front,
  so a core installation gets the same typed `zuk_error_capability` as every
  other optional-dependency path instead of a bare "no package called" error.

## Models

* **TimesFM 2.5** is the first supported native model. The R `torch` port owns
  patch tokenization, cumulative reversible normalization, 20 causal rotary
  transformer layers, cached autoregressive decoding, the continuous quantile
  head, flip invariance, positivity inference, and quantile-crossing repair.
* The complete 232-tensor safetensors state dict maps directly onto meta-device
  module placeholders, avoiding a retained second CPU copy during construction.
  Modules run in evaluation mode under disabled gradients.
* Single-series and vectorized batch inference pass every pinned CPU golden
  fixture, including exact short-context, 16,256-value truncation, and
  batch/loop cases. Repeated calls are identical and silent.
* Numerical comparisons use `|actual - expected| <= atol + rtol * |expected|`,
  the criterion `torch.testing.assert_close()` and the upstream TimesFM tests
  use, instead of a single absolute threshold that is simultaneously too tight
  for large values and too loose for small ones. Block-level checks use
  PyTorch's float32 defaults (`atol = 1e-5, rtol = 1.3e-6`); model fixtures
  record `atol = 1e-4, rtol = 1e-5`.
* Inference places explicit interrupt boundaries between engine batches and
  autoregressive blocks, where R delivers a pending interrupt; interrupt
  conditions propagate without being rewrapped and without returning a partial
  result. A single `torch` call remains uninterruptible.
* Requested quantile levels are reconciled to the checkpoint's trained levels at
  the engine boundary and handed to the architecture in the checkpoint's own
  spelling. `seq(0.1, 0.9, by = 0.1)` and the literals parsed from `config.json`
  differ at `0.3` and `0.7`; previously, requesting the levels advertised by
  `zuk_models()` produced `NA` forecasts that surfaced as an internal contract
  error. An unsupported level is now a `zuk_error_quantile_levels` refusal
  raised before any tensor work.
* TimesFM truncates context per series using that series' own horizon, so a
  forecast no longer depends on which other series share its batch or on
  `options(zuk.batch_size)`. The usable history is
  `context_length - ceiling(h / 128) * 128`; `max_context` reports the h ≤ 128
  best case, and the README tabulates the rest.

* Added five compact, committed TimesFM reference fixtures from the pinned
  official implementation: typical, short-context, context-truncation,
  mixed-sign, and two-series batch/loop cases. The locked generator records
  every forecast flag and upstream dependency pin.
* The mixed-sign fixture closes a real coverage gap. The official decoder clamps
  forecasts at zero only when every observed value is non-negative, and every
  other committed context is strictly positive, so the unclamped branch had no
  reference behaviour attached to it. Its forecast is entirely negative.
* Regenerating the fixtures reproduces the `.f32` inputs byte-identically but
  not the expected outputs: the same pinned environment on different hardware
  moves them by up to 2 float32 ulps. The reference implementation is not
  bit-reproducible across CPU architectures, which is why parity is asserted
  against an `atol`/`rtol` budget. Do not regenerate existing fixtures casually.
* Captured and validated the exact pinned TimesFM state layout: 232 float32
  tensors and 231,289,280 parameters. The full 925,181,104-byte checkpoint
  loads into a named R state dict and native module.
* The native TimesFM feasibility gate passed: a reduced R `torch` transformer
  block and continuous quantile head match the pinned official PyTorch output
  below `1e-5` — a threshold set by float32 accumulation across LibTorch builds,
  not by the port; the exact-width CPU spike executes successfully. This does not
  yet constitute checkpoint support.
* **Toto 2.0 4M** has a native R `torch` port: unit-scaled projections, xPos
  rotary attention over the time axis alternating with full attention over the
  variate axis, a gated feed-forward path, tau-rule residuals, a float64 causal
  patch scaler composed with `asinh`, and single-pass quantile decoding. The
  full network and end-to-end forecasts match the pinned reference on the real
  checkpoint to within 9e-07 relative. It is catalogued as `experimental`, not
  `supported`, until committed golden fixtures back that claim offline.
* **Stub** remains an executable random-walk test fixture, not a foundation
  model.
* **TTM** remains a registered scaffold deferred until the engine represents
  point-only output.
* The old **Chronos-2** Brulee adapter is no longer registered or advertised as
  executable. It is kept as prior art under `.agents/reference/` rather than
  shipped in `R/`, and `brulee` has left `Suggests`: the package no longer
  declares a dependency for code nothing calls. `zuk_pretrained()` continues
  to reject Chronos-2 ids before any network or tensor work until the required
  native port passes its `0.1.0` support gates.
