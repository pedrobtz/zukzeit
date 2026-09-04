# `zukzeit` engine plan

`zukzeit` provides a native R engine for loading and running open time-series
foundation models. This document fixes the package architecture and public
contract. [`roadmap.md`](./roadmap.md) owns release order and gates;
[`consumer-api.md`](./consumer-api.md) specifies the surface downstream
packages rely on.

*Last reviewed: 2026-08-23.*

---

## Mission

Given a supported checkpoint and numeric history, `zukzeit` downloads immutable
weights, constructs a native R `torch` module, and returns trustworthy point
and quantile forecasts through a stable, framework-neutral interface.

The package fills the model-runtime gap in R. Existing packages already provide
temporal data structures, resampling, metrics, reconciliation, and deployment;
`zukzeit` should integrate with them instead of reproducing them.

The first release proves a small, complementary portfolio: TimesFM 2.5 as the
general quantile baseline, Toto 2.0 4M as the efficient model, and Chronos-2 as
the multivariate and covariate-aware model. A model is not supported merely
because it is registered. It must pass the architecture contract and numerical
parity against a pinned upstream implementation.

## Design principles

1. **Engine first.** Loading, validation, batching, device placement, and
   inference do not depend on tidymodels, tidyverts, or another model wrapper.
2. **Native execution.** Supported architectures execute in R `torch` without
   Python or Brulee at runtime. Brulee remains useful prior art, not a backend.
3. **Truthful metadata.** Capability and catalogue entries describe executable
   behaviour at immutable checkpoint revisions.
4. **Quantiles first.** Contract v1 returns an `h` by `q` quantile matrix.
   A backward-compatible contract v2 adds grouped targets and past/future
   covariates for Chronos-2. Point-only and sample-path outputs remain later
   contract work.
5. **Adapters have equal standing.** Plain R, fabletools, and parsnip adapt the
   same engine objects. Optional ecosystems stay in `Suggests`.
6. **Consumer-safe operation.** Discovery is offline, failures are classed,
   expensive lifecycle steps are explicit, reuse is bounded, and inference is
   deterministic, quiet, and interruptible.
7. **Curated discovery.** The package owns the checkpoint catalogue. Runtime
   constructor registration never changes what `zuk_models()` reports, so
   downstream selection remains reproducible and `supported` retains one
   release-controlled meaning.

## Public contract

### Discovery and lifecycle

```r
zuk_models(state = "supported")
zuk_download(model_id, revision = NULL, progress = interactive())
zuk_pretrained(model_id, revision = NULL, device = NULL, reuse = TRUE, ...)
zuk_cache_status()
zuk_unload(model_id = NULL, revision = NULL, device = NULL)
```

`zuk_models()` is a static checkpoint catalogue and never loads a config,
weights, or model. Supported rows include immutable revision, architecture,
capabilities, context and quantile limits, parameter and download estimates,
offline cache status, and weight licence.

The catalogue is package-owned. `zuk_register_arch()` changes only the
constructor mapping used after a curated entry resolves; it does not add model
IDs or support claims. This keeps `tsai` discovery stable across installed and
loaded packages.

`hfhub` owns downloaded files. `zukzeit` owns a bounded in-process LRU of
constructed handles, configured by `options(zuk.max_loaded_models = 1L)`.
Disk presence and resident state are reported separately.

### Model and execution contract

```text
predict_fn(context, h, quantile_levels) -> matrix[h, q]

predict_batch_fn(contexts, horizons, quantile_levels, device)
  -> list(matrix[h, q])
```

A `zuk_model` records checkpoint identity, resolved revision and device,
capabilities, preprocessing configuration, and callable single/batch inference
functions. Validation rejects unsupported requests before download or tensor
work where possible. Modules run in evaluation mode with gradients disabled.

Stage 6 adds a contract-v2 structured input for grouped targets and aligned
past/future covariates. Its exact R shape must be frozen in this plan,
`consumer-api.md`, and the public architecture documentation before the
Chronos-2 implementation begins. The extension must leave contract-v1 model
handles and univariate `forecast()` calls source-compatible.

Every engine error inherits from `zuk_error` and exactly one policy family:
`zuk_error_recoverable`, `zuk_error_external`, or `zuk_error_internal`.
Leaf classes and structured fields are defined in `consumer-api.md`; consumers
never need to inspect error messages.

### Forecast and adapter contract

The core returns `zuk_forecast`, preserving key, index, target, horizon,
quantile, model ID, and revision metadata. Required `0.1.0` surfaces are:

- `forecast(model, data, h)` for single-series and batched panel execution;
- `as.data.frame()` and `fabletools::as_fable()` for result consumption;
- `zuk_reg()` plus working parsnip `fit()` and `predict()` methods;
- `TSFM(formula, ...)` for composition inside `fabletools::model()`.

The fable model-definition path fits per key and therefore does not batch a
panel. It reuses the shared resident handle. Users who prioritize throughput
use batched `forecast()` followed by `as_fable()`; users who prioritize fable
cross-validation, ensembles, or reconciliation use `TSFM()`.

## Engine organization

```text
checkpoint catalogue + classed conditions
                    |
Hub manifest -> safetensors -> architecture registry -> resident-handle LRU
                    |
validation -> preprocessing -> batching/device -> native forward pass
                    |
             zuk_forecast
          /         |          \
     plain R     fabletools    parsnip
```

- The **catalogue** stores checkpoint-level static metadata and required-file
  manifests. Architecture registration remains separate because several
  checkpoints may share one constructor and weight map.
- The **loader** resolves immutable revisions, downloads manifest files, maps
  tensors into a registered module, and attaches provenance.
- The **execution layer** validates the request, normalizes panel data, batches
  by compatible shapes, resolves device/dtype, and restores stable row order.
- Each **architecture port** owns configuration interpretation,
  architecture-specific preprocessing, module construction, weight mapping,
  and forward-output conversion.
- **Adapters** translate public ecosystem objects without changing engine
  semantics or becoming hard runtime dependencies.

Shared preprocessing abstractions are extracted only after two native models
demonstrate real duplication. Prematurely forcing unrelated architectures into
one declarative pipeline would make parity harder to diagnose.

## Downstream `tsai` contract

`tsai`, the R implementation of TimeCopilot, is an explicit downstream
consumer and a design stress test. It must be able to:

1. enumerate supported checkpoints and their cost without network access;
2. choose only models whose declared capabilities satisfy a request;
3. prefetch weights before an unattended run;
4. distinguish fallback, retry-later, and bug conditions by class;
5. reuse an already-constructed model across resamples and fable keys; and
6. compose `TSFM()` with statistical models in a fable workflow.

`zukzeit` does not select a model or implement fallback policy. It exposes the
metadata and typed outcomes required for `tsai` to make those decisions. It
contains no prompts, LLM tools, agent logic, or dependency on `tsai`.

## Model strategy

The maintained catalogue is curated around capability coverage, licence,
reference reproducibility, and native R feasibility.

| Priority | Family | Role |
|---|---|---|
| P0 | TimesFM 2.5 | `0.1.0` general quantile baseline; implemented and certified |
| P0 | Toto 2.0 4M | `0.1.0` efficient probabilistic model |
| P0 | Chronos-2 | `0.1.0` multivariate and covariate-aware model through contract v2 |
| P1 | Granite PatchTST-FM R1 | Later reconstruction and missing-value-imputation model |
| P1 | Sundial / Timer 3.0 | Later generative model after sample-path output support |

Hosted-only services and unpinned community checkpoints are outside the engine
catalogue. Detailed ordering and blockers live in `roadmap.md`.

## Verification strategy

Every native architecture has two independent support gates:

- **Contract conformance:** shapes, finite values, monotone quantiles, context
  limits, metadata, and batch/loop agreement.
- **Numerical parity:** committed fixtures generated from immutable upstream
  source and checkpoint revisions, compared on CPU within documented tolerance.

Core tests remain deterministic and network-safe. Hub and full-checkpoint smoke
tests are explicit opt-ins. Adapter tests use the stub for speed, then add an
opt-in path for every supported checkpoint. Consumer-contract tests cover
offline catalogue discovery, error inheritance, LRU eviction and device keys,
silent repeated inference, interrupt cleanup, and
`fabletools::model(TSFM(...))`.

## Deliberate non-goals for `0.1.0`

- Python or Brulee runtime fallback.
- Point-only forecasts, sample paths, reconstruction/imputation, or
  fine-tuning.
- Granite TTM, Granite PatchTST-FM, Sundial, or any model beyond the three
  release checkpoints.
- New backtesting, selection, metric, ensemble, reconciliation, or conformal
  frameworks.
- Hard dependencies on fabletools, parsnip, modeltime, or `tsai`.
- CRAN acceptance as a condition for the first GitHub tag; CRAN-quality checks
  are still required.
