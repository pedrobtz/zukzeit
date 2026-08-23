# `tsfm` consumer API

This document defines the public engine surface that orchestration packages
need from `tsfm`. The motivating consumer is **`tsai`**, the R implementation
of Python's TimeCopilot. At the package boundary, `tsai` is downstream of
`tsfm`: it may inspect and call this API, but `tsfm` must never depend on it.

The same contract supports benchmark runners, unattended jobs, and model
selection systems. It is therefore ordinary programmatic API, not an
agent-specific layer.

*Last reviewed: 2026-08-23.*

---

## Requirements for `0.1.0`

| # | Requirement | Roadmap stage |
|---|---|---|
| R1 | Offline checkpoint catalogue | Stage 1 |
| R2 | Structured error hierarchy | Stage 1 |
| R3 | Explicit download and bounded handle reuse | Stage 2 |
| R4 | Deterministic, quiet, interruptible inference | Stage 3 |
| R5 | `TSFM()` fabletools model definition | Stage 4 |

All five requirements are release gates. They let a consumer discover a safe
model before loading it, choose a response to failure without parsing messages,
control expensive downloads and model construction, and compose the engine
with native R forecasting workflows.

The final `0.1.0` supported catalogue must contain the pinned TimesFM 2.5,
Toto 2.0 4M, and Chronos-2 checkpoints. Stage 6 will amend this contract with
the exact backward-compatible structured-input API for Chronos-2 multivariate
targets and covariates before that model is implemented. Until then, the
contract-v1 univariate surface below remains the only executable inference API.

## R1 — Offline checkpoint catalogue

```r
tsfm_models(state = "supported")
tsfm_models(state = NULL) # include experimental and scaffold entries
```

`tsfm_models()` returns one row per checkpoint, not one row per architecture.
Its default is safe for user-facing selection: only checkpoints that have
passed conformance and numerical-parity gates are returned. It performs no
network requests and does not construct a model.

The catalogue is package-owned and release-curated. Runtime calls to
`tsfm_register_arch()` do not add rows or change support states. Consequently,
`state = "supported"` always means certified by `tsfm`, and `tsai` discovery is
independent of which extension packages happen to be installed or loaded.

Required columns are:

| Column | Contract |
|---|---|
| `model_id`, `architecture`, `revision` | Checkpoint identity; supported revisions are immutable commit SHAs |
| `state` | One of `supported`, `experimental`, or `scaffold` |
| `max_context`, `quantile_levels` | Limits known before inference; quantiles use a list-column |
| `multivariate`, `past_covariates`, `future_covariates` | Executable capability flags, never roadmap intent |
| `n_params`, `size_bytes` | Static cost estimates for selection and download warnings |
| `cached` | Whether every required checkpoint file is present locally |
| `license` | Upstream weight licence identifier |

Each registered checkpoint carries a static required-file manifest.
`cached` is computed by probing those files through `hfhub` with
`local_files_only = TRUE`. A known incomplete manifest returns `FALSE`; `NA`
is reserved for entries whose manifest is not yet defined. Catalogue discovery
must never turn an unknown cache state into a network request.

## R2 — Structured error hierarchy

Every public engine failure inherits from `tsfm_error` and one policy class:

```text
tsfm_error
├── tsfm_error_recoverable
│   ├── tsfm_error_capability
│   ├── tsfm_error_context_length
│   ├── tsfm_error_quantile_levels
│   └── tsfm_error_device
├── tsfm_error_external
│   └── tsfm_error_download
└── tsfm_error_internal
    ├── tsfm_error_checkpoint
    └── tsfm_error_contract
```

- **Recoverable** means another model, device, or valid request may succeed.
- **External** means the environment failed, for example network access,
  authentication, or a damaged download; callers may retry the operation later.
- **Internal** means a checkpoint is incompatible or an architecture violated
  its contract; callers should stop and report the defect.

Leaf conditions include structured fields appropriate to the failure, such as
`model_id`, `revision`, `capability`, requested and supported limits, or
requested and resolved devices. Consumers branch on `inherits()` and fields,
never on translated message text.

## R3 — Download and model-handle lifecycle

Disk caching and live model reuse are separate concepts:

```r
tsfm_download(model_id, revision = NULL, progress = interactive())
tsfm_pretrained(model_id, revision = NULL, device = NULL, reuse = TRUE, ...)
tsfm_cache_status()
tsfm_unload(model_id = NULL, revision = NULL, device = NULL)
```

- `tsfm_download()` resolves a catalogue entry, reports its expected size,
  fetches its manifest, and returns the local paths invisibly. It does not
  construct a `torch` module.
- `tsfm_pretrained()` resolves the revision and device, then reuses or creates
  a model handle. `reuse = FALSE` bypasses resident reuse without deleting
  downloaded files.
- `tsfm_cache_status()` returns checkpoint identity, resolved device, disk
  availability, resident state, estimated bytes, and last-use metadata.
- `tsfm_unload()` evicts matching resident handles only. `hfhub` continues to
  own its on-disk cache; `tsfm` does not expose disk deletion in `0.1.0`.

Resident handles use an R-session least-recently-used cache. The default is
`options(tsfm.max_loaded_models = 1L)`; `0L` disables reuse. A key includes the
model ID, resolved revision SHA, resolved device, and every option that affects
module construction. Eviction must release the package's strong reference to
the handle.

This prevents `parsnip::fit()` resamples and fabletools per-key fits from
rebuilding the same 200M-parameter module while bounding process memory.

## R4 — Safe inference inside another loop

Inference has three behavioural guarantees:

1. **Deterministic:** identical context, checkpoint, arguments, and device
   produce identical output. Future sample-based APIs must accept an explicit
   seed rather than silently using global RNG state.
2. **Quiet by default:** normal inference emits no `cat()`, `print()`, progress,
   or informational messages. Explicit progress uses a suppressible mechanism.
3. **Interruptible:** batch execution checks for interrupts between batches,
   runs cleanup on exit, and never returns a partial `tsfm_forecast`.

Modules run in evaluation mode with gradients disabled. An interrupt is not
wrapped as a checkpoint or contract error.

## R5 — `TSFM()` for fabletools composition

`as_fable()` converts an already-computed forecast. `TSFM()` supplies the
separate ability to participate in `fabletools::model()`:

```r
fits <- fabletools::model(
  data,
  arima = fable::ARIMA(value),
  timesfm = tsfm::TSFM(
    value,
    model_id = "google/timesfm-2.5-200m-pytorch",
    revision = "<pinned-commit-sha>"
  )
)
fabletools::forecast(fits, h = 12)
```

The public constructor is:

```r
TSFM(formula, model_id, revision = NULL,
     quantile_levels = NULL, device = NULL, ...)
```

It is implemented in `tsfm` with `new_model_class()` and
`new_model_definition()`. Zero-shot training stores the series context and
resolved model metadata; `forecast.model_tsfm()` obtains the shared model
handle and returns a `distributional` vector. `fabletools`, `tsibble`, and
related packages remain optional `Suggests` dependencies and are checked only
when this adapter is called.

Fabletools fits and forecasts one mable row per key, so this path cannot use
the engine's panel batch function. Both paths remain public and documented:

| Path | Panel batching | fable model/CV/ensemble composition |
|---|---|---|
| `forecast(model, panel, h)` then `as_fable()` | Yes | No |
| `model(panel, TSFM(value, ...))` | No | Yes |

The resident-handle cache prevents the composable path from loading one module
per key. Callers that prioritize throughput should use the first path.

## Consumer boundary

`tsfm` owns checkpoint metadata, loading, execution, forecast distributions,
and framework adapters. A consumer such as `tsai` owns model selection,
fallback policy, backtesting, metrics, ensembles, prompts, and tool schemas.
No LLM, agent, or `tsai` dependency belongs in `tsfm`.

In `0.1.0`, `tsai` selects only from the package-owned catalogue. External
architecture constructors and manually created handles are not injected into
consumer discovery. Adding third-party checkpoint discovery would require a
future consumer-contract revision with explicit provenance and trust semantics.
