# tsfm

<!-- badges: start -->
[![R-CMD-check](https://github.com/pedrobtz/tsfm/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/pedrobtz/tsfm/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

**An inference engine for time-series foundation models, natively in R.**

The target is simple: given a supported pretrained checkpoint and a numeric
context, `tsfm` produces predictive quantiles.

R already has every layer a foundation-model forecasting stack needs — Hub
access (`hfhub`), checkpoint format (`safetensors`), tensor runtime (`torch`),
temporal structures (`tsibble`), forecast distributions (`distributional`),
backtesting (`rsample`), reconciliation and metrics (`fabletools`), and conformal
calibration (`conformalForecast`). What it lacks is the **model layer**: a
catalogue of natively implemented time-series foundation models (TSFMs) behind
one capability-aware interface. `tsfm` provides that layer:

- **One loader.** `tsfm_pretrained("...")` resolves a supported model into a
  uniform handle. TimesFM 2.5 is the first supported native architecture; the
  weight-free `stub` remains the fast engine test fixture.
- **One contract.** Every architecture implements
  `predict_fn(context, h, quantile_levels)` and nothing more. See
  `` ?`tsfm-architecture-contract` `` and verify yours with
  `tsfm_check_architecture()`.
- **Capability metadata, checked first.** `tsfm_capabilities()` reports context
  and horizon limits, trained quantile levels, and the weight licence. Contract
  v1 fixes multivariate, covariate, sample, and fine-tuning flags at `FALSE`
  until those inputs and outputs have real execution channels.
- **An offline checkpoint catalogue.** `tsfm_models()` lists only checkpoints
  that passed the support gates; `tsfm_models(state = NULL)` also exposes
  experimental and scaffold records. Listing only probes the local Hub cache
  and never downloads anything.
- **One forecast object** backed by `distributional`.

### Framework-neutral by design

The engine has no opinion about where your context came from or where the
quantiles go. Interfaces to the surrounding ecosystem are **optional adapters of
equal standing**, all in `Suggests`:

| Adapter | Surface | Needs |
|---------|---------|-------|
| plain R | `forecast()` on a `data.frame`, `as.data.frame()` | — |
| tidyverts | `fabletools::as_fable()` → `fable`; `TSFM()` inside `fabletools::model()` | `fabletools`, `tsibble` |
| tidymodels | `tsfm_fit()` / `predict()` | — |
| tidymodels | `tsfm_reg()` parsnip spec, tunable `context_length` | `parsnip`, `dials` |

There are two routes into the tidyverts, and they trade convenience for
throughput. `forecast(model, panel, h) |> fabletools::as_fable()` runs **one
batched call** across every series. `TSFM()` composes inside
`fabletools::model()` next to ARIMA or ETS, but fabletools evaluates one key at
a time, so the engine sees a batch of one per series; the checkpoint is loaded
once and reused from the resident cache. Note that fabletools derives `.mean`
from the distribution, so a `TSFM()` fable reports the distribution's mean while
the other routes report the engine's exact median — `median()` recovers the
point forecast on any route. See `vignette("timesfm-zero-shot")`.

`tsfm` deliberately does **not** define competing `forecast()` or `as_fable()`
generics: it re-exports `generics::forecast` and registers its `as_fable` method
onto `fabletools`, so attaching `tsfm` alongside them never masks the verbs.

## Installation

```r
# After the CRAN release:
install.packages("tsfm")

# Development version:
# install.packages("pak")
pak::pak("pedrobtz/tsfm")
```

The pinned TimesFM checkpoint is about 925 MB. It is downloaded once through
`hfhub`, loaded into native R `torch`, and reused from a bounded in-session
handle cache. Use `tsfm_download()` to prefetch it explicitly.

## Quick start

```r
library(tsfm)

history <- data.frame(
  time = 1:24,
  value = 100 + cumsum(sin((1:24) * pi / 6))
)

# Weight-free random-walk fixture for exercising the engine shell.
model <- tsfm_pretrained("stub")
tsfm_capabilities(model)

# Plain-R engine path.
fc <- forecast(model, history, h = 6, index = "time", target = "value")
as.data.frame(fc)

# Optional tidyverts adapter.
fc |> fabletools::as_fable()

# Or compose the model inside a mable, next to any other tidyverts model.
history_ts <- tsibble::as_tsibble(history, index = time)
fits <- fabletools::model(history_ts, tsfm = TSFM(value, model_id = "stub"))
fabletools::forecast(fits, h = 6)
```

Swap `"stub"` for `"google/timesfm-2.5-200m-pytorch"` and the same code runs the
real checkpoint.

Vignettes: `vignette("timesfm-zero-shot")` walks the whole public surface on a
real checkpoint; `vignette("electricity-demand")` scores it against seasonal
naive, ETS, and ARIMA on held-out data; `vignette("zero-shot-workflow")` and
`vignette("rolling-origin-tuning")` cover the engine and tuning with the stub.

## Developing an architecture

Architecture authors can implement and verify the public execution contract:

```r
my_arch <- function(config, weights) {
  new_tsfm_model(
    architecture = "my-arch",
    config       = config,
    capabilities = new_tsfm_capabilities("my-arch", max_context = 512L),
    predict_fn   = function(context, h, quantile_levels) {
      # ... -> matrix[h, length(quantile_levels)]
    }
  )
}

tsfm_check_architecture(my_arch)          # contract conformance gate
```

The checkpoint catalogue is owned and curated by `tsfm`. Registering a
constructor with `tsfm_register_arch()` changes only the current session's
architecture mapping; it does not add a row to `tsfm_models()` or make an
arbitrary model ID loadable through `tsfm_pretrained()`, `TSFM()`, or parsnip.
Catalogue inclusion requires an immutable checkpoint revision, manifest,
licence metadata, contract conformance, and committed numerical-parity evidence
to be reviewed in this repository. Independently constructed `tsfm_model`
handles can use the plain `forecast()` path without catalogue registration.

## Model catalogue

The programmatic catalogue is the source of truth:

```r
tsfm_models() # safe default: supported checkpoints only
```

| Model | id | Status |
|-------|----|--------|
| Stub | `stub` | executable random-walk test fixture; not a foundation model |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | supported native point and 0.1–0.9 quantile inference |
| TTM (TinyTimeMixer) | `ibm-granite/granite-timeseries-ttm-r2` | registered scaffold; deferred until point-only output is supported |
| Chronos-2 | `amazon/chronos-2` | unregistered until the required native `0.1.0` port passes conformance and parity |

TimesFM is marked supported because it passes both the architecture conformance
gate and every committed numerical-parity fixture against the pinned official
implementation.

The release roadmap adds Toto 2.0 4M and a new native Chronos-2 port before
`0.1.0`; Granite PatchTST-FM and Sundial are the selected later capability
representatives. Planned models do not appear in `tsfm_models()` until their
metadata and executable support gates are complete.

`tsfm_models()` is package-owned: `supported` means the checkpoint has passed
`tsfm`'s release gates, not that another package registered a constructor in the
current session.

#### Horizon and effective context

TimesFM's 16,384 positions cover the context **and** the forecast, and a request
consumes whole 128-value output blocks, so the history actually used is
`16384 - ceiling(h / 128) * 128`:

| Horizon | Observations used |
|---|---|
| 1–128 | 16,256 (the reported `max_context`) |
| 129–256 | 16,128 |
| 1024 (the maximum) | 15,360 |

`max_context` reports the best case. Longer histories are truncated to the most
recent values, per series — a forecast never depends on which other series share
its batch.

### Checkpoint provenance, cache, size, and licence

The TimesFM feasibility work is pinned to official source commit
`3dae50b20d7a724981e8ea36cda75578f80dd2dc` and checkpoint revision
`1d952420fba87f3c6dee4f240de0f1a0fbc790e3`. The catalogue never resolves a
moving `main` branch for curated support metadata.

`cached` means every file in the checkpoint's static manifest is already in
the cache managed by `hfhub`. Probes always set `local_files_only = TRUE`.
`hfhub` uses `HUGGINGFACE_HUB_CACHE` when set and otherwise defaults to
`~/.cache/huggingface/hub`; `tsfm` does not create a second disk cache. A known
incomplete manifest is `FALSE`, while `NA` means the catalogue entry has no
manifest definition yet.

`size_bytes` is a static first-download estimate exposed before loading. The
pinned TimesFM `model.safetensors` is 925,181,104 bytes (about 925 MB, or 882
MiB), excluding small metadata files and live tensor/runtime overhead. The
catalogue's `license` column is the upstream **weight** licence—Apache-2.0 for
the current curated records. The original `tsfm` code is MIT licensed; the
native TimesFM files identified in `inst/COPYRIGHTS` are modified derivative
works under Apache-2.0 and retain Google LLC's copyright notice. The complete
Apache licence is installed with the package. `tsfm` reports upstream weight
terms and never accepts gated-repository terms on a user's behalf.

### Download and constructed-handle lifecycle

Disk files and live R objects are deliberately separate:

```r
tsfm_cache_status()
tsfm_unload() # releases resident handles; does not delete Hub files
```

`tsfm_download(model_id, revision = NULL)` explicitly fetches a curated
checkpoint manifest and returns its local paths invisibly without constructing
a module. `tsfm_pretrained()` rejects non-supported catalogue states before any
download or tensor work.

Constructed handles use an R-session least-recently-used cache keyed by model
ID, immutable revision, resolved device, and every load-affecting option.
`options(tsfm.max_loaded_models = 1L)` is the default; `0L` disables resident
reuse. `reuse = FALSE` constructs a fresh handle without changing the existing
cached handle or deleting downloaded files.

## Known limitations

- **Contract v1 is univariate and quantile-only.** No covariates, multivariate
  targets, sample paths, or fine-tuning. These are declared `FALSE` and rejected
  before inference rather than silently ignored.
- **CPU is the certified baseline.** CUDA and MPS resolve and run, but are not
  numerically certified; the parity fixtures are CPU references. A handle is
  bound to the device it was loaded on and refuses inference on another.
- **`TSFM()` is not batched.** fabletools evaluates one key at a time. Use
  `forecast(model, panel, h) |> fabletools::as_fable()` for throughput.
- **`.mean` differs by route.** fabletools derives it from the distribution, so
  a `TSFM()` fable reports the distribution's mean where the other routes report
  the engine's exact median. `median()` recovers the point forecast anywhere.
- **`hilo()` can return an `NA` bound.** `fabletools::hilo(fc, 80)` computes
  `(1 - 80/100)/2`, which lands two ulps below `0.1`; `distributional`'s
  percentile distribution does not extrapolate past its outermost stored
  percentile and returns `NA`. This is upstream behaviour. Request levels that
  bracket the interval you need, or read quantiles from the forecast directly.
- **Effective context shrinks with the horizon** — see the table above.
- **Reference fixtures are not bit-reproducible across machines.** Regenerating
  under the same pinned environment on different hardware moves expected outputs
  by a couple of float32 ulps, which is why parity uses an `atol`/`rtol` budget.

## Scope

`tsfm` owns the model layer and thin adapters only. Backtesting, metrics,
reconciliation, conformal calibration, tuning, and feature engineering are
delegated to the existing R stack.

Explicitly **out of scope**: hosted-API models (`nixtlar` covers TimeGPT),
training from scratch, and distributed execution. `tsfm` runs open weights
locally; anything served over a network belongs to its own client.

See the [release roadmap](https://github.com/pedrobtz/tsfm/blob/main/.agents/roadmap.md)
and [implementation plan](https://github.com/pedrobtz/tsfm/blob/main/.agents/plan.md)
for the staged plan and full gap analysis.
