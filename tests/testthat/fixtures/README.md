# Golden parity fixtures

Each native architecture ships a small set of **golden fixtures**: fixed inputs
paired with the reference implementation's outputs. The parity tests
(`test-parity-*.R`) replay them and assert the R port matches within float
tolerance, so CI verifies correctness without needing Python, torch weights, or
Hub access at run time.

## Tolerance

Comparisons use `|actual - expected| <= atol + rtol * |expected|` — the criterion
`torch.testing.assert_close()` and the upstream TimesFM tests use — through
`expect_close_f32()` in `tests/testthat/helper-close.R`. Each fixture records the
`atol`/`rtol` pair it was generated against.

A single absolute threshold is the wrong shape for this comparison: it is too
tight for large values and too loose for small ones. The TimesFM forecasts sit
near 116, where a bare `1e-4` is `8.6e-7` relative — tighter than PyTorch's own
float32 default of `1.3e-6`, and only 3.3x above the observed spread.

Tolerance is set by float32 accumulation, not by correctness. Reassociation
across LibTorch builds and BLAS backends moves results by single-digit to
low-tens of ulps (one ulp is about `1.2e-7` relative); a structural error moves
them by `1e-1` or more. Block-level checks use PyTorch's float32 defaults
(`atol=1e-5, rtol=1.3e-6`); end-to-end model fixtures use `atol=1e-4, rtol=1e-5`.

Bit-exactness across builds is not a reachable goal, and the tests do not chase
it. For single-threaded reduction on one host, export `OMP_NUM_THREADS=1` before
starting R: LibTorch's native parallel backend will not change intraop threads
once parallel work has begun, so calling `torch_set_num_threads()` from inside a
test is a one-way door that warns when restored.

## Layout

```
fixtures/
└── timesfm/
    ├── typical.json + typical-context-1.f32
    ├── short_context.json + short_context-context-1.f32
    ├── context_truncation.json + context_truncation-context-1.f32
    ├── mixed_sign.json + mixed_sign-context-1.f32
    └── batch_agreement.json + two batch_agreement-context-*.f32 files
```

`mixed_sign` exists for one reason: the official decoder clamps forecasts at
zero only when every observed value is non-negative, and every other context
here is strictly positive. Without it the unclamped branch has no reference
behaviour attached to it. Its forecast is entirely negative.

## Fixture schema (`*.json`)

```json
{
  "schema_version": 1,
  "name": "typical",
  "source_commit": "<pinned source sha>",
  "model_id": "google/timesfm-2.5-200m-pytorch",
  "revision": "<pinned checkpoint sha>",
  "reference": {"torch": "2.2.2", "...": "forecast flags"},
  "context_specs": [{"kind": "trend_seasonal", "length": 96}],
  "context_files": ["typical-context-1.f32"],
  "max_context": 96,
  "horizon": 12,
  "quantile_levels": [0.1, 0.2, "...", 0.9],
  "expected_point": [[/* series by horizon */]],
  "expected_quantiles": [[[/* series by horizon by quantile */]]],
  "atol": 1e-4,
  "rtol": 1e-5
}
```

The little-endian float32 context files preserve NumPy's exact generated input,
including intermediate float32 rounding, while keeping the long-context case
compact. The JSON descriptors remain human-readable regeneration instructions.

## Regenerating the TimesFM fixtures

The generator is `.agents/generate-timesfm-reference.py`; its Python packages
are locked in `.agents/timesfm-reference-requirements.txt`. It refuses to run
unless `TIMESFM_SOURCE` is exactly the recorded source commit.

The locked environment needs Python 3.11: `torch==2.2.2` publishes no wheels for
newer interpreters, so a current system Python cannot build it.

```sh
uv venv --python 3.11 .venv-timesfm-reference
uv pip install --python .venv-timesfm-reference/bin/python \
  -r .agents/timesfm-reference-requirements.txt
TIMESFM_SOURCE=/path/to/pinned/timesfm \
TIMESFM_CHECKPOINT=/path/to/model.safetensors \
  .venv-timesfm-reference/bin/python \
  .agents/generate-timesfm-reference.py \
  tests/testthat/fixtures/timesfm
```

The upstream checkout needs `GIT_LFS_SKIP_SMUDGE=1`, or the LFS filters disabled,
if `git-lfs` is not installed; the Python sources themselves are not LFS objects.

### Regeneration is not bit-reproducible

The `.f32` inputs regenerate byte-identically, but the expected outputs do not.
Rerunning the pinned environment on a different machine on 2026-08-14 moved them
by up to 2 float32 ulps (`1.53e-05` absolute, `1.36e-07` relative) — the
reference implementation is not bit-reproducible across CPU architectures. This
is exactly why parity is asserted against an `atol`/`rtol` budget.

The practical consequence: **do not regenerate existing fixtures casually.** A
rerun on a new host rewrites every expected value by a few ulps, producing a
large diff that carries no information. Regenerate when the pinned source or
checkpoint changes; to add a case, generate into a scratch directory and copy in
only the new files.

The two-series fixture also evaluates each context separately and records the
maximum batch/loop difference. Fixture generation is outside CI. Replay is
Python-free and explicitly enabled with `ZUK_RUN_CHECKPOINT_TEST=true`; the
pinned checkpoint must already be available through the Hub cache.
