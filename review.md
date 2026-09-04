# `zukzeit` implementation review — Stages 0–3

*Reviewed 2026-08-13 against the working tree at commit `1200b0e` plus the
uncommitted Stage 3 delta. **All findings below were fixed the same day**; see
[Resolution](#resolution) for what changed and how each fix was verified.*

> This is a historical implementation review, not the current release-status
> report. See [the roadmap](.agents/roadmap.md) and [CRAN comments](cran-comments.md)
> for the current check and CI state.

## Scope and method

This is a static review of every stage the roadmap describes as implemented:
Stage 0 (honest baseline), Stage 1 (feasibility, contract, catalogue), Stage 2
(reference, loader, lifecycle), and Stage 3 (native TimesFM inference). It
covers all 23 files in `R/`, the 22 test files, the catalogue and fixture data,
and the claims made in `README.md`, `NEWS.md`, and `.agents/roadmap.md`.

At review time the test suite could not be executed: `torch`, `hfhub`,
`safetensors`, and `distributional` were not installed in any R library on this
machine, and `testthat::test_local()` failed in `load_imports()` before running a
single test. The findings below were therefore derived from reading the source
plus targeted R snippets exercising the pure-R logic in isolation.

The dependencies were installed during the fix pass that followed, and every
claim in this document has since been checked against a running suite. Two
findings were confirmed by execution rather than reading, and two further defects
surfaced that static review had missed — both recorded under
[Resolution](#resolution).

## Overall assessment

The engine design is genuinely good, and unusually disciplined for a package at
this stage. Three things stand out as better than typical:

1. **The contract is real, not aspirational.** `?zuk-architecture-contract` is
   prose, `zuk_check_architecture()` is the executable form of the same
   document, and `new_zuk_capabilities()` *refuses* to construct a capability
   record that declares a channel contract v1 cannot carry
   ([R/capabilities.R:101-113](R/capabilities.R#L101-L113)). Capability metadata
   cannot drift ahead of the implementation because the constructor rejects the
   drift.
2. **The error taxonomy is consistently applied.** Every abort in the package
   goes through `zuk_abort()`, which `stopifnot()`s exactly one policy parent
   ([R/conditions.R:32-47](R/conditions.R#L32-L47)). I found no bare `stop()` on
   a user-facing path in `R/`.
3. **The engine boundary holds.** `zuk_run_batches()` is a genuine single choke
   point: truncation, horizon validation, quantile validation, batching, device
   resolution, and return-shape validation all happen there, so architectures
   stay small. `R/timesfm-*.R` contains no adapter or framework awareness at all.

The Stage 3 port itself is careful work. Meta-device construction so the module
adopts safetensors storage without a second CPU copy
([R/timesfm-module.R:3-5](R/timesfm-module.R#L3-L5),
[R/arch-timesfm.R:122-124](R/arch-timesfm.R#L122-L124)) is the right call for a
925 MB checkpoint, and the identity weight map means a checkpoint-vs-module
mismatch is a diagnosable error rather than a silent misload.

The findings below are mostly about the seam between the *declared* contract and
the *executed* one — which is exactly where this design concentrates its risk.

---

## Findings

### 1. Requesting all nine trained quantile levels produces NA forecasts — HIGH

**Confirmed by simulation.** The package advertises one spelling of the trained
quantile levels and matches against a different, bit-incompatible one.

- The catalogue declares `quantile_levels = seq(0.1, 0.9, by = 0.1)`
  ([R/catalogue.R:15](R/catalogue.R#L15)).
- The checkpoint's `config.json` declares `[0.1, 0.2, …, 0.9]`, which `jsonlite`
  parses to the literal doubles (`.agents/timesfm-config.json`).
- These are not the same doubles. `seq()` accumulates: its third and seventh
  elements are `0.30000000000000004` and `0.7000000000000001`.

Validation is tolerance-based and passes
([R/capabilities.R:279-283](R/capabilities.R#L279-L283)), but the forward pass
then matches **exactly**:

```r
trained  <- as.numeric(config$quantiles)                     # literals from config.json
channels <- match(as.numeric(quantile_levels), trained) + 1L # seq() values from the catalogue
```

[R/timesfm-inference.R:208-209](R/timesfm-inference.R#L208-L209)

Reproduced in plain R:

```
channels: 2 3 NA 5 6 7 NA 9 10
dim: 6 9   n_nonfinite: 12
```

The `NA` channel indices propagate into the returned matrix as `NA` values,
which `validate_quantile_matrix()` catches
([R/batching.R:193-203](R/batching.R#L193-L203)) and reports as
`zuk_error_contract` — a `zuk_error_internal`
([R/conditions.R:169-189](R/conditions.R#L169-L189)), i.e. "this is a bug in
zukzeit, file a report". So a valid request, built from the package's own
discovery API, fails as an internal engine bug.

This is squarely on the documented consumer path: `consumer-api.md` R1 lists
`quantile_levels` as a discovery column, so the natural `tsai` idiom —
read the levels from `zuk_models()`, pass them to `forecast()` — is the failing
case. Typing `seq(0.1, 0.9, by = 0.1)` by hand fails identically.

Why no test catches it: every real-checkpoint test uses `c(0.1, 0.5, 0.9)`
([test-parity-timesfm.R:63-73](tests/testthat/test-parity-timesfm.R#L63-L73)),
which happens to be the three levels where both spellings agree; and the parity
fixtures read their levels from JSON
([test-parity-timesfm.R:31](tests/testthat/test-parity-timesfm.R#L31)), so they
match `config.json` bit-for-bit. `zuk_check_architecture()` also defaults to
`c(0.1, 0.5, 0.9)` ([R/conformance.R:97](R/conformance.R#L97)). The codebase
already mixes the two spellings without noticing — `test-timesfm-reference.R:31`
compares them with tolerance-based `expect_equal()` and passes.

**Fix.** Have `check_quantile_levels()` return the *supported* values it matched
rather than the user's, so everything downstream is canonical — it already
computes the tolerance match, it just discards the result
([R/capabilities.R:279-299](R/capabilities.R#L279-L299)). Independently, make
`timesfm_predict_batch()` match within tolerance and abort
`zuk_error_quantile_levels` on an unmatched level instead of emitting `NA`.
Add a parity or conformance case using all nine levels.

### 2. Effective context length shrinks with the horizon, silently — MEDIUM

`timesfm_capabilities()` declares a flat `max_context` of
`context_length - horizon_length` = 16256
([R/arch-timesfm.R:71-76](R/arch-timesfm.R#L71-L76)), and `zuk_run_batches()`
truncates to exactly that. The architecture then truncates a *second* time,
using a horizon-dependent budget:

```r
rounded_horizon <- timesfm_round_up(max(horizons), point_horizon)
usable_context  <- limit - rounded_horizon
contexts <- lapply(contexts, function(x) utils::tail(as.numeric(x), usable_context))
```

[R/timesfm-inference.R:171-173](R/timesfm-inference.R#L171-L173)

At `h = 1024` the real budget is 15360, so up to 896 observations the model
declared it would use are dropped with no message and no error. The contract
states that capabilities describe executable behaviour; here they describe the
best case only. Either surface the horizon-dependent limit or document it on
`max_context`.

### 3. Forecasts depend on batch composition — MEDIUM

Same line as above: `rounded_horizon` uses `max(horizons)` across the batch, and
the resulting truncation is applied to **every** context in it. A series with
16,000 observations batched alongside a `h = 1024` request gets a shorter
context — and therefore a different forecast — than the same series forecast
alone or in a batch of short horizons.

Because `zuk_run_batches()` chunks by `batch_size`
([R/batching.R:283](R/batching.R#L283)), the grouping itself depends on
`options(zuk.batch_size)`, so the same panel can produce different numbers at
different batch sizes. That contradicts the contract's guarantee that "the batch
path is an optimisation, never a different model"
([R/contract.R:55-57](R/contract.R#L55-L57)).

The conformance batch check cannot see this — it compares two copies of one
context at one identical horizon
([R/conformance.R:235-253](R/conformance.R#L235-L253)). Triggering it needs a
context above 15360 values *and* mixed horizons, so it is rare in practice, but
it is a genuine determinism hole. Computing the truncation per series from that
series' own horizon would remove it.

### 4. Wasted key concatenation in attention — LOW

`all_key` is computed from the pre-RoPE keys
([R/timesfm-module.R:122-129](R/timesfm-module.R#L122-L129)) and then
unconditionally recomputed from the post-RoPE keys
([R/timesfm-module.R:136-140](R/timesfm-module.R#L136-L140)). The first
computation is dead. In the cached decode branch it is a full KV-cache
concatenation — allocated, copied, and discarded once per layer per decode step
(20 layers × up to 7 steps). The correct value is the second one; deleting the
first is a pure win and makes the RoPE-then-cache ordering legible.

### 5. Channel indices are hardcoded behind a config-driven façade — LOW

`outputs <- length(config$quantiles) + 1L` is computed
([R/timesfm-module.R:201](R/timesfm-module.R#L201),
[R/timesfm-inference.R:79](R/timesfm-inference.R#L79)) but never used to *index*
anything. The actual indexing is literal: `2:10` in `timesfm_flip_channels()`
([R/timesfm-inference.R:126](R/timesfm-inference.R#L126)), and `c(2:5, 7:10)`,
`6L`, `5:2`, `7:10` in the flag application
([R/timesfm-inference.R:147-160](R/timesfm-inference.R#L147-L160)).

This is safe *today* only because `validate_timesfm_config()` rejects any
variant with different quantiles ([R/arch-timesfm.R:50-63](R/arch-timesfm.R#L50-L63)).
It becomes a silent-wrong-answer generator the moment that gate is relaxed for a
second checkpoint. A comment naming the coupling, or deriving the indices from
`outputs`, would make the constraint survive the next contributor.

### 6. The interrupt guarantee is weaker than advertised — LOW

`zuk_check_user_interrupt()` calls an option hook and otherwise does nothing
([R/batching.R:112-116](R/batching.R#L112-L116)). The default hook is unset, so
in an ordinary session it is a no-op. Interrupts between batches are delivered by
R's own evaluator, not by this function, and a single long `torch` call is not
interruptible by either.

The behaviour is fine; the claim is what needs adjusting. The roadmap's "engine
batches and autoregressive blocks expose interrupt boundaries" reads as an active
check. The test simulates an interrupt by `stop()`ing a condition with class
`interrupt` ([test-batching.R:104-110](tests/testthat/test-batching.R#L104-L110)),
which verifies non-wrapping and the absence of a partial result — both real and
worth having — but not interruptibility itself.

### 7. The conformance `max_context` check is a no-op for TimesFM — LOW

`context_limit` returns early when the probe cannot reach the declared limit
([R/conformance.R:220-231](R/conformance.R#L220-L231)). With a 64-value default
probe and `max_context` of 16256, it always returns early. The roadmap's "passes
all ten applicable checks" is literally true, but this check contributes nothing
for this architecture. Real truncation coverage comes from the
`context_truncation` fixture instead — that is adequate, but the harness result
should not be read as covering it.

### 8. Optional dependencies are reached without the package's own guard — LOW

The codebase's convention is `zuk_require_namespace()`, which produces a typed
`zuk_error_capability` ([R/conditions.R:13-30](R/conditions.R#L13-L30)), and
`CLAUDE.md` mandates a guard for optional deps. Two exported functions skip it:

- `zuk_reg()` calls `parsnip::new_model_spec()` directly
  ([R/parsnip.R:35-42](R/parsnip.R#L35-L42));
- `context_length()` calls `dials::new_quant_param()` directly
  ([R/parsnip.R:166-175](R/parsnip.R#L166-L175)).

Both packages are in `Suggests`, so on a core installation these raise a bare
"there is no package called 'parsnip'" instead of the typed, actionable error a
consumer can branch on. `fit.R`, `forecast.R`, and `hub.R` all get this right.

### 9. `brulee` remains in Suggests for unregistered dead code — LOW

Chronos-2 is correctly unregistered ([R/zzz.R:10-13](R/zzz.R#L10-L13)) and
rejected before any network or tensor work
([R/hub.R:49-51](R/hub.R#L49-L51)) — Stage 0's disposition decision was made and
enforced. But `R/arch-chronos2.R` survives as reference-only code and `brulee`
stays in `Suggests` solely to support it. Stage 5 owns the dependency audit;
noting it here so it is not forgotten.

Minor sibling: `jsonlite` is a hard `Import` yet is reached through
`zuk_require_namespace("jsonlite")` as though optional
([R/hub.R:208-211](R/hub.R#L208-L211)). Harmless, but inconsistent.

---

## Stage-by-stage verdict

| Stage | Claimed | Assessment |
|---|---|---|
| 0 — Honest baseline | Closed locally | **Holds.** Registry, README table, catalogue, and tests agree. Chronos-2 unregistered and rejected pre-network. Remote CI still unrecorded. |
| 1 — Feasibility, contract, catalogue | Complete locally | **Holds.** Capability constructor enforces contract v1; all seven error leaves carry exactly one policy parent; `zuk_models()` is static and offline. Finding 8 is a small gap in the optional-dependency convention. |
| 2 — Reference, loader, lifecycle | Complete locally | **Holds.** Exact-header validation, 232-tensor spec, LRU keyed on model/revision/device/options, prefetch that constructs nothing. Committing exact float32 inputs alongside the JSON was the right call for reproducible parity. |
| 3 — Native inference | Complete locally (table says "Not started") | **Substantially complete, with findings 1–3 outstanding.** Finding 1 breaks a documented consumer path; 2 and 3 are contract-honesty and determinism gaps. |

## Test-coverage gaps

- **No test uses more than three quantile levels on a real handle.** This is what
  hides finding 1. A nine-level case belongs in the parity suite.
- **All TimesFM numerical evidence is opt-in.** Everything meaningful is gated on
  `ZUK_RUN_CHECKPOINT_TEST=true` plus a 925 MB cached checkpoint. Keeping CI
  network-free is correct, but it means the `state = "supported"` claim rests
  entirely on a manual local run that no CI has reproduced.
- **No weight-free test of the inference logic.** `timesfm_prepare_batch()`,
  padding/masking, running statistics, and the decode loop are all pure tensor
  logic that could be exercised on a small synthetic config with random weights
  in normal CI — shape, mask placement, and truncation behaviour without the
  checkpoint. `test-state-dict.R` already builds the module on the meta device,
  so the scaffolding for this exists.
- **Batch composition is untested.** No test mixes horizons within a batch
  (finding 3) or varies `batch_size` and asserts identical output.

## Documentation and process

- [.agents/roadmap.md:306](.agents/roadmap.md#L306) still records Stage 3 as
  `Not started` while lines 538-561 carry its full close-out.
- The entire Stage 3 delta is uncommitted: `R/timesfm-module.R`,
  `R/timesfm-inference.R`, `tests/testthat/helper-timesfm.R`, five `.f32`
  fixtures, and rewrites of `R/arch-timesfm.R` and six test files.
- README and NEWS were updated to match the new `supported` state and no longer
  overclaim; the Stage 0 honesty property has been maintained through Stage 3.
- GitHub Actions has still never run on `main`. Every "complete locally" in the
  roadmap depends on an environment that, as noted above, no longer exists on
  this machine.

## Recommended order of work

1. Fix finding 1 and add a nine-level parity case — it breaks the primary
   consumer path and currently reports itself as an internal engine bug.
2. Fix finding 3 (per-series truncation) and document or surface finding 2.
   Both are cheap and both are contract-honesty issues.
3. Restore the dependency environment and actually re-run the suite plus
   `R CMD check` before committing Stage 3; correct the roadmap table row.
4. Add weight-free tests for the preprocessing and decode paths so CI covers
   more than the stub.
5. Findings 4–9 are cleanup and can ride along with Stage 4 or Stage 5.

Findings 1–3 all live at the same seam: the engine validates permissively and
the architecture consumes strictly. Whatever the fixes, the durable improvement
is to have `check_quantile_levels()`, `check_horizon()`, and context truncation
hand the architecture *canonical, already-reconciled* values, so an architecture
never re-derives what the engine has already decided.

---

## Resolution

All nine findings are fixed. Two further defects surfaced once the suite could
actually run — both recorded below as findings 10 and 11.

The durable change is the one the summary above called for: `check_quantile_levels()`
now returns the *checkpoint's own* level values rather than the caller's
spelling, so an architecture selecting an output channel by position never
re-derives a match the engine already made.

| # | Finding | Fix |
|---|---|---|
| 1 | Nine-level request → NA forecasts | `check_quantile_levels()` canonicalises to the matched supported values ([capabilities.R](R/capabilities.R)); new `zuk_match_quantile_levels()` does tolerant matching; `timesfm_predict_batch()` resolves channels *before* tensor work and raises `zuk_error_quantile_levels` instead of emitting `NA` |
| 2 | Undeclared horizon-dependent context shrink | `timesfm_usable_context()` names the rule; `max_context` documented as the h ≤ 128 best case, with a horizon/context table in the README |
| 3 | Forecast depended on batch composition | Truncation is per series from that series' own horizon, not `max(horizons)` across the batch |
| 4 | Dead KV concat per layer per decode step | Removed; `all_key` is built once, after RoPE |
| 5 | Hardcoded channel indices | `timesfm_channel_layout()` derives every index; a test pins them to `2:10` / `6` / `2:5` / `7:10` for the pinned config |
| 6 | Overstated interrupt guarantee | Comment, NEWS, and roadmap now say what is actually guaranteed: boundary *placement*, with R delivering the interrupt and a single `torch` call uninterruptible |
| 7 | `max_context` check silently passed | Reports not-applicable with an actionable message when the probe is too short to reach the limit |
| 8 | Unguarded `parsnip`/`dials` calls | `zuk_reg()` and `context_length()` use `zuk_require_namespace()`, so a core install gets the same typed error as every other optional path |
| 9 | `brulee` for unregistered dead code | `arch-chronos2.R` moved to `.agents/reference/`, `brulee` dropped from `Suggests`; the pre-network rejection path is unchanged. Redundant `jsonlite` guard removed |

### 10. The CPU parity gate could not run on an accelerator host — HIGH

Found by execution. `test-parity-timesfm.R` loaded the handle with default
`device = NULL`, so `zuk_resolve_device()` picked this machine's MPS backend,
and the CPU-pinned fixture comparison then failed with `zuk_error_device`. The
gate was silently host-dependent: on any CUDA or MPS machine, the one test that
certifies numerical correctness could not run at all.

This is the sharpest illustration of the review's process point — the Stage 3
close-out was recorded on a host that happened to resolve to CPU. Fixed by
pinning `device = "cpu"`, which is what the CPU reference fixtures require.

### 11. Recursive default argument in the TimesFM batch closure — MEDIUM

`predict_batch <- function(..., device = device)` was a self-reference: calling
`model$predict_batch_fn()` without `device` raised R's opaque "promise already
under evaluation" error rather than defaulting to the handle's device. Masked
because every internal caller passes `device` explicitly. Now
`device = config$device`.

### Verification

- Deterministic suite: green. `R CMD check`: **0 errors, 0 warnings, 0 notes**.
- Real-checkpoint gate (`ZUK_RUN_CHECKPOINT_TEST=true`, 925 MB checkpoint
  downloaded and size-validated): all four golden parity fixtures pass on CPU
  within `1e-4`; `zuk_check_architecture()` passes against the real handle;
  repeated inference is identical and silent.
- New regression tests: canonical levels reach the architecture (via the exact
  values `zuk_models()` advertises); levels collapsing onto one trained level
  are refused; per-series truncation is independent of batch composition, both
  weight-free and against the real checkpoint; all nine levels forecast finite,
  monotone, and agree with the three-level request on the shared median;
  unexercised `max_context` reports not-applicable.

### Also changed, outside the findings

- **Numerical tolerance reworked to the ecosystem criterion.** The spike test
  failed on arrival, before any edit, at `1.07e-6` — 9–18 float32 ulps, normal
  reassociation rather than a port error. The deeper problem was the *form*: a
  bare `max(abs(diff)) < tol` is too tight for large values and too loose for
  small ones. Both levels now use `|actual - expected| <= atol + rtol*|expected|`
  via `expect_close_f32()` ([helper-close.R](tests/testthat/helper-close.R)),
  matching `torch.testing.assert_close()` and the upstream TimesFM tests.
  Block level takes PyTorch's float32 defaults (`atol=1e-5, rtol=1.3e-6`); the
  model fixtures record `atol=1e-4, rtol=1e-5`, and the generator emits the pair.

  This mattered more than it first appeared. Measured against the real
  checkpoint, the fixture gate's worst case is `3.05e-5` absolute and `7.33e-7`
  relative (4–12 ulps) — so the old bare `1e-4` had only **3.3× headroom**, and
  at these magnitudes (|v| ≈ 116) it was `8.6e-7` relative, *tighter than
  PyTorch's own float32 default*. With CI having never run, the first Linux x86
  runner was a plausible spurious failure on the one gate that certifies the
  supported model. The new budget gives ~40× headroom while staying orders of
  magnitude tighter than any structural error.

  Thread pinning was tried and rejected: LibTorch's native backend refuses to
  change intraop threads after parallel work starts, so `torch_set_num_threads()`
  inside a test is a one-way door that warns on restore. Documented as an
  `OMP_NUM_THREADS=1` env-var choice instead. It is not needed for correctness —
  repeated inference is already asserted bit-identical under default threading.
- **roxygen2 7.3.3 → 8.0.0.** `devtools::document()` ran under the newer roxygen
  available here, which replaced `RoxygenNote` with `Config/roxygen2/version` and
  reformatted three `.Rd` cross-references. Cosmetic, but it will churn if other
  contributors or CI pin roxygen 7 — worth a deliberate decision.
- `review.md` added to `.Rbuildignore`.

### Known follow-up, not fixed

`zuk_infer()` injects `0.5` into the requested levels so the point forecast is
exact ([forecast.R](R/forecast.R)). A checkpoint whose trained levels exclude
`0.5` would have that injection rejected by its own capability check. No current
or catalogued model has that shape, and fixing it means *deciding* what `.mean`
should be for a checkpoint without a median — a design question, not a defect to
patch silently. Worth settling before the second native architecture lands.
