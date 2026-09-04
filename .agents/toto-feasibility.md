# Toto 2.0 4M native R `torch` feasibility

*Recorded: 2026-09-04.*

## Decision

**Go, pending the operator spike.** Every fact needed to commit to the port is
now pinned, and the checkpoint's contract fit is better than the roadmap
assumed: Toto 2.0 emits **native quantiles at the same nine levels as TimesFM**,
so it lands in contract v1 unchanged and needs no new output type.

Two properties make this a cheaper port than TimesFM:

- **Single-pass decoding is available.** Toto is a next-patch predictor, and
  with `decode_block_size = 0` the whole horizon decodes in one forward pass:
  no autoregressive loop, no KV cache, no per-block interrupt boundary. That is
  a *choice*, not a property of the model — see "Inference knobs" below.
- **A quantile head trained with pinball loss.** Toto 1.0's Student-T mixture
  is gone, so quantiles are read directly off the head rather than derived by
  numerical CDF inversion or sampling. Parity is therefore deterministic and
  achievable, which a sampling-based head would not have been.

This is a feasibility result, not a support claim. The operator spike,
end-to-end preprocessing/decode semantics, and golden parity fixtures remain
Stage 5 gates. `toto2` stays out of the catalogue until both gates pass.

## Pins and checkpoint facts

- Official source repository: `DataDog/toto`
- Source commit: `46bd92adeeef6b3c2afb21008659f607fa203e38`
- Checkpoint: `Datadog/Toto-2.0-4m`
- Checkpoint revision: `8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9`
- Weight licence: Apache-2.0
- Required-file manifest: `config.json`, `model.safetensors`
- `config.json`: 592 bytes
- `model.safetensors`: 16,582,848 bytes, 48 tensors, all float32
- Parameters represented by the state dict: 4,144,456
- Technical report: arXiv:2605.20119

The safetensors header was inspected over an HTTP range request without
transferring the data payload, then confirmed by loading the file into an R
`torch` state dict: 48 named tensors, 4,144,456 parameters, all `Float`, all
finite. The two scalar residual gains load as 0-dimension tensors and read back
through `$item()`.

At 16 MB this checkpoint is 56x smaller than TimesFM's, so unlike the TimesFM
gates it is cheap enough to consider exercising in more than a single opt-in
test path.

## Configuration

```
d_model 256   d_ff 688    num_layers 4    num_heads 4
qk_dim 64     v_dim 64    patch_size 32   num_output_patches 1
num_groups 4  heads_per_group 1           per_dim_scale true
pre_norm true norm_eps 1e-4               norm_include_weight false
attn_bias true              mlp_bias false             qk_norm false
use_xpos true               layer_group_size 4
num_variate_layers_per_group 1            variate_layer_first false
residual_mult 0.75          residual_attn_ratio 5.136215466577748
```

`num_groups` / `heads_per_group` describe attention-head grouping, not layer
grouping: `in_proj` is `[768, 256]`, so Q, K and V are each full width and the
4 heads x 64 arrangement is plain multi-head attention. Layer alternation is
governed instead by `layer_group_size 4` with `num_variate_layers_per_group 1`
and `variate_layer_first false`, which reads as one group of four layers whose
**last** layer attends over the variate axis and whose first three attend
causally over time. That reading is consistent with the state dict — all four
layers carry identical parameter shapes, because the distinction is the axis
attention runs over, not the parameters — but it is an inference from the
config and must be confirmed against the pinned source before the port lands.

## Forward-graph inventory

The head width confirms the contract fit directly:
`288 = patch_size 32 x 9 quantiles`, with `num_output_patches 1`.

**Every component below is affected by unit scaling — see the next section.**
Nothing is a drop-in reuse of the TimesFM modules.

| Component | TimesFM equivalent | Status |
|---|---|---|
| Patch projection, `[1024, 64]` = `2 x patch_size` in | `timesfm_residual_block` | **adapt** — same tensor shapes, different arithmetic |
| Per-dimension query scale, `[64]` | `timesfm_per_dim_scale` | **adapt** — Toto omits the `1/sqrt(head_dim)` factor |
| Fused QKV, `in_proj [768, 256]` | `timesfm_attention` qkv | **adapt** — Toto carries a bias |
| RMS normalization | `timesfm_rms_norm` | **adapt** — `norm_include_weight false`, so parameter-free; note the state dict carries no norm tensors at all |
| Output head, residual block to `[288]` | continuous quantile head | **adapt** |
| Positional encoding | RoPE | **new** — Toto uses xPos (`use_xpos true`) |
| Feed-forward | plain SiLU `ff0`/`ff1` | **new** — `fc1 [1376, 256]` is `2 x d_ff`, i.e. gated (SwiGLU), no bias |
| Residual path | plain addition | **new** — learnable scalar `attn_tau` / `mlp_tau` per layer, with `residual_mult` and `residual_attn_ratio` |
| Attention axes | causal over time only | **new** — alternating causal-time and full-variate attention |
| Input scaling | cumulative reversible normalization | **adapt** — see below; closer to TimesFM than the model card suggests |
| Decoding | autoregressive with KV cache | **simpler** — single pass when `decode_block_size = 0` |

The shapes line up with TimesFM's modules, but the arithmetic does not: the
whole forward pass runs through `dd_unit_scaling`, so no TimesFM module can be
reused unmodified.

## Unit scaling changes the forward pass

The most consequential finding, and one the state dict gives no hint of. Toto is
trained under u-muP, and `dd_unit_scaling` (`unit-scaling` 0.3.5, vendored by
Datadog) applies **inference-time** scaling factors inside every `Linear`,
`SiLU`, residual and per-dimension scale. A port that treats `uu.Linear` as a
plain linear layer produces wrong numbers everywhere, silently, with a state
dict that maps perfectly.

Measured against the pinned library, the rules are closed-form:

| Operator | Rule |
|---|---|
| `uu.Linear(x, W, b)` | `linear(x, W, b) / sqrt(fan_in)` — the bias is *inside* the scale |
| `uu.LinearReadout(x, W, b)` | `linear(x, W, b) / fan_in` |
| `U.silu(x)` | `silu(x) * 1.76678` , exactly `exp(0.2 log 2 + 0.8 log sqrt(2 / (1 - 1/pi)))` at `mult = 1` |
| `U.residual_split(x, tau)` | `(x, x)` — identity in the forward pass; it only shapes gradients |
| `U.residual_add(h, skip, tau)` | `(tau * h + skip) / sqrt(1 + tau^2)` |
| `uu.PerDimScale(x, p)` | `x * softplus(p) / log 2` |
| `uu.RMSNorm(x)` with `include_weight = FALSE` | plain parameter-free RMS normalization |

`PerDimScale` is where the difference bites quietly: TimesFM computes
`softplus(p) / (log 2 * sqrt(head_dim))` and Toto drops the `1/sqrt(head_dim)`,
because unit-scaled attention accounts for it elsewhere.

`attn_tau` and `mlp_tau` are `register_buffer`s derived from
`transformer_residual_scaling_rule(residual_mult, residual_attn_ratio)`, but
they are persistent buffers and therefore **present in the state dict** — the
port reads the trained values rather than reimplementing the rule.

### Spike result

All seven primitives are implemented in R `torch` and agree with the pinned
reference on shared random inputs:

```
uu.Linear              max|diff| = 0.000e+00
uu.LinearReadout       max|diff| = 0.000e+00
U.silu                 max|diff| = 0.000e+00
uu.PerDimScale         max|diff| = 2.384e-07
uu.RMSNorm             max|diff| = 1.192e-07
residual_add tau=.52   max|diff| = 2.384e-07
residual_add tau=.09   max|diff| = 1.192e-07
```

Three are bit-identical; the rest sit at float32 rounding. The foundational
layer of the port is therefore verified. Attention, the decoder loop, and the
end-to-end forecast remain.

## Attention

Verified against `toto2.model.SelfAttention` for both regimes. Four details
carry real parity risk and none are visible from the state dict:

- **Only half of each head dimension is rotated.** `partial_factor = (0.0, 0.5)`
  splits the 64-wide head into a rotated leading 32 and an untouched trailing
  32.
- **Query and key take opposite xPos exponents,** `+1` and `-1`, so their
  product decays with distance. Applying one exponent to both is a plausible
  mistake that leaves shapes intact.
- **Attention is scaled by `1 / qk_dim`, not `1 / sqrt(qk_dim)`** — MuP, to stop
  logits exploding as width grows. Off by a factor of 8 at `qk_dim = 64`.
- **The rotation is interleaved,** pairing adjacent channels via `(dim r)` with
  `r = 2`, rather than the split-half convention used elsewhere in the
  ecosystem. The cos/sin tables repeat each frequency twice to match.

Structurally: `in_proj` splits `[qk*heads, qk*groups, v*groups] = [256, 256,
256]`; `PerDimScale` applies to the query only; `qk_norm` is off for this
checkpoint; variate layers receive **no** rotary projection at all and attend
non-causally, while time layers are causal.

### Spike result

```
time-axis layer      max|diff| = 1.863e-09   rel = 1.704e-07
variate-axis layer   max|diff| = 2.328e-09   rel = 2.649e-07
```

Both regimes agree with the reference at float32 rounding. Every novel operator
in the port is now verified; what remains is assembly, preprocessing, decode,
and the two gates.

## Pinned inference knobs

Per the decision recorded below, `zukzeit` implements
`decode_block_size = 0`, `scaler_fallback_min_obs = 0`,
`quantile_real_cap_k = 0` — the raw `Toto2Model.forecast()` defaults.

## Contract fit

- **Quantiles.** Trained levels are `0.1, 0.2, ..., 0.9` — bit-identical in
  spelling terms to the set TimesFM declares, so the engine's existing
  reconciliation at the batch boundary applies unchanged.
- **Univariate only for `0.1.0`.** Toto supports multivariate targets through
  its variate-axis attention, but contract v1 has no channel for them.
  `multivariate` stays `FALSE` and the port advertises only the univariate
  subset it can demonstrate with fixtures, as the roadmap permits. The variate
  layers still execute — attention over a single variate is a learned
  transform, not an identity — so they must be implemented, not skipped.
- **No sample paths.** `samples` stays `FALSE`; nothing in the head produces
  them.

## Resolved against the pinned source

All six questions opened by `config.json` are now answered from
`DataDog/toto@46bd92ad`, `toto2/toto2/{configuration,model}.py`.

1. **Trained context length is 4096.** `Toto2ModelConfig` carries no
   `context_length`, but it derives `residual_attn_ratio` from one:
   `sqrt(S / log S)` with `S = context_length / patch_size`. Inverting the
   checkpoint's stored `5.136215466577748` gives `S = 128` exactly, so
   `context_length = 128 x 32 = 4096`. The reconstruction is bit-identical in
   float64, not approximate.
2. **There is no hard positional cap.** The rotary tables are built with
   `max_len = 8192` *patches* and regrow on demand, and they are non-persistent
   buffers — which is why they are absent from the 48-tensor state dict. Beyond
   4096, xPos extrapolates rather than failing. `max_context` is therefore a
   quality judgement, and 4096 is the defensible declaration.
3. **The scaler is `PatchedCausalStdScaler` composed with `asinh`,** not an
   arcsinh scaler as the model card's summary implies. Location and scale are
   causal cumulative mean and Welford variance, computed **in float64**, then
   made patch-aware by taking each patch's last value and repeating it across
   the patch. The model applies `asinh` to the scaled series, and inverts with
   `sinh(q) * scale + loc` on every quantile channel. The loc/scale half is a
   close cousin of TimesFM's cumulative RevIN, so this is an adaptation rather
   than new work — but the float64 accumulation must be reproduced or parity
   will drift.
4. **xPos:** `base = 10000`, `xpos_scale_base = 256`, `xpos_scale_exponent = 1`,
   with `xpos_base_scale = (arange(0, w, 2) + 0.4w) / 1.4w`. The decay is
   centred on `floor((max_position + 1) / 2)`, so the rotation depends on the
   sequence length being decoded — a parity trap worth an explicit fixture.
5. **Layer alternation confirmed.** `_if_variate_layer()` with
   `variate_layer_first = False` reduces to `layer_idx %% 4 >= 3`, so layers
   0-2 attend causally over time and layer 3 attends over the variate axis,
   exactly as the config reading predicted.
6. **The head can emit crossing quantiles.** `QuantileKnotsOutputHead$forward()`
   projects and rearranges with no monotonic parameterisation — no cumulative
   softplus, no sort. The port must repair crossings before returning, as
   TimesFM does, or the engine's monotonicity check will reject real forecasts.

## Inference knobs

The one finding that needs a decision rather than an implementation. Three
kwargs change the numbers, and their defaults differ between the raw
`Toto2Model.forecast()` API and the `Toto2GluonTSModelConfig` wrapper the model
card demonstrates:

| Knob | `forecast()` default | GluonTS config default | Effect |
|---|---|---|---|
| `decode_block_size` | `0` (single pass) | `None`, card example passes `768` | Blockwise decoding with median feedback between blocks, re-running the causal scaler each iteration |
| `scaler_fallback_min_obs` | `0` (no-op) | `8` | Backfills loc/scale on short leading patches |
| `quantile_real_cap_k` | `0` (disabled) | `1e4` | Clips each real-space quantile to `[ctx_min - K*scale, ctx_max + K*scale]` |

`zukzeit` must pin one combination, implement exactly that, and record it in
the catalogue entry and the fixture generator. Fixtures generated under
different knobs are not comparable, and a user checking our output against the
upstream example would otherwise see an unexplained difference.

Recommendation: `decode_block_size = 0`, `scaler_fallback_min_obs = 0`,
`quantile_real_cap_k = 0` — the raw model defaults. They give single-pass
decoding, no feedback loop, and no post-hoc clipping, which is the smallest
honest surface to certify for contract v1. The other combinations can follow
once the baseline has pinned parity evidence.

## Next gates

- Operator spike in R `torch` for xPos, the gated FFN, the scalar-tau residual
  path, and variate-axis attention, compared against the pinned Python
  reference below `1e-5`, mirroring the TimesFM spike.
- Golden fixtures from the pinned source for short, typical, truncated,
  mixed-sign, and multi-series batch/loop cases.
- `zuk_check_architecture()` conformance and CPU parity within a recorded
  `atol`/`rtol` budget, at which point `toto2` may become a `supported`
  catalogue row.

The parity environment follows the TimesFM pattern: a pinned requirements file
plus a generator that asserts the source commit and writes fixtures. `torch`
2.14.0 publishes a `cp314` macOS arm64 wheel and this workspace also has Python
3.12 and 3.13 plus `uv` and `conda`, so the reference environment is
constructible here; nothing about parity is blocked on tooling.
