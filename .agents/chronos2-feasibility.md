# Chronos-2 native R `torch` feasibility

*Recorded: 2026-09-04.*

## Decision

**Go, and contract v2 is smaller than the roadmap assumes.**

The roadmap treats multivariate targets, past covariates, and future-known
covariates as three input channels that contract v2 must add. Chronos-2 does
not work that way. It has **one** mechanism: every series is a row, `group_ids`
says which rows form a task, and a row's *role* is expressed purely by whether
future values are supplied for it. That collapses most of the anticipated
design work.

## Pins and checkpoint facts

- Official source repository: `amazon-science/chronos-forecasting`
- Source commit: `8589d1988e9676817548e9626738ff06b6ca6370`
- Checkpoint: `amazon/chronos-2`
- Checkpoint revision: `29ec3766d36d6f73f0696f85560a422f50e8498c`
- Weight licence: Apache-2.0
- Required-file manifest: `config.json`, `model.safetensors`
- `model.safetensors`: 477,930,472 bytes, 170 tensors, all float32
- Parameters represented by the state dict: 119,477,664

Read from the safetensors header over an HTTP range request; no weights were
transferred. At 478 MB it sits between Toto's 16 MB and TimesFM's 925 MB.

## The input contract

This is the finding that matters, and it comes straight from the forward
signature:

```
forward(context,            # (batch, context_length)   -- every series is a row
        context_mask,       # (batch, context_length)
        group_ids,          # (batch,)                  -- which rows share a task
        future_covariates,  # (batch, future_length)
        future_covariates_mask,
        num_output_patches, ...)
```

There are no separate multivariate, past-covariate, or future-covariate
channels. A row is a target, a past-only covariate, or a future-known covariate
according to whether `future_covariates` carries values or `NaN` for it.
`group_ids` is an arbitrary labelling: rows sharing an id exchange information
through group attention, rows in different groups do not. Upstream's own
example makes the uniformity explicit --- one batch can hold a two-series
multivariate task, a covariate-informed task, and an independent univariate
series, distinguished only by ids and NaNs.

### What this means for contract v2

Contract v1 already passes a **list** of contexts:

```
predict_batch_fn(contexts, horizons, quantile_levels, device)
```

The extension is therefore additive rather than structural: v2 supplies
optional `group_ids` and future values alongside the existing list. A v1
architecture never receives them and needs no dummy arguments, which is exactly
the backward-compatibility promise in `consumer-api.md`. The open design
questions become smaller and more concrete: how a caller expresses roles and
groups at the *engine* boundary, and how the optional adapters carry that
through, not what the architecture signature should look like.

## Architecture

Encoder-only T5 --- no decoder, so a single forward pass:

- 12 blocks, `d_model` 768, `d_ff` 3072, 12 heads x `d_kv` 64
- **two attention sublayers per block**, plus one MLP and three layer norms.
  That is time attention and group attention in *every* block, where Toto
  alternates them across blocks
- plain ReLU feed-forward (`is_gated_act: false`), unlike Toto's SwiGLU
- RoPE at `theta = 10000`, layer-norm epsilon 1e-6
- `shared.weight [2, 768]`: embeddings for the pad and register tokens
  (`use_reg_token: true`)

Forecast configuration: context 8192, input and output patch 16, stride 16,
`max_output_patches` 64 -- so a maximum horizon of **1024** -- `use_arcsinh`,
and a `time_encoding_scale` of 8192.

**21 quantile levels**, 0.01 to 0.99, emitted directly:
`output_patch_embedding.output_layer` is `[336, 3072]` and `336 = 16 x 21`.
That is a far richer set than the nine TimesFM and Toto share, and the engine's
level reconciliation will meet its first checkpoint whose trained levels are
not a superset-of-nine.

Input embedding is `[3072, 48]` with `48 = 3 x 16`: the patch carries a **time
encoding**, the values, and the mask, concatenated on the feature axis. The
time encoding is `arange(-context_length, 0) / time_encoding_scale`, so absolute
position is a feature here, not only a rotation. Patch validity follows the same
rule as Toto: a patch attends if it holds at least one observation.

## Operator spike

A full `Chronos2EncoderBlock` --- T5 layer norm, RoPE, time attention, group
attention, and the feed-forward --- was implemented in R `torch` and compared
against the pinned reference on five series arranged as two groups of two plus
a singleton, with one series carrying unobserved leading patches:

```
Chronos-2 encoder block   max|diff| = 4.768e-07   rel = 2.193e-07
```

It matched on the first attempt, which is itself informative: **there is no
unit-scaling layer.** Plain `nnf_linear` reproduces the reference exactly, so
the sixth open question below is resolved, and the largest single risk carried
over from the Toto port does not apply here.

Four details carry parity risk and none are visible from the state dict:

- **Attention is unscaled.** `scaled_dot_product_attention(..., scale=1.0)`,
  the T5 convention where scaling is folded into initialization. The
  conventional `1/sqrt(d_kv)` would be wrong by a factor of 8 at `d_kv = 64`.
- **RoPE is split-half** --- `cat((-x2, x1))` over halves of the head, with
  `emb = cat((freqs, freqs))`. This is the opposite of Toto's interleaved
  adjacent-pair convention, so the package will carry two rotary conventions
  and neither can be borrowed for the other.
- **Group attention swaps the time and series axes** before attending, then
  swaps back, and uses no RoPE --- there is no natural ordering along the
  series axis. Structurally this is Toto's variate layer, but expressed through
  the mask rather than a reshape, and present in *every* block.
- **Masks are additive**, `0` for attend and `finfo.min` for refuse, combined
  as `einsum("qb,bt->qbt", group_mask, time_mask)` so a token attends only to
  tokens in its own group that are also observed at that time.

`Chronos2LayerNorm` is T5-style: no mean subtraction, a learnable weight, and
variance accumulated in float32 --- unlike Toto's parameter-free norm.

## Open questions

1. The exact `instance_norm` and `arcsinh` composition, and whether scaling is
   per row or per group.
2. How `_prepare_patched_future` encodes future-known covariates, and how the
   register token participates.
3. The group-attention mask construction, including how groups of unequal size
   are padded.
4. Whether the 21 trained levels are emitted sorted, or need the same repair
   Toto's head does.
5. Categorical covariates: the model card advertises real *and* categorical,
   but the forward signature takes float tensors, so the encoding happens
   upstream in the pipeline and needs locating.
6. ~~Whether anything rescales the forward pass the way `dd_unit_scaling` does
   in Toto.~~ **Resolved: it does not.** The encoder block reproduces exactly
   with plain linear layers.

## Next gates

Phase A is complete: the checkpoint is pinned, the input contract is understood,
and the encoder block is verified. What remains is Phase B and Phase C.

- Freeze contract v2 against these findings --- optional group ids and future
  values alongside the existing context list, with v1 architectures untouched.
- Port: input embedding with its time-encoding channel, the register token,
  `instance_norm` composed with `asinh`, the twelve-block encoder, and the
  21-level output head.
- Conformance, then pinned parity fixtures, then a `supported` catalogue row.

The reference environment is `chronos-forecasting>=2.0` plus `transformers`;
the port itself needs neither.
