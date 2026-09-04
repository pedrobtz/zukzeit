# TimesFM 2.5 native R `torch` feasibility

*Recorded: 2026-08-13.*

## Decision

**Go.** The hard TimesFM 2.5 transformer path can be expressed with the R
`torch`/LibTorch surface used by `zukzeit`. A reduced deterministic block agrees
with the pinned official PyTorch implementation to less than `1e-6`, including
RMS normalization, fused QKV, rotary embeddings, normalized attention,
per-dimension query scaling, the feed-forward residual path, and the continuous
quantile-head layout.

This is a feasibility result, not a support claim. Loading the complete
checkpoint, reproducing preprocessing/decode semantics, and passing the
end-to-end golden forecast fixtures remain Stage 2 and Stage 3 gates.

## Pins and checkpoint facts

- Official source repository: `google-research/timesfm`
- Source commit: `3dae50b20d7a724981e8ea36cda75578f80dd2dc`
- Checkpoint: `google/timesfm-2.5-200m-pytorch`
- Checkpoint revision: `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`
- Weight licence: Apache-2.0
- `model.safetensors`: 925,181,104 bytes, 232 tensors, all float32
- Parameters represented by the state dict: 231,289,280

The safetensors header was inspected without loading the data payload. PyTorch
linear tensors already use the R `torch` `[out_features, in_features]` layout,
so the fused QKV checkpoint tensor is copied without a transpose. Each layer's
fused QKV weight is `[3840, 1280]`; the query, key, and value views are
`[batch, patches, 16, 80]`.

The pinned configuration declares 20 transformer layers, model dimension 1280,
16 heads, head dimension 80, patch length 32, context length 16,384, point-head
horizon 128, continuous-quantile horizon 1,024, and trained quantiles 0.1
through 0.9.

## Forward-graph inventory

The implementation was traced through the pinned official files under
`src/timesfm/torch/`, particularly `normalization.py`, `dense.py`,
`transformer.py`, and `timesfm_2p5/timesfm_2p5_torch.py`.

| Graph operation | Dtype/layout | R `torch` expression used by the spike | Checkpoint transformation |
|---|---|---|---|
| Patch tokenization | Float32; 32 values concatenated with 32 masks, then `[B, P, 64]` | Linear + SiLU residual block; full tokenizer deferred | None; linear weights remain `[out, in]` |
| RMS normalization | Float32, normalize last dimension, epsilon `1e-6` | `torch_mean(torch_square(x))`, `torch_rsqrt()` | Copy one scale vector |
| Fused QKV | `[B, P, 1280] -> [B, P, 3840]` | `nn_linear(..., bias = FALSE)`, `torch_chunk()` | Direct copy of `[3840, 1280]`; no transpose |
| Rotary position embedding | `[B, P, 16, 80]` | `torch_arange()`, `torch_exp()`, `torch_sin()`, `torch_cos()`, split/cat | None |
| Q/K normalization | Normalize each 80-dimensional head | Same RMS module | Direct scale-vector copies |
| Per-dimension query scale | Float32 vector `[80]` | `nnf_softplus()` and elementwise multiply | Direct parameter copy |
| Causal masked attention | Q/K/V permuted to `[B, 16, P, 80]`; upstream uses `scale=1` | `torch_matmul()`, logical mask, `torch_where()`, `nnf_softmax()` | None |
| Attention projection | `[B, P, 1280]` | Bias-free `nn_linear()` | Direct `[1280, 1280]` copy |
| Transformer residuals | Pre/post RMS norms around attention and feed-forward | Elementwise addition | Direct scale-vector copies |
| Feed-forward | `1280 -> 1280 -> 1280` | Two bias-free linears and `nnf_silu()` | Direct matrix copies |
| Point head | Last dimension reshaped to `128 x 10` | SiLU residual block | Direct copies; full decode deferred |
| Continuous quantile head | Last dimension reshaped to `1024 x 10` | SiLU residual block and `reshape()` | Direct copies; output matrices are `[10240, 1280]` |

The complete checkpoint additionally needs left padding to a patch boundary,
per-patch running statistics/reversible normalization, mask propagation,
autoregressive patch decoding for horizons beyond 128, continuous-head channel
selection, and quantile-crossing repair. Those are semantic implementation
work, not missing R operators.

## Reference comparison

`.agents/generate-timesfm-spike-reference.py` imports the pinned official
PyTorch modules and assigns deterministic values to a 4-dimensional, two-head,
one-block model. `tests/testthat/test-timesfm-feasibility.R` assigns the same
values to the R implementation and compares:

- transformer embeddings with shape `[1, 3, 4]`;
- continuous-head output with shape `[1, 3, 3, 3]`;
- a fused QKV tensor with shape `[12, 4]`, including rejection of a transposed
  or otherwise incompatible tensor.

The R results agree with the official outputs at an absolute tolerance below
`1e-6`. The parity fixture uses a fully observed three-patch sequence. Prefix
mask behavior is part of the end-to-end reference fixtures rather than this
small operator fixture, because older PyTorch fused-attention releases differ
in the value returned for a query whose entire key row is masked; that row is
not consumed by TimesFM decoding.

## CPU measurement

`.agents/benchmark-timesfm-spike.R` exercises one exact-width transformer
block over 512 patches, equivalent to the maximum 16,384-value context at patch
length 32, and applies the exact-size continuous head to the final patch.

Measurement host:

- x86_64 macOS Sequoia
- R 4.5.2
- R `torch` 0.17 with LibTorch 2.8, CPU

Observed result:

| Measurement | Result |
|---|---:|
| Parameters in one block plus continuous head | 37,688,560 |
| Float32 parameter bytes | 150,754,240 |
| Transformer output | `[1, 512, 1280]` |
| Continuous-head output | `[1, 1, 1024, 10]` |
| Timed forward pass | 0.133 seconds |
| Maximum resident set, including R/LibTorch startup | 649,007,104 bytes |
| Approximate RSS above an R/LibTorch baseline | 302,845,952 bytes |

This is not a full-model performance forecast: 20 layers and the 925 MB state
dict materially increase resident memory and runtime. It establishes that the
largest sequence shape and exact layer/head operations execute on CPU with
bounded intermediate memory, and that only the final patch needs the large
continuous head.

## Remaining risks

- Stage 2 loaded all 232 float32 tensors in about 5.3 seconds with roughly
  1.40 GB maximum RSS. Stage 3 must avoid retaining both the named state dict
  and a second copied module state after assignment.
- Full parity depends on faithfully porting masking, reversible normalization,
  autoregressive decode, continuous quantile adjustment, and crossing repair.
- CPU latency must be measured again with all 20 layers and representative
  forecast horizons.
- CUDA and MPS remain opportunistic execution targets; CPU is the `0.1.0`
  portability gate.
