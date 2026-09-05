# Toto at its real dimensions, with random weights instead of trained ones.
#
# Unlike the TimesFM helper this is *not* reduced, and cannot be:
# `toto_constructor()` validates every dimension against the pinned
# configuration, so a smaller variant is refused before construction. Going
# around it with `toto_module()` would buy a smaller model at the cost of
# testing the constructor, which is a worse trade -- and it would buy little,
# since construction takes 0.2s and the whole file runs in about the same time
# as the reduced TimesFM one.
#
# The point is that no checkpoint is downloaded. The structural invariants ---
# batch/loop agreement, determinism, monotone quantiles, the unobserved-patch
# masking --- hold for any weights, so they run on every platform rather than
# only where the 16 MB checkpoint is cached.
toto_synthetic_config <- function() {
  list(
    architecture = "toto2",
    model_id = "synthetic/toto",
    revision = strrep("c", 40L),
    device = "cpu",
    d_model = 256L,
    d_ff = 688L,
    num_heads = 4L,
    num_layers = 4L,
    qk_dim = 64L,
    v_dim = 64L,
    patch_size = 32L,
    num_output_patches = 1L,
    layer_group_size = 4L,
    num_variate_layers_per_group = 1L,
    variate_layer_first = FALSE,
    norm_eps = 1e-4,
    attn_bias = TRUE,
    mlp_bias = FALSE,
    qk_norm = FALSE,
    norm_include_weight = FALSE,
    pre_norm = TRUE,
    per_dim_scale = TRUE,
    use_xpos = TRUE,
    residual_mult = 0.75,
    residual_attn_ratio = sqrt(128 / log(128)),
    quantiles = seq(0.1, 0.9, by = 0.1)
  )
}

toto_synthetic_weights <- function(config = toto_synthetic_config(),
                                   seed = 20260904L) {
  torch::torch_manual_seed(seed)
  module <- toto_module(config)
  map <- toto_weight_map(config)
  shapes <- module$state_dict()
  weights <- lapply(names(map), function(source) {
    shape <- dim(shapes[[map[[source]]]])
    # The two residual gains are 0-dimension scalars, not shaped tensors.
    if (!length(shape)) {
      torch::torch_scalar_tensor(0.5)
    } else {
      torch::torch_randn(shape) * 0.05
    }
  })
  stats::setNames(weights, names(map))
}

toto_synthetic_model <- function(config = toto_synthetic_config()) {
  toto_constructor(config, toto_synthetic_weights(config))
}
