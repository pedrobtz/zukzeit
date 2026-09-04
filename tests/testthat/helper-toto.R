# A tiny Toto of the same shape as the real one: same layer types, same weight
# map, same decode path, ~19k parameters instead of 4.1M. The structural
# invariants --- batch/loop agreement, determinism, monotone quantiles, the
# unobserved-patch masking --- hold for any weights, so they can be checked on
# every platform rather than only where the checkpoint is cached.
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
