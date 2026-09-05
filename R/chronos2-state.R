# Portions derived from Chronos-2, Copyright Amazon.com, Inc. or its affiliates.
# Translated and modified for R/torch by the zukzeit authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Checkpoint-to-module tensor mapping for the pinned Chronos-2 checkpoint.
#
# Upstream nests each block's two attention sublayers under `layer.0` and
# `layer.1`, with the feed-forward at `layer.2`; the R module names them for
# what they attend over. The map is explicit so an unsupported checkpoint
# variant is diagnosable before any tensor is assigned.

chronos2_weight_map <- function(config) {
  map <- c(
    "input_patch_embedding.hidden_layer.weight"    = "input_embedding.hidden",
    "input_patch_embedding.hidden_layer.bias"      = "input_embedding.hidden_bias",
    "input_patch_embedding.output_layer.weight"    = "input_embedding.out",
    "input_patch_embedding.output_layer.bias"      = "input_embedding.out_bias",
    "input_patch_embedding.residual_layer.weight"  = "input_embedding.skip",
    "input_patch_embedding.residual_layer.bias"    = "input_embedding.skip_bias",
    "output_patch_embedding.hidden_layer.weight"   = "output_embedding.hidden",
    "output_patch_embedding.hidden_layer.bias"     = "output_embedding.hidden_bias",
    "output_patch_embedding.output_layer.weight"   = "output_embedding.out",
    "output_patch_embedding.output_layer.bias"     = "output_embedding.out_bias",
    "output_patch_embedding.residual_layer.weight" = "output_embedding.skip",
    "output_patch_embedding.residual_layer.bias"   = "output_embedding.skip_bias",
    "shared.weight"                                = "shared",
    "encoder.final_layer_norm.weight"              = "final_norm"
  )
  for (idx in seq_len(as.integer(config$num_layers)) - 1L) {
    src <- sprintf("encoder.block.%d", idx)
    dst <- sprintf("blocks.%d", idx)
    map <- c(map, stats::setNames(
      c(
        paste0(dst, ".time_attn.", c("q", "k", "v", "o")),
        paste0(dst, ".time_norm"),
        paste0(dst, ".group_attn.", c("q", "k", "v", "o")),
        paste0(dst, ".group_norm"),
        paste0(dst, c(".ff_norm", ".wi", ".wo"))
      ),
      c(
        paste0(src, ".layer.0.self_attention.", c("q", "k", "v", "o"), ".weight"),
        paste0(src, ".layer.0.layer_norm.weight"),
        paste0(src, ".layer.1.self_attention.", c("q", "k", "v", "o"), ".weight"),
        paste0(src, ".layer.1.layer_norm.weight"),
        paste0(src, c(".layer.2.layer_norm.weight", ".layer.2.mlp.wi.weight",
                      ".layer.2.mlp.wo.weight"))
      )
    ))
  }
  map
}

chronos2_load_module_weights <- function(module, weights, config) {
  map <- chronos2_weight_map(config)
  if (!is.list(weights) || is.null(names(weights))) {
    zuk_abort_checkpoint(
      "Chronos-2 requires a complete named state dict.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      expected = names(map), actual = names(weights)
    )
  }
  missing <- setdiff(names(map), names(weights))
  unexpected <- setdiff(names(weights), names(map))
  module_names <- names(module$state_dict())
  if (length(missing) || length(unexpected) || !setequal(unname(map), module_names)) {
    zuk_abort_checkpoint(
      "Chronos-2 checkpoint tensors do not map exactly onto the native module.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      tensor = if (length(missing)) missing[[1]] else NA_character_,
      expected = module_names, actual = names(weights)
    )
  }
  mapped <- weights[names(map)]
  names(mapped) <- unname(map)
  module$load_state_dict(mapped, .refer_to_state_dict = TRUE)
  invisible(module)
}
