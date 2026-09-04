# Portions derived from Toto 2.0, Copyright 2026 Datadog, Inc.
# Translated and modified for R/torch by the zukzeit authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Checkpoint-to-module tensor mapping for the pinned Toto 2.0 4M checkpoint.
#
# The R module uses flat parameter names where the reference nests a module per
# projection, so the map is explicit rather than an identity. Keeping it
# explicit is what makes an unsupported checkpoint variant diagnosable before
# any tensor is assigned.

toto_weight_map <- function(config) {
  layers <- as.integer(config$num_layers)
  map <- c(
    "patch_proj.linear1.weight"   = "patch_proj.linear1",
    "patch_proj.linear1.bias"     = "patch_proj.linear1_bias",
    "patch_proj.linear2.weight"   = "patch_proj.linear2",
    "patch_proj.linear2.bias"     = "patch_proj.linear2_bias",
    "patch_proj.skip_proj.weight" = "patch_proj.skip",
    "patch_proj.skip_proj.bias"   = "patch_proj.skip_bias"
  )
  for (idx in seq_len(layers) - 1L) {
    src <- sprintf("transformer.layers.%d", idx)
    dst <- sprintf("layers.%d", idx)
    map <- c(map, stats::setNames(
      paste0(dst, c(
        ".attn.in_proj", ".attn.in_proj_bias", ".attn.out_proj",
        ".attn.out_proj_bias", ".attn.per_dim_scale",
        ".attn_tau", ".mlp_tau", ".ff1", ".ff2"
      )),
      paste0(src, c(
        ".attn.in_proj.weight", ".attn.in_proj.bias", ".attn.out_proj.weight",
        ".attn.out_proj.bias", ".attn._pds.per_dim_scale",
        ".attn_tau", ".mlp_tau", ".ffn.fc1.weight", ".ffn.fc2.weight"
      ))
    ))
  }
  head <- "output_head.param_projection.proj"
  c(map, stats::setNames(
    paste0("output_head.", c("linear1", "linear1_bias", "linear2",
                             "linear2_bias", "skip", "skip_bias")),
    paste0(head, c(".linear1.weight", ".linear1.bias", ".linear2.weight",
                   ".linear2.bias", ".skip_proj.weight", ".skip_proj.bias"))
  ))
}

toto_load_module_weights <- function(module, weights, config) {
  map <- toto_weight_map(config)
  if (!is.list(weights) || is.null(names(weights))) {
    zuk_abort_checkpoint(
      "Toto requires a complete named state dict.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      expected = names(map),
      actual = names(weights)
    )
  }
  missing <- setdiff(names(map), names(weights))
  unexpected <- setdiff(names(weights), names(map))
  module_names <- names(module$state_dict())
  if (length(missing) || length(unexpected) || !setequal(unname(map), module_names)) {
    zuk_abort_checkpoint(
      "Toto checkpoint tensors do not map exactly onto the native module.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      tensor = if (length(missing)) missing[[1]] else NA_character_,
      expected = module_names,
      actual = names(weights)
    )
  }
  mapped <- weights[names(map)]
  names(mapped) <- unname(map)
  module$load_state_dict(mapped, .refer_to_state_dict = TRUE)
  invisible(module)
}
