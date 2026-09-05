# Portions derived from Chronos-2, Copyright Amazon.com, Inc. or its affiliates.
# Translated and modified for R/torch by the zukzeit authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Chronos-2 preprocessing, single-pass encoding, and forecast transforms.

NEGATIVE_INFINITY <- -3.4028234663852886e38

# `InstanceNorm`: whole-series standardization in float32, then arcsinh. Unlike
# the Toto port's causal scaler this looks at the entire context at once, so a
# patch is normalised by statistics that include everything after it.
chronos2_instance_norm <- function(values, observed, epsilon = 1e-5) {
  # float32, matching the reference. Accumulating in float64 here would be
  # strictly more precise and would still pass parity, but it is a deviation
  # and it cannot run on backends without double precision.
  high <- values$to(dtype = torch::torch_float32())
  keep <- observed$to(dtype = torch::torch_float32())
  counts <- keep$sum(dim = -1, keepdim = TRUE)$clamp_min(1)
  location <- (high * keep)$sum(dim = -1, keepdim = TRUE) / counts
  spread <- torch::torch_sqrt(
    (torch::torch_square(high - location) * keep)$sum(dim = -1, keepdim = TRUE) / counts
  )
  spread <- torch::torch_where(spread == 0, torch::torch_full_like(spread, epsilon), spread)
  list(loc = location$to(dtype = values$dtype), scale = spread$to(dtype = values$dtype))
}

chronos2_scale <- function(values, stats) {
  torch::torch_asinh((values - stats$loc) / stats$scale)
}

chronos2_unscale <- function(values, stats) {
  torch::torch_sinh(values) * stats$scale + stats$loc
}

# A patch carries three channels: a sequential time index divided by the
# encoding scale, the values, and the observation mask. Absolute position is
# therefore a feature here, not only a rotation.
chronos2_patch_channels <- function(values, observed, patch, time_index, scale) {
  fold <- function(x) x$reshape(c(x$shape[[1]], x$shape[[2]] %/% patch, patch))
  encoding <- (time_index / scale)$unsqueeze(1)$expand(
    c(values$shape[[1]], time_index$shape[[1]])
  )
  torch::torch_cat(
    list(fold(encoding), fold(values * observed$to(dtype = values$dtype)),
         fold(observed$to(dtype = values$dtype))),
    dim = -1
  )
}

# Left-pad each row to a whole number of patches. A row shorter than the batch
# width is padded with unobserved positions, which the attention mask drops.
chronos2_prepare_batch <- function(contexts, config, device) {
  patch <- as.integer(config$input_patch_size)
  limit <- as.integer(config$context_length)
  contexts <- lapply(contexts, function(ctx) utils::tail(as.numeric(ctx), limit))
  width <- as.integer(ceiling(max(vapply(contexts, length, integer(1))) / patch) * patch)
  values <- matrix(0, nrow = length(contexts), ncol = width)
  observed <- matrix(FALSE, nrow = length(contexts), ncol = width)
  for (i in seq_along(contexts)) {
    at <- seq.int(width - length(contexts[[i]]) + 1L, width)
    values[i, at] <- contexts[[i]]
    observed[i, at] <- TRUE
  }
  list(
    values = torch::torch_tensor(values, dtype = torch::torch_float32(), device = device),
    observed = torch::torch_tensor(observed, dtype = torch::torch_bool(), device = device),
    width = width
  )
}

chronos2_predict_batch <- function(module, contexts, horizons, quantile_levels,
                                   config, device, groups = NULL) {
  if (!length(contexts)) return(list())
  horizons <- as.integer(horizons)
  patch <- as.integer(config$output_patch_size)
  trained <- as.numeric(config$quantiles)
  matched <- zuk_match_quantile_levels(quantile_levels, trained)
  if (anyNA(matched)) {
    zuk_abort_quantile_levels(
      c(
        "Chronos-2 cannot emit every requested quantile level.",
        "x" = "Unsupported: {.val {as.numeric(quantile_levels)[is.na(matched)]}}.",
        "i" = "Trained levels: {.val {trained}}."
      ),
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      requested = as.numeric(quantile_levels), supported = trained
    )
  }
  groups <- groups %||% list(id = as.character(seq_along(contexts)),
                             target = rep(TRUE, length(contexts)),
                             future = vector("list", length(contexts)))

  device <- torch::torch_device(device)
  prepared <- chronos2_prepare_batch(contexts, config, device)
  horizon <- max(horizons)
  n_out <- as.integer(ceiling(horizon / patch))
  padded <- n_out * patch
  scale <- as.numeric(config$time_encoding_scale)
  rows <- length(contexts)

  result <- torch::with_no_grad({
    zuk_check_user_interrupt()
    stats <- chronos2_instance_norm(prepared$values, prepared$observed)
    scaled <- chronos2_scale(prepared$values, stats) *
      prepared$observed$to(dtype = torch::torch_float32())

    context_time <- torch::torch_arange(-prepared$width, -1,
                                        dtype = torch::torch_float32(), device = device)
    context_patches <- chronos2_patch_channels(
      scaled, prepared$observed, as.integer(config$input_patch_size),
      context_time, scale
    )
    embeddings <- module$input_embedding(context_patches)
    attend <- (prepared$observed$to(dtype = torch::torch_int32())$reshape(
      c(rows, prepared$width %/% as.integer(config$input_patch_size),
        as.integer(config$input_patch_size))
    )$sum(dim = -1) > 0)$to(dtype = torch::torch_float32())

    # The register token sits between context and future, always attended.
    reg <- module$shared$narrow(1, 2L, 1L)$unsqueeze(1)$expand(c(rows, 1L, module$shared$shape[[2]]))

    # Known future values ride in the future patches; absent ones stay zero and
    # unobserved, which is how a pure forecast request looks.
    future_values <- torch::torch_zeros(c(rows, padded), dtype = torch::torch_float32(),
                                        device = device)
    future_observed <- torch::torch_zeros(c(rows, padded), dtype = torch::torch_bool(),
                                          device = device)
    for (i in seq_along(groups$future)) {
      ahead <- groups$future[[i]]
      if (is.null(ahead)) next
      supplied <- torch::torch_tensor(as.numeric(ahead), dtype = torch::torch_float32(),
                                      device = device)
      future_values[i, 1:length(ahead)] <- supplied
      future_observed[i, 1:length(ahead)] <- TRUE
    }
    future_scaled <- chronos2_scale(future_values, stats) *
      future_observed$to(dtype = torch::torch_float32())
    future_time <- torch::torch_arange(0, padded - 1, dtype = torch::torch_float32(),
                                       device = device)
    future_patches <- chronos2_patch_channels(future_scaled, future_observed, patch,
                                              future_time, scale)
    future_embeddings <- module$input_embedding(future_patches)

    embeddings <- torch::torch_cat(list(embeddings, reg, future_embeddings), dim = 2L)
    attend <- torch::torch_cat(
      list(attend, torch::torch_ones(c(rows, 1L + n_out), dtype = torch::torch_float32(),
                                     device = device)),
      dim = -1
    )
    time_mask <- ((1 - attend) * NEGATIVE_INFINITY)$unsqueeze(2)$unsqueeze(2)

    # Rows share a task when they share an id; the group pass may only attend
    # within a task, and only where that row is observed at that step.
    same <- torch::torch_tensor(
      outer(groups$id, groups$id, `==`) + 0, dtype = torch::torch_float32(), device = device
    )
    group_time <- torch::torch_einsum("qb,bt->qbt", list(same, attend))
    group_time_mask <- ((1 - group_time$permute(c(3, 1, 2))) * NEGATIVE_INFINITY)$unsqueeze(2)

    hidden <- module(embeddings, time_mask, group_time_mask)
    forecast_embeds <- hidden$narrow(2, hidden$shape[[2]] - n_out + 1L, n_out)
    predictions <- module$output_embedding(forecast_embeds)
    # (rows, n_out, quantiles * patch) -> (rows, quantiles, n_out * patch)
    predictions <- predictions$reshape(c(rows, n_out, length(trained), patch))
    predictions <- predictions$permute(c(1, 3, 2, 4))$reshape(c(rows, length(trained), padded))
    # The statistics gain a quantile axis so they broadcast over all 21 levels.
    chronos2_unscale(predictions, list(loc = stats$loc$unsqueeze(-1),
                                       scale = stats$scale$unsqueeze(-1)))
  })

  array <- as.array(result$to(device = torch::torch_device("cpu")))
  targets <- which(groups$target)
  lapply(targets, function(i) {
    matrix(
      aperm(array[i, matched, seq_len(horizons[[i]]), drop = FALSE], c(1, 3, 2)),
      nrow = horizons[[i]], ncol = length(matched)
    )
  })
}
