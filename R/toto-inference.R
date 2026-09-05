# Portions derived from Toto 2.0, Copyright 2026 Datadog, Inc.
# Translated and modified for R/torch by the zukzeit authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Toto preprocessing, single-pass decoding, and forecast transforms.
#
# This port pins the raw `Toto2Model$forecast()` knobs: no block decoding, no
# short-patch scaler fallback, and no real-space quantile cap. That is the
# smallest surface that can be certified against fixtures, and it is recorded
# in the catalogue so a caller comparing against the upstream GluonTS wrapper,
# whose defaults differ, can see why the numbers move.

toto_round_up <- function(value, multiple) {
  as.integer(ceiling(value / multiple) * multiple)
}

# `PatchedCausalStdScaler`. Statistics accumulate causally in float64 --- the
# reference is explicit about the dtype, and float32 accumulation drifts
# visibly over a long context --- then each patch adopts the statistics at its
# own last position, so a patch is normalised by everything up to its end and
# nothing after it.
toto_causal_scaler <- function(data, mask, patch, minimum_scale = 1e-6,
                               correction = 1) {
  # The reference accumulates in float64 and falls back to float32 where the
  # backend has no double precision, warning when it does. MPS is exactly that
  # case; porting the float64 without the fallback made the model unusable on
  # Apple silicon.
  precision <- tryCatch(
    {
      data$to(dtype = torch::torch_float64())
      torch::torch_float64()
    },
    error = function(e) {
      cli::cli_warn(c(
        "Float64 is unsupported on device {.val {as.character(data$device)}}.",
        "i" = "Using float32 for the scaler; forecasts may differ in the last digits."
      ))
      torch::torch_float32()
    }
  )
  high <- data$to(dtype = precision)
  keep <- mask$to(dtype = precision)
  counts <- torch::torch_cumsum(keep, dim = -1)$clamp_min(1)
  location <- torch::torch_cumsum(high * keep, dim = -1) / counts

  # Welford: the increment uses the *previous* running mean, so the first
  # position contributes zero variance rather than a spurious value.
  previous <- torch::torch_cat(
    list(torch::torch_zeros_like(location$narrow(-1, 1L, 1L)),
         location$narrow(-1, 1L, location$shape[[length(location$shape)]] - 1L)),
    dim = -1
  )
  increment <- (high - previous) * (high - location) * keep
  variance <- torch::torch_cumsum(increment, dim = -1) /
    (counts - correction)$clamp(min = 1)
  scale <- torch::torch_sqrt(variance)$clamp(min = minimum_scale)

  # Patch-aware: broadcast each patch's final statistic across the patch.
  last_in_patch <- function(x) {
    shape <- x$shape
    n <- shape[[length(shape)]]
    folded <- x$reshape(c(utils::head(shape, -1), n %/% patch, patch))
    tail <- folded$narrow(-1, patch, 1L)
    tail$expand(c(utils::head(folded$shape, -1), patch))$reshape(shape)
  }
  list(
    loc = last_in_patch(location)$to(dtype = data$dtype),
    scale = last_in_patch(scale)$to(dtype = data$dtype)
  )
}

# Left-pad each context to a common patch-aligned width. Unobserved leading
# patches are marked so attention can drop them, which is how a short series
# rides in the same batch as a long one without seeing invented history.
toto_prepare_batch <- function(contexts, horizons, config, device) {
  patch <- as.integer(config$patch_size)
  usable <- as.integer(config$context_length %||% 4096L)
  contexts <- lapply(contexts, function(ctx) utils::tail(as.numeric(ctx), usable))
  width <- toto_round_up(max(vapply(contexts, length, integer(1))), patch)
  values <- matrix(0, nrow = length(contexts), ncol = width)
  observed <- matrix(FALSE, nrow = length(contexts), ncol = width)
  for (i in seq_along(contexts)) {
    at <- seq.int(width - length(contexts[[i]]) + 1L, width)
    values[i, at] <- contexts[[i]]
    observed[i, at] <- TRUE
  }
  # The reference forces the final context patch observed: training never
  # systematically leaves the end of a context unobserved, so a tail-padded
  # series would be out of distribution.
  observed[, seq.int(width - patch + 1L, width)] <- TRUE
  list(
    values = torch::torch_tensor(values, dtype = torch::torch_float32(), device = device),
    observed = torch::torch_tensor(observed, dtype = torch::torch_bool(), device = device),
    width = width
  )
}

# Fold a flat series into patches and pair it with its mask channel.
toto_embed_patches <- function(module, values, observed, patch) {
  fold <- function(x) x$reshape(c(x$shape[[1]], x$shape[[2]] %/% patch, patch))
  module$patch_proj(torch::torch_cat(
    list(fold(values), fold((!observed)$to(dtype = values$dtype))), dim = -1
  ))
}

# Replace infinities with the extreme finite value across the quantile axis,
# mirroring the reference's guard before quantiles are sorted.
toto_clamp_nonfinite <- function(x, dim) {
  finite <- torch::torch_isfinite(x)
  upper <- torch::torch_where(finite, x, torch::torch_full_like(x, -Inf))$amax(dim = dim, keepdim = TRUE)
  lower <- torch::torch_where(finite, x, torch::torch_full_like(x, Inf))$amin(dim = dim, keepdim = TRUE)
  torch::torch_where(
    x == Inf, upper,
    torch::torch_where(x == -Inf, lower, x)
  )
}

toto_predict_batch <- function(module, contexts, horizons, quantile_levels,
                               config, device) {
  if (!length(contexts)) return(list())
  horizons <- as.integer(horizons)
  patch <- as.integer(config$patch_size)
  trained <- as.numeric(config$quantiles)
  matched <- zuk_match_quantile_levels(quantile_levels, trained)
  if (anyNA(matched)) {
    zuk_abort_quantile_levels(
      c(
        "Toto cannot emit every requested quantile level.",
        "x" = "Unsupported: {.val {as.numeric(quantile_levels)[is.na(matched)]}}.",
        "i" = "Trained levels: {.val {trained}}."
      ),
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      requested = as.numeric(quantile_levels),
      supported = trained
    )
  }

  device <- torch::torch_device(device)
  prepared <- toto_prepare_batch(contexts, horizons, config, device)
  horizon <- max(horizons)
  n_patches <- as.integer(ceiling(horizon / patch))
  predicted <- n_patches * patch

  result <- torch::with_no_grad({
    zuk_check_user_interrupt()
    # The prediction region is appended as zeros and marked unobserved, so the
    # causal statistics see the context alone.
    zeros <- torch::torch_zeros(c(prepared$values$shape[[1]], predicted),
                                dtype = torch::torch_float32(), device = device)
    full_values <- torch::torch_cat(list(prepared$values, zeros), dim = -1)
    full_observed <- torch::torch_cat(
      list(prepared$observed,
           torch::torch_zeros_like(zeros)$to(dtype = torch::torch_bool())),
      dim = -1
    )
    stats <- toto_causal_scaler(full_values, full_observed, patch)

    scaled <- torch::torch_where(
      full_observed,
      (full_values - stats$loc) / stats$scale,
      torch::torch_zeros_like(full_values)
    )$asinh()

    embedded <- toto_embed_patches(module, scaled, full_observed, patch)
    # A patch with no observations at all carries group id -1 upstream, which
    # excludes it from attention. Left-padding a short series to the width of
    # its batch creates exactly such patches, so without this a series forecasts
    # differently depending on which other series share its batch.
    # Only *context* patches are ever marked invalid. The prediction region is
    # unobserved by construction, and excluding it would stop the patches being
    # forecast from attending to any history at all.
    context_patches <- prepared$width %/% patch
    observed_per_patch <- prepared$observed$to(dtype = torch::torch_int32())$reshape(
      c(prepared$observed$shape[[1]], context_patches, patch)
    )$sum(dim = -1)
    valid <- torch::torch_cat(
      list(observed_per_patch > 0,
           torch::torch_ones(c(prepared$observed$shape[[1]], n_patches),
                             dtype = torch::torch_bool(), device = device)),
      dim = -1
    )
    attn_mask <- (valid$unsqueeze(-1) == valid$unsqueeze(-2))$unsqueeze(2)
    state <- module$run_layers(embedded, attn_mask)
    # Next-patch prediction: the anchor is the last context patch, so the
    # aligned outputs are the final `n_patches + 1` positions less the last.
    total <- state$shape[[2]]
    anchor <- state$narrow(2, total - n_patches, n_patches)
    block <- module$output_head(anchor)$unflatten(-1, c(patch, length(trained)))

    fold <- function(x) {
      tail_x <- x$narrow(-1, prepared$width + 1L, predicted)
      tail_x$reshape(c(tail_x$shape[[1]], n_patches, patch))
    }
    real <- torch::torch_sinh(block) * fold(stats$scale)$unsqueeze(-1) +
      fold(stats$loc)$unsqueeze(-1)
    real <- toto_clamp_nonfinite(real, dim = -1)
    # The head has no monotone parameterisation, so quantiles are sorted
    # across the level axis exactly as the reference does.
    real <- torch::torch_sort(real, dim = -1)[[1]]
    real$reshape(c(real$shape[[1]], n_patches * patch, length(trained)))
  })

  array <- as.array(result$to(device = torch::torch_device("cpu")))
  channels <- matched
  lapply(seq_along(contexts), function(i) {
    matrix(
      array[i, seq_len(horizons[[i]]), channels, drop = FALSE],
      nrow = horizons[[i]], ncol = length(channels)
    )
  })
}
