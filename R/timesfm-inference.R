# Portions derived from TimesFM, Copyright 2025 Google LLC.
# Translated and modified for R/torch by the zukzeit authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# TimesFM preprocessing, autoregressive decoding, and forecast transforms.

timesfm_round_up <- function(value, multiple) {
  as.integer(ceiling(value / multiple) * multiple)
}

# Output channels of the pinned checkpoint: channel 1 carries the point
# forecast and channels 2..(q+1) the trained quantiles in ascending order.
# Every index the decode and forecast-flag transforms use is derived here, so a
# checkpoint with a different quantile set cannot silently read a wrong channel.
timesfm_channel_layout <- function(config) {
  quantiles <- as.numeric(config$quantiles)
  quantile_channels <- seq_along(quantiles) + 1L
  median_channel <- which.min(abs(quantiles - 0.5)) + 1L
  list(
    outputs           = length(quantiles) + 1L,
    quantile_channels = quantile_channels,
    median_channel    = median_channel,
    lower_channels    = quantile_channels[quantile_channels < median_channel],
    upper_channels    = quantile_channels[quantile_channels > median_channel]
  )
}

# The checkpoint's position budget covers the context *and* the rounded forecast
# block, so usable history shrinks as the horizon grows. Deriving it per series
# keeps a forecast independent of whichever other series share its batch.
timesfm_usable_context <- function(horizon, config) {
  limit <- as.integer(config$context_length)
  point_horizon <- as.integer(config$horizon_length)
  limit - timesfm_round_up(horizon, point_horizon)
}

timesfm_update_running_stats <- function(n, mu, sigma, x, mask) {
  legitimate <- torch::torch_logical_not(mask)
  inc_n <- legitimate$to(dtype = x$dtype)$sum(dim = -1)
  inc_n_safe <- torch::torch_where(inc_n == 0, torch::torch_ones_like(inc_n), inc_n)
  inc_mu <- (x * legitimate)$sum(dim = -1) / inc_n_safe
  inc_mu <- torch::torch_where(inc_n == 0, torch::torch_zeros_like(inc_mu), inc_mu)
  inc_var <- (torch::torch_square(x - inc_mu$unsqueeze(-1)) * legitimate)$sum(dim = -1) /
    inc_n_safe
  inc_var <- torch::torch_where(inc_n == 0, torch::torch_zeros_like(inc_var), inc_var)
  inc_sigma <- torch::torch_sqrt(inc_var)
  new_n <- n + inc_n
  new_n_safe <- torch::torch_where(new_n == 0, torch::torch_ones_like(new_n), new_n)
  new_mu <- (n * mu + inc_n * inc_mu) / new_n_safe
  new_mu <- torch::torch_where(new_n == 0, torch::torch_zeros_like(new_mu), new_mu)
  new_var <- (
    n * torch::torch_square(sigma) +
      inc_n * torch::torch_square(inc_sigma) +
      n * torch::torch_square(mu - new_mu) +
      inc_n * torch::torch_square(inc_mu - new_mu)
  ) / new_n_safe
  new_var <- torch::torch_where(new_n == 0, torch::torch_zeros_like(new_var), new_var)
  list(n = new_n, mu = new_mu, sigma = torch::torch_sqrt(torch::torch_clamp(new_var, min = 0)))
}

timesfm_revin <- function(x, mu, sigma, reverse = FALSE) {
  while (length(mu$shape) < length(x$shape)) {
    mu <- mu$unsqueeze(-1)
    sigma <- sigma$unsqueeze(-1)
  }
  if (isTRUE(reverse)) {
    x * sigma + mu
  } else {
    (x - mu) / torch::torch_where(
      sigma < 1e-6, torch::torch_ones_like(sigma), sigma
    )
  }
}

timesfm_running_patch_stats <- function(patches, masks, initial = NULL) {
  batch <- patches$shape[[1]]
  if (is.null(initial)) {
    n <- torch::torch_zeros(batch, dtype = patches$dtype, device = patches$device)
    mu <- torch::torch_zeros_like(n)
    sigma <- torch::torch_zeros_like(n)
  } else {
    n <- initial$n
    mu <- initial$mu
    sigma <- initial$sigma
  }
  mus <- vector("list", patches$shape[[2]])
  sigmas <- vector("list", patches$shape[[2]])
  for (i in seq_len(patches$shape[[2]])) {
    state <- timesfm_update_running_stats(n, mu, sigma, patches[, i, ], masks[, i, ])
    n <- state$n
    mu <- state$mu
    sigma <- state$sigma
    mus[[i]] <- mu
    sigmas[[i]] <- sigma
  }
  list(
    n = n,
    mu = mu,
    sigma = sigma,
    patch_mu = torch::torch_stack(mus, dim = 2L),
    patch_sigma = torch::torch_stack(sigmas, dim = 2L)
  )
}

timesfm_decode <- function(module, horizon, inputs, masks, config,
                           layout = timesfm_channel_layout(config)) {
  patch <- as.integer(config$patch_length)
  point_horizon <- as.integer(config$horizon_length)
  quantile_horizon <- as.integer(config$quantile_horizon_length)
  outputs <- layout$outputs
  batch <- inputs$shape[[1]]
  decode_steps <- (as.integer(horizon) - 1L) %/% point_horizon
  patched_inputs <- inputs$reshape(c(batch, -1L, patch))
  patched_masks <- masks$reshape(c(batch, -1L, patch))
  stats <- timesfm_running_patch_stats(patched_inputs, patched_masks)
  normed <- timesfm_revin(
    patched_inputs, stats$patch_mu, stats$patch_sigma, reverse = FALSE
  )
  normed <- torch::torch_where(patched_masks, torch::torch_zeros_like(normed), normed)
  forward <- module(normed, patched_masks)
  point <- timesfm_revin(
    forward$point, stats$patch_mu, stats$patch_sigma, reverse = TRUE
  )$reshape(c(batch, -1L, point_horizon, outputs))
  quantiles <- timesfm_revin(
    forward$quantiles, stats$patch_mu, stats$patch_sigma, reverse = TRUE
  )$reshape(c(batch, -1L, quantile_horizon, outputs))
  quantiles <- quantiles[, quantiles$shape[[2]], , ]

  autoregressive <- vector("list", decode_steps)
  last_output <- point[, point$shape[[2]], , layout$median_channel]
  caches <- forward$caches
  current <- list(n = stats$n, mu = stats$mu, sigma = stats$sigma)
  if (decode_steps > 0L) {
    patches_per_output <- point_horizon %/% patch
    for (step in seq_len(decode_steps)) {
      zuk_check_user_interrupt()
      new_input <- last_output$reshape(c(batch, patches_per_output, patch))
      new_mask <- torch::torch_zeros_like(new_input, dtype = torch::torch_bool())
      new_stats <- timesfm_running_patch_stats(new_input, new_mask, current)
      current <- list(n = new_stats$n, mu = new_stats$mu, sigma = new_stats$sigma)
      new_normed <- timesfm_revin(
        new_input, new_stats$patch_mu, new_stats$patch_sigma, reverse = FALSE
      )
      forward <- module(new_normed, new_mask, caches)
      caches <- forward$caches
      new_point <- timesfm_revin(
        forward$point, new_stats$patch_mu, new_stats$patch_sigma, reverse = TRUE
      )$reshape(c(batch, patches_per_output, point_horizon, outputs))
      autoregressive[[step]] <- new_point[, patches_per_output, , ]
      last_output <- new_point[, patches_per_output, , layout$median_channel]
    }
  }
  list(point = point, quantiles = quantiles, autoregressive = autoregressive)
}

timesfm_flip_channels <- function(x, layout) {
  torch::torch_cat(
    list(
      x[, , 1L, drop = FALSE],
      torch::torch_flip(x[, , layout$quantile_channels], dims = -1)
    ),
    dim = -1
  )
}

timesfm_full_forecast <- function(decoded, horizon) {
  pieces <- list(decoded$point[, decoded$point$shape[[2]], , ])
  if (length(decoded$autoregressive)) pieces <- c(pieces, decoded$autoregressive)
  torch::torch_cat(pieces, dim = 2L)[, seq_len(horizon), ]
}

timesfm_apply_forecast_flags <- function(module, inputs, masks, horizons, config,
                                         layout = timesfm_channel_layout(config)) {
  max_horizon <- max(horizons)
  decoded <- timesfm_decode(module, max_horizon, inputs, masks, config, layout)
  full <- timesfm_full_forecast(decoded, max_horizon)
  spreads <- decoded$quantiles

  flipped <- timesfm_decode(module, max_horizon, -inputs, masks, config, layout)
  flipped_full <- timesfm_flip_channels(
    timesfm_full_forecast(flipped, max_horizon), layout
  )
  flipped_spreads <- timesfm_flip_channels(flipped$quantiles, layout)
  full <- (full - flipped_full) / 2
  spreads <- (spreads - flipped_spreads) / 2

  median_channel <- layout$median_channel
  for (channel in c(layout$lower_channels, layout$upper_channels)) {
    full[, , channel] <- spreads[, seq_len(max_horizon), channel] -
      spreads[, seq_len(max_horizon), median_channel] + full[, , median_channel]
  }
  for (channel in rev(layout$lower_channels)) {
    full[, , channel] <- torch::torch_minimum(
      full[, , channel], full[, , channel + 1L]
    )
  }
  for (channel in layout$upper_channels) {
    full[, , channel] <- torch::torch_maximum(
      full[, , channel], full[, , channel - 1L]
    )
  }
  positive <- (inputs >= 0)$all(dim = -1, keepdim = TRUE)$unsqueeze(-1)
  torch::torch_where(
    positive, torch::torch_maximum(full, torch::torch_zeros_like(full)), full
  )
}

timesfm_prepare_batch <- function(contexts, horizons, config, device) {
  patch <- as.integer(config$patch_length)
  usable <- vapply(horizons, timesfm_usable_context, integer(1), config = config)
  if (any(usable <= 0L)) {
    zuk_abort_capability(
      "Requested horizon leaves no room for context inside the checkpoint's position budget.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      capability = "horizon",
      requested = horizons[usable <= 0L],
      supported = as.integer(config$context_length) - as.integer(config$horizon_length)
    )
  }
  contexts <- lapply(seq_along(contexts), function(i) {
    utils::tail(as.numeric(contexts[[i]]), usable[[i]])
  })
  padded_length <- timesfm_round_up(max(vapply(contexts, length, integer(1))), patch)
  values <- matrix(0, nrow = length(contexts), ncol = padded_length)
  masks <- matrix(TRUE, nrow = length(contexts), ncol = padded_length)
  for (i in seq_along(contexts)) {
    index <- seq.int(padded_length - length(contexts[[i]]) + 1L, padded_length)
    values[i, index] <- contexts[[i]]
    masks[i, index] <- FALSE
  }
  list(
    inputs = torch::torch_tensor(values, dtype = torch::torch_float32(), device = device),
    masks = torch::torch_tensor(masks, dtype = torch::torch_bool(), device = device)
  )
}

timesfm_predict_batch <- function(module, contexts, horizons, quantile_levels,
                                  config, device) {
  if (!length(contexts)) return(list())
  horizons <- as.integer(horizons)
  if (length(horizons) != length(contexts)) {
    zuk_abort_capability(
      "TimesFM needs one horizon per context.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      capability = "horizon",
      requested = length(horizons),
      supported = length(contexts)
    )
  }
  if (any(vapply(contexts, length, integer(1)) == 0L)) {
    zuk_abort_capability(
      "TimesFM requires at least one observed context value.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      capability = "context",
      requested = 0L,
      supported = "at least one finite observation"
    )
  }
  # Resolve output channels before any tensor work, so an unsupported level is a
  # typed refusal rather than an NA column the engine later reports as a bug.
  layout <- timesfm_channel_layout(config)
  trained <- as.numeric(config$quantiles)
  matched <- zuk_match_quantile_levels(quantile_levels, trained)
  if (anyNA(matched)) {
    zuk_abort_quantile_levels(
      c(
        "TimesFM cannot emit every requested quantile level.",
        "x" = "Unsupported: {.val {as.numeric(quantile_levels)[is.na(matched)]}}.",
        "i" = "Trained levels: {.val {trained}}."
      ),
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      requested = as.numeric(quantile_levels),
      supported = trained
    )
  }
  channels <- layout$quantile_channels[matched]

  device <- torch::torch_device(device)
  prepared <- timesfm_prepare_batch(contexts, horizons, config, device)
  result <- torch::with_no_grad({
    timesfm_apply_forecast_flags(
      module, prepared$inputs, prepared$masks, horizons, config, layout
    )
  })
  array <- as.array(result$to(device = torch::torch_device("cpu")))
  lapply(seq_along(contexts), function(i) {
    matrix(
      array[i, seq_len(horizons[[i]]), channels, drop = FALSE],
      nrow = horizons[[i]],
      ncol = length(channels)
    )
  })
}
