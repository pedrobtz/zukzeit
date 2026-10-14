# Portions derived from Toto 2.0, Copyright 2026 Datadog, Inc.
# Translated and modified for R/zulm by the zukzeit authors.
# Licensed under Apache-2.0; see inst/COPYRIGHTS and
# inst/LICENSES/Apache-2.0.txt.

# Toto 2.0 4M on zulm's tensor runtime (zulm design §7.9, D33).
#
# The model layer stays here: the configuration and weight map, the causal
# scaler and asinh in R doubles, patching, the attention mask, the next-patch
# alignment, the inverse transform and the quantile sort. zulm holds the
# network: one graph per batch shape, built once and cached.
#
# Layouts are zulm's, feature-first and column-major: torch's (B, L, D) is
# c(D, L, B), and a torch weight [out, in] is the array c(in, out).

# The weights as R arrays in zulm's layout: torch tensors (the synthetic
# helper's) are reversed, arrays from zulm::zl_safetensors_read() already are.
toto_zulm_arrays <- function(weights) {
  lapply(weights, function(w) {
    if (inherits(w, "torch_tensor")) {
      a <- as.array(w$cpu())
      if (length(dim(a)) > 1L) aperm(a) else as.numeric(a)
    } else {
      w
    }
  })
}

toto_zulm_store <- function(weights, config) {
  map <- toto_weight_map(config)
  missing <- setdiff(names(map), names(weights))
  unexpected <- setdiff(names(weights), names(map))
  if (length(missing) || length(unexpected)) {
    zuk_abort_checkpoint(
      "Toto checkpoint tensors do not map exactly onto the native module.",
      model_id = config$model_id %||% NA_character_,
      revision = config$revision %||% NA_character_,
      tensor = if (length(missing)) missing[[1]] else unexpected[[1]],
      expected = names(map),
      actual = names(weights)
    )
  }
  w <- toto_zulm_arrays(weights)
  names(w) <- unname(map[names(w)])
  p <- zulm::zl_params()
  put <- function(name, value) {
    value <- if (is.null(dim(value))) as.numeric(value) else value
    d <- if (is.null(dim(value))) length(value) else dim(value)
    zulm::zl_param(p, name, d, init = zulm::zl_init_zeros())
    zulm::zl_param_set(p, name, value)
  }
  for (pre in c("patch_proj", "output_head")) {
    for (part in c("linear1", "linear1_bias", "linear2", "linear2_bias", "skip", "skip_bias")) {
      put(paste0(pre, ".", part), w[[paste0(pre, ".", part)]])
    }
  }
  heads <- as.integer(config$num_heads)
  width <- as.integer(config$qk_dim)
  vw <- as.integer(config$v_dim)
  ff <- as.integer(config$d_ff)
  taus <- list()
  for (idx in seq_len(as.integer(config$num_layers)) - 1L) {
    l <- sprintf("layers.%d", idx)
    inw <- w[[paste0(l, ".attn.in_proj")]]       # c(D, 3 * heads * width)
    inb <- w[[paste0(l, ".attn.in_proj_bias")]]
    qk <- heads * width
    cols <- list(q = seq_len(qk), k = qk + seq_len(qk), v = 2L * qk + seq_len(heads * vw))
    for (part in names(cols)) {
      put(paste0(l, ".", part), inw[, cols[[part]], drop = FALSE])
      put(paste0(l, ".", part, "_bias"), inb[cols[[part]]])
    }
    put(paste0(l, ".o"), w[[paste0(l, ".attn.out_proj")]])
    put(paste0(l, ".o_bias"), w[[paste0(l, ".attn.out_proj_bias")]])
    # uu.PerDimScale: softplus(s) / log(2), fixed at inference
    s <- as.numeric(w[[paste0(l, ".attn.per_dim_scale")]])
    put(paste0(l, ".pds"), log1p(exp(s)) / log(2))
    ff1 <- w[[paste0(l, ".ff1")]]                  # c(D, 2 * ff): value, then gate
    put(paste0(l, ".ff_value"), ff1[, seq_len(ff), drop = FALSE])
    put(paste0(l, ".ff_gate"), ff1[, ff + seq_len(ff), drop = FALSE])
    put(paste0(l, ".ff2"), w[[paste0(l, ".ff2")]])
    taus[[l]] <- c(attn = as.numeric(w[[paste0(l, ".attn_tau")]]),
                   mlp = as.numeric(w[[paste0(l, ".mlp_tau")]]))
  }
  list(params = p, taus = taus)
}

# The network of toto-module.R as a zulm graph. `x` holds the inputs:
# patches c(2 * patch, P, B), mask c(P, P, 1, B), pos (i32, P), xq and xk
# c(qk_dim, 1, P), anchor (i32, n_patches * B).
toto_zulm_build <- function(store, config, n_patches) {
  p <- store$params
  heads <- as.integer(config$num_heads)
  width <- as.integer(config$qk_dim)
  vw <- as.integer(config$v_dim)
  dim <- as.integer(config$d_model)
  eps <- as.numeric(config$norm_eps)
  half <- width %/% 2L
  silu_mult <- exp(0.2 * log(2) + 0.8 * log(sqrt(2 / (1 - 1 / pi))))
  t <- function(name) zulm::zl_param_tensor(p, name)
  fan_in <- function(name) zulm::zl_param_tensor(p, name)$dim[[1]]
  # uu.Linear and uu.LinearReadout: the bias is inside the scale
  lin <- function(x, name, bias = TRUE, readout = FALSE) {
    y <- zulm::zl_crossprod(t(name), x)
    if (bias) y <- y + t(paste0(name, "_bias"))
    k <- fan_in(name)
    y * (1 / if (readout) k else sqrt(k))
  }
  residual <- function(hidden, skip, tau) (hidden * tau + skip) * (1 / sqrt(1 + tau * tau))
  mlp <- function(x, pre, readout) {
    second <- function(v, name) lin(v, paste0(pre, ".", name), readout = readout)
    h <- zulm::zl_silu(lin(x, paste0(pre, ".linear1"))) * silu_mult
    residual(second(h, "linear2"), second(x, "skip"), 1)
  }
  layer <- function(x, l, variate, input) {
    taus <- store$taus[[l]]
    n <- dim(x)[[2L]]
    b <- dim(x)[[3L]]
    h <- zulm::zl_rms_norm(x, eps)
    if (variate) {
      # attention over a single token returns its value exactly
      attended <- lin(h, paste0(l, ".v"))
    } else {
      split <- function(v) zulm::zl_reshape(v, c(width, heads, n, b))
      q <- split(lin(h, paste0(l, ".q"))) * t(paste0(l, ".pds"))
      k <- split(lin(h, paste0(l, ".k")))
      v <- split(lin(h, paste0(l, ".v")))
      # rotary on the leading half of each head, then the xPos decay
      q <- zulm::zl_rope(q, input$pos, half) * input$xq
      k <- zulm::zl_rope(k, input$pos, half) * input$xk
      perm <- function(v) zulm::zl_permute(v, c(1, 3, 2, 4))  # c(width, n, heads, b)
      s <- zulm::zl_crossprod(perm(k), perm(q))              # c(n_k, n_q, heads, b)
      a <- zulm::zl_softmax(s, mask = input$mask, scale = 1 / width)
      o <- zulm::zl_crossprod(zulm::zl_permute(perm(v), c(2, 1, 3, 4)), a)  # c(vw, n, heads, b)
      attended <- zulm::zl_reshape(zulm::zl_permute(o, c(1, 3, 2, 4)), c(heads * vw, n, b))
    }
    x <- residual(lin(attended, paste0(l, ".o")), x, taus[["attn"]])
    h <- zulm::zl_rms_norm(x, eps)
    hidden <- lin(h, paste0(l, ".ff_value"), bias = FALSE) *
      zulm::zl_silu(lin(h, paste0(l, ".ff_gate"), bias = FALSE))
    residual(lin(hidden, paste0(l, ".ff2"), bias = FALSE), x, taus[["mlp"]])
  }
  function(input) {
    x <- mlp(input$patches, "patch_proj", readout = FALSE)
    for (idx in seq_len(as.integer(config$num_layers)) - 1L) {
      x <- layer(x, sprintf("layers.%d", idx), toto_is_variate_layer(idx, config), input)
    }
    x <- zulm::zl_rms_norm(x, eps)
    d <- dim(x)
    anchor <- zulm::zl_get_rows(zulm::zl_reshape(x, c(d[[1]], d[[2]] * d[[3]])), input$anchor)
    list(quantiles = mlp(anchor, "output_head", readout = TRUE))
  }
}

# xPos factors c(width, 1, P): the leading half decays by position, the
# rotary-free half is left alone; `exponent` is +1 for queries, -1 for keys.
toto_zulm_xpos <- function(n, width, exponent, scale_base = 256) {
  half <- width %/% 2L
  base <- (seq(0, half - 1, by = 2) + 0.4 * half) / (1.4 * half)
  pos <- seq_len(n) - 1
  centre <- floor((max(pos) + 1) / 2)
  power <- (pos - centre) / scale_base
  s <- outer(rep(base, each = 2), power, `^`)^exponent      # c(half, n)
  array(rbind(s, matrix(1, width - half, n)), c(width, 1L, n))
}

# The causal scaler of toto-inference.R on R doubles: rows are series.
toto_zulm_scaler <- function(values, observed, patch, minimum_scale = 1e-6, correction = 1) {
  keep <- observed * 1
  cum <- function(m) t(apply(m, 1L, cumsum))
  counts <- pmax(cum(keep), 1)
  location <- cum(values * keep) / counts
  previous <- cbind(0, location[, -ncol(location), drop = FALSE])
  increment <- (values - previous) * (values - location) * keep
  variance <- cum(increment) / pmax(counts - correction, 1)
  scale <- pmax(sqrt(variance), minimum_scale)
  last <- function(m) {
    idx <- rep(seq(patch, ncol(m), by = patch), each = patch)
    m[, idx, drop = FALSE]
  }
  list(loc = last(location), scale = last(scale))
}

toto_zulm_predict_batch <- function(runtime, contexts, horizons, quantile_levels, config) {
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
  zuk_check_user_interrupt()
  usable <- as.integer(config$context_length %||% 4096L)
  contexts <- lapply(contexts, function(ctx) utils::tail(as.numeric(ctx), usable))
  width <- toto_round_up(max(vapply(contexts, length, integer(1))), patch)
  b <- length(contexts)
  values <- matrix(0, b, width)
  observed <- matrix(FALSE, b, width)
  for (i in seq_len(b)) {
    at <- seq.int(width - length(contexts[[i]]) + 1L, width)
    values[i, at] <- contexts[[i]]
    observed[i, at] <- TRUE
  }
  observed[, seq.int(width - patch + 1L, width)] <- TRUE
  # the reference reads the context in float32
  values <- zulm_f32(values)

  horizon <- max(horizons)
  n_patches <- as.integer(ceiling(horizon / patch))
  predicted <- n_patches * patch
  full_values <- cbind(values, matrix(0, b, predicted))
  full_observed <- cbind(observed, matrix(FALSE, b, predicted))
  stats <- toto_zulm_scaler(full_values, full_observed, patch)
  stats <- lapply(stats, zulm_f32)
  scaled <- ifelse(full_observed, asinh(zulm_f32((full_values - stats$loc) / stats$scale)), 0)

  total <- ncol(full_values) %/% patch
  # patches c(2 * patch, P, B): the scaled values, then 1 where unobserved
  patches <- array(0, c(2L * patch, total, b))
  for (i in seq_len(b)) {
    patches[seq_len(patch), , i] <- matrix(scaled[i, ], patch)
    patches[patch + seq_len(patch), , i] <- matrix(as.numeric(!full_observed[i, ]), patch)
  }
  context_patches <- width %/% patch
  mask <- array(0, c(total, total, 1L, b))
  for (i in seq_len(b)) {
    seen <- colSums(matrix(observed[i, ], patch)) > 0
    valid <- c(seen, rep(TRUE, n_patches))
    allowed <- outer(seq_len(total), seq_len(total), `<=`) & outer(valid, valid, `==`)
    mask[, , 1L, i] <- ifelse(allowed, 0, -Inf)
  }
  qk <- as.integer(config$qk_dim)
  anchor <- as.integer(rep((seq_len(b) - 1L) * total, each = n_patches) +
                         rep(total - n_patches - 1L + seq_len(n_patches) - 1L, b))
  out <- zulm::zl_forward(runtime$cache, list(
    patches = patches,
    mask = mask,
    pos = seq_len(total) - 1L,
    xq = toto_zulm_xpos(total, qk, 1),
    xk = toto_zulm_xpos(total, qk, -1),
    anchor = anchor
  ))$quantiles                                  # c(patch * Q, n_patches * B)
  nq <- length(trained)
  block <- array(out, c(nq, patch, n_patches, b))   # [q, p, patch index, series]
  lapply(seq_len(b), function(i) {
    tail_at <- width + seq_len(predicted)
    loc <- stats$loc[i, tail_at]
    sc <- stats$scale[i, tail_at]
    real <- t(matrix(block[, , , i], nq)) # rows: time steps, columns: levels
    real <- sinh(real) * sc + loc
    real <- t(apply(real, 1L, function(r) {
      fin <- is.finite(r)
      if (!all(fin)) {
        r[r == Inf] <- if (any(fin)) max(r[fin]) else Inf
        r[r == -Inf] <- if (any(fin)) min(r[fin]) else -Inf
      }
      sort(r)
    }))
    real[seq_len(horizons[[i]]), matched, drop = FALSE]
  })
}

zulm_f32 <- function(x) {
  d <- dim(x)
  y <- readBin(writeBin(as.double(x), raw(), size = 4), "double", n = length(x), size = 4)
  if (!is.null(d)) dim(y) <- d
  y
}

toto_zulm_runtime <- function(config, weights) {
  store <- toto_zulm_store(weights, config)
  build <- toto_zulm_build(store, config, NULL)
  list(store = store, cache = zulm::zl_graph_cache(build, store$params))
}
