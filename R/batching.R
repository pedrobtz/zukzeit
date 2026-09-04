# Batched panel inference and device placement.
#
# This module is the single choke point through which every architecture is
# executed, so context truncation, batching, and device selection live in one
# place rather than being re-implemented per model. The stub runs series-by-
# series via `predict_fn`; native torch architectures supply a vectorised
# `predict_batch_fn` and are fed in chunks of `batch_size`.

# Default batch size, overridable with `options(zuk.batch_size = ...)`.
zuk_default_batch_size <- function() {
  as.integer(getOption("zuk.batch_size", 64L))
}

# Validate a device string. Accepts "auto", "cpu", "mps", "cuda", and indexed
# CUDA devices like "cuda:1".
is_valid_device <- function(device) {
  length(device) == 1L && is.character(device) && !is.na(device) &&
    grepl("^(auto|cpu|mps|cuda(:[0-9]+)?)$", device)
}

#' Resolve a compute device
#'
#' Returns a concrete device string for torch models. `"auto"` (the default,
#' overridable with `options(zuk.device = ...)`) picks CUDA, then MPS, then CPU,
#' based on what the installed torch backend reports, and warns if a requested
#' accelerator is unavailable, falling back to CPU. Without torch it is always
#' `"cpu"`, so the stub path never touches torch.
#'
#' @param device One of `"auto"`, `"cpu"`, `"mps"`, `"cuda"`, `"cuda:N"`, or
#'   `NULL` to read the `zuk.device` option.
#' @return A concrete device string (never `"auto"`).
#' @export
#' @examples
#' zuk_resolve_device("cpu")
#'
#' # "auto" picks the best backend the installed torch reports.
#' zuk_resolve_device("auto")
zuk_resolve_device <- function(device = NULL) {
  device <- device %||% getOption("zuk.device", "auto")
  if (!is_valid_device(device)) {
    zuk_abort_device(c(
      "Invalid {.arg device}: {.val {device}}.",
      "i" = "Use one of {.val auto}, {.val cpu}, {.val mps}, {.val cuda}, or {.val cuda:N}."
    ), requested_device = device)
  }
  if (!identical(device, "auto")) {
    return(validate_available_device(device))
  }
  if (!requireNamespace("torch", quietly = TRUE)) {
    return("cpu")
  }
  if (isTRUE(tryCatch(torch::cuda_is_available(), error = function(e) FALSE))) {
    return("cuda")
  }
  if (isTRUE(tryCatch(torch::backends_mps_is_available(), error = function(e) FALSE))) {
    return("mps")
  }
  "cpu"
}

# For an explicitly requested accelerator, confirm the backend actually has it;
# warn and fall back to CPU rather than erroring deep in a forward pass.
validate_available_device <- function(device) {
  backend <- sub(":.*$", "", device)
  if (backend == "cpu" || !requireNamespace("torch", quietly = TRUE)) {
    return(device)
  }
  available <- switch(
    backend,
    cuda = isTRUE(tryCatch(torch::cuda_is_available(), error = function(e) FALSE)),
    mps  = isTRUE(tryCatch(torch::backends_mps_is_available(), error = function(e) FALSE)),
    TRUE
  )
  if (!available) {
    cli::cli_warn(c(
      "Requested device {.val {device}} is not available; falling back to {.val cpu}.",
      "i" = "Check your torch installation and drivers."
    ))
    return("cpu")
  }
  device
}

#' Set or get the default zukzeit compute device
#'
#' Thin wrapper over the `zuk.device` option consulted by
#' [zuk_resolve_device()].
#'
#' @param device A device string, or `NULL` to only read the current setting.
#' @return Invisibly, the previous option value.
#' @export
#' @examples
#' previous <- zuk_set_device("cpu")
#' zuk_resolve_device()
#'
#' # Restore whatever was configured before.
#' zuk_set_device(previous)
zuk_set_device <- function(device) {
  if (!is_valid_device(device)) {
    zuk_abort_device(
      "Invalid {.arg device}: {.val {device}}.",
      requested_device = device
    )
  }
  old <- getOption("zuk.device", "auto")
  options(zuk.device = device)
  invisible(old)
}

# Move a torch tensor or module to a device. No-op fallback keeps non-torch
# code paths (the stub) working; native architectures call this in their
# forward pass.
zuk_to_device <- function(x, device) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    return(x)
  }
  x$to(device = torch::torch_device(device))
}

# An explicit, testable interrupt boundary between units of work.
#
# This does not itself poll for an interrupt: R delivers pending interrupts at
# its own evaluation-loop boundaries, which is what makes a long batched run
# interruptible between chunks. A single torch call is not interruptible by
# either mechanism. What this function guarantees is the boundary's *placement*
# --- and the option hook lets tests drive a condition through it deterministically,
# to prove interrupts are neither swallowed nor rewrapped as engine errors and
# that no partial forecast escapes.
zuk_check_user_interrupt <- function() {
  hook <- getOption("zuk.interrupt_check", NULL)
  if (is.function(hook)) hook()
  invisible(NULL)
}

validate_batch_contexts <- function(contexts, model, call = rlang::caller_env()) {
  if (!is.list(contexts)) {
    zuk_abort_capability(
      "{.arg contexts} must be a list of numeric history vectors.",
      model_id = model$model_id,
      revision = model$revision,
      capability = "context",
      requested = class(contexts),
      supported = "list of finite numeric vectors",
      call = call
    )
  }
  lapply(seq_along(contexts), function(i) {
    ctx <- contexts[[i]]
    if (!is.numeric(ctx) || !is.null(dim(ctx))) {
      zuk_abort_capability(
        "Context {i} must be a numeric vector.",
        model_id = model$model_id,
        revision = model$revision,
        capability = "context",
        requested = class(ctx),
        supported = "finite numeric vector",
        call = call
      )
    }
    ctx <- as.numeric(ctx)
    if (any(is.infinite(ctx))) {
      zuk_abort_capability(
        "Context {i} contains an infinite value.",
        model_id = model$model_id,
        revision = model$revision,
        capability = "context",
        requested = ctx,
        supported = "finite observations; NA values are omitted",
        call = call
      )
    }
    ctx <- ctx[!is.na(ctx)]
    if (!length(ctx)) {
      zuk_abort_capability(
        "Context {i} has no observed values.",
        model_id = model$model_id,
        revision = model$revision,
        capability = "context",
        requested = ctx,
        supported = "at least one finite observation",
        call = call
      )
    }
    if (length(ctx) > model$capabilities$max_context) {
      ctx <- utils::tail(ctx, model$capabilities$max_context)
    }
    ctx
  })
}

validate_quantile_matrix <- function(x, h, quantile_levels, model,
                                     call = rlang::caller_env()) {
  expected <- c(as.integer(h), length(quantile_levels))
  if (!is.matrix(x) || !is.numeric(x) || !identical(dim(x), expected)) {
    actual <- if (is.matrix(x)) dim(x) else class(x)
    zuk_abort_contract(
      c(
        "An architecture returned an invalid forecast matrix.",
        "x" = "Expected a numeric matrix with dimensions {.val {expected}}.",
        "i" = "Received {.val {actual}}."
      ),
      architecture = model$architecture,
      model_id = model$model_id,
      contract = "predict return shape",
      expected = expected,
      actual = actual,
      call = call
    )
  }
  if (any(!is.finite(x))) {
    zuk_abort_contract(
      "An architecture returned non-finite forecast values.",
      architecture = model$architecture,
      model_id = model$model_id,
      contract = "finite forecasts",
      expected = "all finite values",
      actual = sum(!is.finite(x)),
      call = call
    )
  }
  if (ncol(x) > 1L && any(apply(x, 1L, function(row) any(diff(row) < 0)))) {
    zuk_abort_contract(
      "An architecture returned crossing predictive quantiles.",
      architecture = model$architecture,
      model_id = model$model_id,
      contract = "monotone quantiles",
      expected = "non-decreasing values across quantile levels",
      actual = x,
      call = call
    )
  }
  x
}

#' Run a model over many series in batches
#'
#' Truncates each context to the model's `max_context`, then evaluates the
#' model. Returns a list, aligned to `contexts`, of `h x length(quantile_levels)`
#' predictive-quantile matrices.
#'
#' @param model A `zuk_model`.
#' @param contexts A list of numeric context vectors (oldest first).
#' @param horizons A list/vector of per-series integer horizons.
#' @param quantile_levels Numeric vector of quantile levels.
#' @param batch_size Series per batch for vectorised models; defaults to
#'   `getOption("zuk.batch_size", 64L)`.
#' @param device Device string; resolved via [zuk_resolve_device()].
#' @return A list of quantile matrices.
#' @export
#' @examples
#' model <- zuk_pretrained("stub")
#'
#' # Two series of different lengths, each with its own horizon.
#' quantiles <- zuk_run_batches(
#'   model,
#'   contexts = list(cumsum(rep(1, 40)), cumsum(rep(2, 25))),
#'   horizons = c(3L, 5L),
#'   quantile_levels = c(0.1, 0.5, 0.9)
#' )
#' lengths(quantiles)
#' quantiles[[1]]
#'
#' zuk_unload("stub")
zuk_run_batches <- function(model, contexts, horizons, quantile_levels,
                             batch_size = NULL, device = NULL) {
  if (!inherits(model, "zuk_model")) {
    zuk_abort_contract(
      "{.arg model} must be a {.cls zuk_model}.",
      contract = "engine input",
      expected = "zuk_model",
      actual = class(model)
    )
  }
  caps <- model$capabilities
  contexts <- validate_batch_contexts(contexts, model)
  raw_horizons <- unlist(horizons, use.names = FALSE)
  n <- length(contexts)
  if (n == 0L) {
    return(list())
  }
  if (length(raw_horizons) != n) {
    zuk_abort_capability(
      "{.arg horizons} must contain one value per context.",
      model_id = model$model_id,
      revision = model$revision,
      capability = "horizon",
      requested = length(raw_horizons),
      supported = n
    )
  }
  check_horizon(caps, raw_horizons, model$model_id, model$revision)
  horizons <- as.integer(raw_horizons)
  quantile_levels <- check_quantile_levels(
    caps, quantile_levels, model$model_id, model$revision
  )
  batch_size <- batch_size %||% zuk_default_batch_size()
  if (length(batch_size) != 1L || !is.numeric(batch_size) || is.na(batch_size) ||
      !is.finite(batch_size) || batch_size <= 0 || batch_size != floor(batch_size) ||
      batch_size > .Machine$integer.max) {
    zuk_abort_capability(
      "{.arg batch_size} must be one positive integer.",
      model_id = model$model_id,
      revision = model$revision,
      capability = "batch_size",
      requested = batch_size,
      supported = "one positive integer"
    )
  }
  batch_size <- as.integer(batch_size)
  device <- zuk_resolve_device(device %||% model$device)

  if (is.function(model$predict_batch_fn)) {
    out <- vector("list", n)
    groups <- split(seq_len(n), (seq_len(n) - 1L) %/% batch_size)
    for (g in groups) {
      zuk_check_user_interrupt()
      res <- model$predict_batch_fn(contexts[g], horizons[g], quantile_levels,
                                    device = device)
      if (!is.list(res) || length(res) != length(g)) {
        zuk_abort_contract(
          "The model's batch function returned an invalid result collection.",
          architecture = model$architecture,
          model_id = model$model_id,
          contract = "predict_batch return length",
          expected = length(g),
          actual = if (is.list(res)) length(res) else class(res)
        )
      }
      res <- lapply(seq_along(g), function(i) {
        validate_quantile_matrix(
          res[[i]], horizons[[g[[i]]]], quantile_levels, model
        )
      })
      out[g] <- res
    }
    out
  } else {
    lapply(seq_len(n), function(i) {
      zuk_check_user_interrupt()
      validate_quantile_matrix(
        model$predict_fn(contexts[[i]], horizons[[i]], quantile_levels),
        horizons[[i]], quantile_levels, model
      )
    })
  }
}
