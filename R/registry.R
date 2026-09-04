# The architecture registry maps an architecture key (as found in a
# checkpoint's config.json) to a constructor that turns a parsed config plus a
# weight state-dict into a `zuk_model`. It is deliberately separate from the
# package-owned checkpoint catalogue: registering a constructor never adds a
# model ID or changes its support state.

.zuk_registry <- new.env(parent = emptyenv())

#' Register a model architecture
#'
#' Associates an architecture key with a constructor. The constructor must have
#' the signature `function(config, weights)` and return a [new_zuk_model()].
#'
#' Registration changes only the current R session's constructor mapping. It
#' does not add a checkpoint to [zuk_models()] or make an arbitrary model ID
#' loadable through [zuk_pretrained()]. The loader consults this registry only
#' after resolving a checkpoint from `zukzeit`'s package-owned curated catalogue.
#' Architecture authors can use [zuk_check_architecture()] independently;
#' catalogue inclusion requires the checkpoint's metadata, immutable revision,
#' licence, and numerical-parity evidence to be reviewed in `zukzeit`.
#'
#' @param architecture Character scalar key (matched against `config.json`).
#' @param constructor A function `function(config, weights)`.
#' @param overwrite Logical; if `FALSE` (default) re-registering an existing key
#'   errors.
#' @return `zuk_register_arch()` invisibly returns the architecture key;
#'   `zuk_registry_has()` returns one logical value; and
#'   `zuk_registry_archs()` returns the registered keys as a character vector.
#' @export
#' @examples
#' zuk_registry_has("timesfm")
#' zuk_registry_archs()
zuk_register_arch <- function(architecture, constructor, overwrite = FALSE) {
  architecture <- as.character(architecture)
  if (length(architecture) != 1L || is.na(architecture) || !nzchar(architecture)) {
    zuk_abort_capability(
      "{.arg architecture} must be one non-empty registry key.",
      capability = "architecture_registration",
      requested = architecture,
      supported = "one non-empty string"
    )
  }
  if (!is.function(constructor)) {
    zuk_abort_contract(
      "{.arg constructor} must be a function of {.code (config, weights)}.",
      architecture = architecture,
      contract = "architecture registration",
      expected = "function(config, weights)",
      actual = class(constructor)
    )
  }
  if (!isTRUE(overwrite) && zuk_registry_has(architecture)) {
    zuk_abort_capability(c(
      "Architecture {.val {architecture}} is already registered.",
      "i" = "Pass {.code overwrite = TRUE} to replace it."
    ),
    capability = "architecture_registration",
    requested = architecture,
    supported = "an unregistered key or overwrite = TRUE")
  }
  assign(architecture, constructor, envir = .zuk_registry)
  invisible(architecture)
}

#' @rdname zuk_register_arch
#' @export
zuk_registry_has <- function(architecture) {
  architecture <- as.character(architecture)
  if (length(architecture) != 1L || is.na(architecture) || !nzchar(architecture)) {
    return(FALSE)
  }
  exists(architecture, envir = .zuk_registry, inherits = FALSE)
}

#' @rdname zuk_register_arch
#' @export
zuk_registry_archs <- function() {
  sort(ls(envir = .zuk_registry))
}

# Fetch a constructor, with a helpful error listing what *is* available.
zuk_registry_get <- function(architecture, call = rlang::caller_env()) {
  architecture <- as.character(architecture)
  if (!zuk_registry_has(architecture)) {
    available <- zuk_registry_archs()
    zuk_abort_capability(
      c(
        "No constructor registered for architecture {.val {architecture}}.",
        "i" = if (length(available)) {
          "Registered architectures: {.val {available}}."
        } else {
          "No architectures are registered."
        }
      ),
      capability = "architecture",
      requested = architecture,
      supported = available,
      call = call
    )
  }
  get(architecture, envir = .zuk_registry, inherits = FALSE)
}
