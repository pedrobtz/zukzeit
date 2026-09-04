#' Structured zukzeit conditions
#'
#' Public engine failures inherit from `zuk_error`, one policy class, and one
#' leaf class. Consumers should branch on those classes and structured fields,
#' never on condition messages.
#'
#' The policy classes are:
#'
#' * `zuk_error_recoverable`: the request can be changed or another model can
#'   be selected.
#' * `zuk_error_external`: authentication, download, or network state may be
#'   retried or repaired outside the engine.
#' * `zuk_error_internal`: the checkpoint or architecture violates the engine
#'   contract and should not be retried unchanged.
#'
#' @section Leaf classes and fields:
#' Every condition also has the standard `message` and `call` fields.
#'
#' * `zuk_error_capability`: `model_id`, `revision`, `capability`, `requested`,
#'   and `supported`.
#' * `zuk_error_context_length`: `model_id`, `revision`, `requested`, and
#'   `supported`.
#' * `zuk_error_quantile_levels`: `model_id`, `revision`, `requested`, and
#'   `supported`.
#' * `zuk_error_device`: `requested_device` and `resolved_device`.
#' * `zuk_error_download`: `model_id`, `revision`, `file`, and the original
#'   `parent` condition when available.
#' * `zuk_error_checkpoint`: `model_id`, `revision`, `tensor`, `expected`, and
#'   `actual`.
#' * `zuk_error_contract`: `architecture`, `model_id`, `contract`, `expected`,
#'   and `actual`.
#'
#' Fields that do not apply to a particular failure may be `NA` or `NULL`.
#'
#' @name zuk-conditions
#' @aliases zuk_error zuk_error_recoverable zuk_error_external zuk_error_internal zuk_error_capability zuk_error_context_length zuk_error_quantile_levels zuk_error_device zuk_error_download zuk_error_checkpoint zuk_error_contract
#' @examples
#' error <- tryCatch(zuk_models("unknown"), error = identity)
#' inherits(error, "zuk_error_recoverable")
#' error$capability
NULL

# Consumers make policy decisions from the documented classes and fields.
# Messages remain for humans rather than control flow.

zuk_error_policies <- c(
  "zuk_error_recoverable",
  "zuk_error_external",
  "zuk_error_internal"
)

zuk_require_namespace <- function(package, reason = NULL,
                                   call = rlang::caller_env()) {
  missing <- package[!vapply(package, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    detail <- if (is.null(reason)) "" else paste0(" ", reason)
    zuk_abort_capability(
      c(
        "Required package{?s} {.pkg {missing}} {?is/are} not installed.",
        "i" = "Install {?it/them} with {.code install.packages({deparse(missing)})}.{detail}"
      ),
      capability = "package_dependency",
      requested = missing,
      supported = package,
      call = call
    )
  }
  invisible(package)
}

zuk_abort <- function(message, class, policy, ...,
                       call = rlang::caller_env(),
                       .envir = rlang::caller_env()) {
  stopifnot(
    length(class) == 1L,
    length(policy) == 1L,
    policy %in% zuk_error_policies
  )
  cli::cli_abort(
    message,
    class = c(class, policy, "zuk_error"),
    ...,
    call = call,
    .envir = .envir
  )
}

zuk_abort_capability <- function(message,
                                  model_id = NA_character_,
                                  revision = NA_character_,
                                  capability,
                                  requested = NULL,
                                  supported = NULL,
                                  call = rlang::caller_env(),
                                  .envir = rlang::caller_env()) {
  zuk_abort(
    message,
    class = "zuk_error_capability",
    policy = "zuk_error_recoverable",
    model_id = as.character(model_id),
    revision = as.character(revision),
    capability = as.character(capability),
    requested = requested,
    supported = supported,
    call = call,
    .envir = .envir
  )
}

zuk_abort_context_length <- function(message,
                                      model_id = NA_character_,
                                      revision = NA_character_,
                                      requested,
                                      supported,
                                      call = rlang::caller_env(),
                                      .envir = rlang::caller_env()) {
  zuk_abort(
    message,
    class = "zuk_error_context_length",
    policy = "zuk_error_recoverable",
    model_id = as.character(model_id),
    revision = as.character(revision),
    requested = requested,
    supported = supported,
    call = call,
    .envir = .envir
  )
}

zuk_abort_quantile_levels <- function(message,
                                       model_id = NA_character_,
                                       revision = NA_character_,
                                       requested,
                                       supported,
                                       call = rlang::caller_env(),
                                       .envir = rlang::caller_env()) {
  zuk_abort(
    message,
    class = "zuk_error_quantile_levels",
    policy = "zuk_error_recoverable",
    model_id = as.character(model_id),
    revision = as.character(revision),
    requested = requested,
    supported = supported,
    call = call,
    .envir = .envir
  )
}

zuk_abort_device <- function(message,
                              requested_device,
                              resolved_device = NA_character_,
                              call = rlang::caller_env(),
                              .envir = rlang::caller_env()) {
  zuk_abort(
    message,
    class = "zuk_error_device",
    policy = "zuk_error_recoverable",
    requested_device = as.character(requested_device),
    resolved_device = as.character(resolved_device),
    call = call,
    .envir = .envir
  )
}

zuk_abort_download <- function(message,
                                model_id,
                                revision,
                                file = NA_character_,
                                parent = NULL,
                                call = rlang::caller_env(),
                                .envir = rlang::caller_env()) {
  zuk_abort(
    message,
    class = "zuk_error_download",
    policy = "zuk_error_external",
    model_id = as.character(model_id),
    revision = as.character(revision),
    file = as.character(file),
    parent = parent,
    call = call,
    .envir = .envir
  )
}

zuk_abort_checkpoint <- function(message,
                                  model_id = NA_character_,
                                  revision = NA_character_,
                                  tensor = NA_character_,
                                  expected = NULL,
                                  actual = NULL,
                                  call = rlang::caller_env(),
                                  .envir = rlang::caller_env()) {
  zuk_abort(
    message,
    class = "zuk_error_checkpoint",
    policy = "zuk_error_internal",
    model_id = as.character(model_id),
    revision = as.character(revision),
    tensor = as.character(tensor),
    expected = expected,
    actual = actual,
    call = call,
    .envir = .envir
  )
}

zuk_abort_contract <- function(message,
                                architecture = NA_character_,
                                model_id = NA_character_,
                                contract = NA_character_,
                                expected = NULL,
                                actual = NULL,
                                call = rlang::caller_env(),
                                .envir = rlang::caller_env()) {
  zuk_abort(
    message,
    class = "zuk_error_contract",
    policy = "zuk_error_internal",
    architecture = as.character(architecture),
    model_id = as.character(model_id),
    contract = as.character(contract),
    expected = expected,
    actual = actual,
    call = call,
    .envir = .envir
  )
}
