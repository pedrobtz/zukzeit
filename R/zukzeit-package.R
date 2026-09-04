#' @keywords internal
"_PACKAGE"

## usethis namespace: start
## usethis namespace: end
NULL

# Internal null-coalescing helper (avoids taking a hard dep on rlang's `%||%`
# export surface, though rlang is imported).
`%||%` <- function(x, y) if (is.null(x)) y else x
