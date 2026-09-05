# Which checkpoint-backed tests may run.
#
# `ZUK_RUN_CHECKPOINT_TEST` was all-or-nothing, which meant the cheapest useful
# tier was also the most expensive one: exercising Toto's 16 MB checkpoint
# pulled TimesFM's 925 MB and Chronos-2's 478 MB with it. Naming architectures
# lets CI run real-checkpoint parity on every push for the small model and
# reserve the full 1.4 GB for main.
#
#   ZUK_RUN_CHECKPOINT_TEST=true            every checkpoint
#   ZUK_RUN_CHECKPOINT_TEST=toto2           just Toto
#   ZUK_RUN_CHECKPOINT_TEST=toto2,chronos2  those two
#   unset                                    none
checkpoint_tests_enabled <- function(architecture) {
  requested <- Sys.getenv("ZUK_RUN_CHECKPOINT_TEST")
  if (!nzchar(requested)) return(FALSE)
  if (identical(requested, "true")) return(TRUE)
  architecture %in% trimws(strsplit(requested, ",", fixed = TRUE)[[1]])
}

skip_unless_checkpoint <- function(architecture) {
  testthat::skip_if_not(
    checkpoint_tests_enabled(architecture),
    sprintf("Set ZUK_RUN_CHECKPOINT_TEST=%s to run %s checkpoint tests.",
            architecture, architecture)
  )
}
