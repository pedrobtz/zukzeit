#!/usr/bin/env Rscript

# Emit the checkpoint cache key and download list for the tier named by
# `ZUK_RUN_CHECKPOINT_TEST`, as GitHub Actions step outputs.
#
# Both values are derived from the curated catalogue rather than written into
# the workflow, so bumping a pinned revision in R/catalogue.R invalidates the
# cache and changes what is prefetched without anyone editing YAML. The
# previous hand-maintained key named one checkpoint while three were being
# downloaded, which silently re-downloaded about 1.4 GB on every run.
#
# The tier string is parsed by the same helper the test suite uses, so CI and
# `testthat` can never disagree about which architectures a tier selects.

suppressMessages(pkgload::load_all(".", quiet = TRUE))
source("tests/testthat/helper-checkpoint.R")

models <- zuk_models()
selected <- models[vapply(models$architecture, checkpoint_tests_enabled, logical(1)), ]

if (!nrow(selected)) {
  stop(
    "ZUK_RUN_CHECKPOINT_TEST=", encodeString(Sys.getenv("ZUK_RUN_CHECKPOINT_TEST"), quote = "\""),
    " selects no supported checkpoint; the job would cache and prefetch nothing.",
    call. = FALSE
  )
}

# Ordered by architecture so the key is stable regardless of catalogue order,
# and truncated because a full 40-character SHA per checkpoint makes the key
# unreadable in the Actions UI without adding uniqueness that matters.
selected <- selected[order(selected$architecture), ]
key <- paste0(
  "hf-",
  paste(selected$architecture, substr(selected$revision, 1L, 12L),
        sep = "-", collapse = "-")
)

cat(sprintf("key=%s\n", key))
cat(sprintf("ids=%s\n", paste(selected$model_id, collapse = ",")))
