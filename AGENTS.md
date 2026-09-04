# Repository Guidelines

## Project Structure & Module Organization

`zukzeit` is an R package. Source files live in `R/`: engine contracts and model
handles are in `contract.R`, `model.R`, and `capabilities.R`; loading and
dispatch are in `hub.R` and `registry.R`; execution is in `batching.R`; model
ports use `arch-<name>.R`. Keep framework integrations isolated in files such
as `forecast.R`, `fit.R`, and `parsnip.R`.

Tests live in `tests/testthat/` and follow the source area they exercise.
Reference outputs belong under `tests/testthat/fixtures/<architecture>/`.
Long-form examples live in `vignettes/`; release scope and design decisions are
tracked in `.agents/roadmap.md` and `.agents/plan.md`, and the surface downstream
packages depend on is specified in `.agents/consumer-api.md`.

## Build, Test, and Development Commands

Run commands from the repository root:

```sh
Rscript -e 'devtools::document()'              # regenerate man/ and NAMESPACE
Rscript -e 'testthat::test_local(".")'         # run the testthat suite
Rscript -e 'devtools::check()'                 # local package check
R CMD build --no-build-vignettes .             # create a source tarball
R CMD check --as-cran zuk_*.tar.gz             # release-oriented check
```

Install dependencies with `pak::pak()` or `devtools::install_deps()`. Model
downloads and Python reference implementations must not be required by normal
CI tests.

## Coding Style & Naming Conventions

Use two-space indentation, `<-` assignment, `snake_case` functions and
variables, and one concern per R file. Name S3 methods `generic.class` and model
modules `arch-<architecture>.R`. Public functions require roxygen2 comments and
`@export`; do not hand-edit generated `NAMESPACE` or `.Rd` files. Use `cli` for
actionable errors and `rlang::check_installed()` for optional dependencies.

Capability metadata must describe executable behavior, not planned features.
Keep adapters in `Suggests` and preserve the framework-neutral core.

## Testing Guidelines

The suite uses testthat edition 3. Name files `test-<area>.R` and write focused
`test_that()` cases. Every native model needs two independent gates:
`zuk_check_architecture()` conformance and numerical parity against committed,
revision-pinned reference fixtures. Tests must be deterministic, network-safe,
and explicit when optional integrations are skipped.

## Commit & Pull Request Guidelines

History uses short, imperative, sentence-case subjects, optionally stage-scoped,
for example `Fix CI failures: ...` or `Stage 1: ...`. Keep commits cohesive.
Pull requests should explain the user-visible outcome, architecture/contract
impact, tests run, skipped tests, checkpoint revision and licence when relevant,
and any roadmap or documentation updates. Do not claim model support until both
conformance and parity gates pass.
