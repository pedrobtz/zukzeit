## R CMD check results

Local check on 2026-09-04:

0 errors | 0 warnings | 2 notes

Checked with R 4.6.1 on macOS 26.6.2 (aarch64-apple-darwin23) from the built
source package, with vignettes enabled and all declared Imports and Suggests
installed.

The notes identify this as a new submission and report that the local HTML Tidy
installation is too old for HTML validation. The package itself produces no
warning or significant diagnostic.

A second source-package check with `_R_CHECK_DEPENDS_ONLY_=true` reports
`Status: OK`, confirming that the package installs, its examples run, and its
tests pass with only the declared Imports present.

These check results validate the implemented TimesFM baseline. They are not
submission results for the expanded `0.1.0` scope; a fresh full matrix and
checkpoint-backed parity jobs for every release model are required before this
file is finalized for CRAN.

## Test scope

* The deterministic suite is network-free and passes locally. Nothing in it
  downloads weights, contacts the Hub, or requires Python.
* `torch` is an Import, but the LibTorch runtime it needs is downloaded
  separately by `torch::install_torch()`. Tests that execute tensors skip when
  that runtime is absent, which is expected on a CRAN check machine. The CI
  matrix exercises both runtime-absent and runtime-installed configurations.
* Numerical parity against the pinned TimesFM reference, and every test that
  loads the 925 MB checkpoint, are opt-in behind
  `ZUK_RUN_CHECKPOINT_TEST=true`. They were run locally against the Hub cache
  and pass: five golden fixtures on CPU within the recorded `atol`/`rtol`
  budget, contract conformance against the real handle, and the four documented
  user workflows end to end.

## Examples and vignettes

* Every exported function has a runnable example. They use the weight-free
  `stub` fixture, so no example downloads anything, contacts the Hub, or needs
  the LibTorch runtime. `zuk_download()` is the only `\dontrun{}`, because it
  transfers a 925 MB checkpoint.
* `zuk_models()` and `zuk_cache_status()` report whether a checkpoint is
  already on disk. They probe the `hfhub` cache with `local_files_only = TRUE`
  and create nothing: a check run under a fresh `HOME` leaves it empty.
* Two vignettes evaluate their code against the `stub` fixture and need no
  network. Chunks depending on suggested packages are guarded on
  `requireNamespace()`.
* The two TimesFM vignettes evaluate no chunks, because building them would
  download the 925 MB checkpoint during `R CMD check`. Their outputs were
  produced by running the code against the real checkpoint and pasted in.

## Release status

This is not yet the `0.1.0` release candidate. The release scope now requires
native, parity-certified Toto 2.0 4M and Chronos-2 implementations in addition
to TimesFM, followed by complete local and remote package validation.

TimesFM 2.5 is the first supported native architecture: it passes contract
conformance and numerical parity against the pinned reference implementation.
TTM remains a registered scaffold whose forward pass aborts by design, and
Chronos-2 remains unregistered and rejected before any network or tensor work
until its new native port is complete. The R TimesFM derivative retains Google
LLC's copyright notice and the upstream Apache-2.0 licence; detailed provenance
is installed in `COPYRIGHTS`. Known limitations are listed in the README.

## Method references

The package implements published architectures rather than novel methods. The
TimesFM reference is cited in the `Description` field as
<doi:10.48550/arXiv.2310.10688>; the vignettes carry the wider benchmark
literature.
