## R CMD check results

Local check on 2026-08-23:

0 errors | 0 warnings | 2 notes

Checked with R 4.6.1 on macOS 26.5.2 (aarch64-apple-darwin23) from the built
source package, with vignettes enabled and all declared Imports and Suggests
installed.

The notes identify this as a new submission and report that the local HTML Tidy
installation is too old for HTML validation. The package itself produces no
warning or significant diagnostic.

A second source-package check with `_R_CHECK_DEPENDS_ONLY_=true` reports the
same 0 errors, 0 warnings, and 2 notes, confirming that the package installs and
its tests run with only the declared Imports present.

The latest published five-platform GitHub Actions matrix predates the local
removal of an undeclared optional integration and is not a release result. It
must be rerun on the release candidate before submission. The dedicated
checkpoint-backed numerical-parity job passed on that published run.

## Test scope

* The deterministic suite is network-free and passes locally. Nothing in it
  downloads weights, contacts the Hub, or requires Python.
* `torch` is an Import, but the LibTorch runtime it needs is downloaded
  separately by `torch::install_torch()`. Tests that execute tensors skip when
  that runtime is absent, which is expected on a CRAN check machine. The CI
  matrix exercises both runtime-absent and runtime-installed configurations.
* Numerical parity against the pinned TimesFM reference, and every test that
  loads the 925 MB checkpoint, are opt-in behind
  `TSFM_RUN_CHECKPOINT_TEST=true`. They were run locally against the Hub cache
  and pass: five golden fixtures on CPU within the recorded `atol`/`rtol`
  budget, contract conformance against the real handle, and the four documented
  user workflows end to end.
* Vignettes evaluate no chunks. Their outputs were produced by running the code
  against the real checkpoint and pasted in, so no vignette build downloads
  anything.

## Release status

This is the `0.1.0` release candidate. Submission remains gated on the complete
GitHub Actions matrix and checkpoint-backed parity job passing on this exact
candidate.

TimesFM 2.5 is the first supported native architecture: it passes both release
gates, contract conformance and numerical parity against the pinned reference
implementation. TTM remains a registered scaffold whose forward pass aborts by
design, and Chronos-2 is unregistered and rejected before any network or tensor
work. The R TimesFM derivative retains Google LLC's copyright notice and the
upstream Apache-2.0 licence; detailed provenance is installed in `COPYRIGHTS`.
Known limitations are listed in the README.
