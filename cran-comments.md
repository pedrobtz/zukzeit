## Submission

This is a new submission.

`zukzeit` runs open time-series foundation models natively in R, through R
`torch`, with no Python at runtime. It loads pretrained checkpoints from a
curated, revision-pinned catalogue and returns predictive quantiles through one
capability-aware interface.

## R CMD check results

0 errors | 0 warnings | 1 note

The note is "New submission".

Checked from the built source package with vignettes enabled and all declared
Imports and Suggests installed.

## Test environments

* Local: macOS, R release.
* GitHub Actions: Ubuntu (R-devel, release, oldrel-1), macOS release, Windows
  release.

## Package size and runtime dependency

`torch` is an Import, but the LibTorch runtime it needs is downloaded
separately by `torch::install_torch()` and is not present on a CRAN check
machine. Every test and example that would execute a tensor skips when that
runtime is absent, so the suite is expected to skip rather than fail there. The
CI matrix deliberately runs both configurations: most jobs without LibTorch,
which is what CRAN sees, and one with it, so the native code is genuinely
exercised.

## Network use

Nothing in the check run contacts the network.

* No example, test, or vignette downloads model weights during `R CMD check`.
* `zuk_models()` and `zuk_cache_status()` report whether a checkpoint is
  already on disk. They probe the local cache with `local_files_only = TRUE`
  and create nothing, so a check run under a fresh `HOME` leaves it empty.
* `zuk_download()` is the only function that transfers weights. It is the only
  `\dontrun{}` in the package, because the smallest supported checkpoint is
  16 MB and the largest is 925 MB.
* Numerical parity against the pinned upstream references is opt-in behind the
  `ZUK_RUN_CHECKPOINT_TEST` environment variable, which names the architectures
  to exercise. All three supported checkpoints pass their golden fixtures on
  CPU within the recorded tolerance, plus contract conformance against the real
  handle. Those runs are outside `R CMD check`.

## Examples

Every exported function has a runnable example. They use the weight-free `stub`
fixture, so no example downloads anything, contacts the network, or needs the
LibTorch runtime.

## Bundled third-party code

The package contains R translations of three Apache-2.0 reference
implementations: TimesFM (Google LLC), Toto 2.0 (Datadog, Inc.), and Chronos-2
(Amazon.com, Inc.). Each copyright holder is credited with a `cph` role in
`Authors@R`, the derived files carry prominent modification notices, and full
provenance including pinned upstream revisions is installed in `COPYRIGHTS`.
The complete Apache-2.0 text is installed in `LICENSES/Apache-2.0.txt`. The
package's own code is MIT.

## Method references

The package implements published architectures rather than novel methods. All
three are cited in the `Description` field: TimesFM
<doi:10.48550/arXiv.2310.10688>, Toto 2.0 <doi:10.48550/arXiv.2605.20119>, and
Chronos-2 <doi:10.48550/arXiv.2510.15821>.

## Spelling

The package uses British spelling throughout and declares `Language: en-GB`.
Domain terms and proper nouns are listed in `inst/WORDLIST`.
