# `self` is bound by fabletools when it evaluates a model definition's
# specials, so it is legitimate at run time but invisible to static analysis.
utils::globalVariables(c("object", "new_data", "self"))

.onLoad <- function(libname, pkgname) {
  # Register constructors for the package-owned catalogue. Constructor
  # registration and checkpoint curation are deliberately separate.
  zuk_register_arch("stub", stub_constructor, overwrite = TRUE)
  zuk_register_arch("ttm", ttm_constructor, overwrite = TRUE)
  zuk_register_arch("timesfm", timesfm_constructor, overwrite = TRUE)
  zuk_register_arch("toto2", toto_constructor, overwrite = TRUE)

  # Chronos-2 is intentionally not registered for 0.1.0. The Brulee bridge is
  # kept as prior art in .agents/reference/arch-chronos2.R rather than shipped:
  # it is not a verified implementation of the architecture contract, so it must
  # not appear as an executable built-in model, and the package must not carry a
  # dependency for code nothing calls. zuk_pretrained() rejects Chronos-2 ids
  # before any network or tensor work; see is_chronos2_id().

  # Adapter methods are registered against generics owned by optional packages.
  # s3_register() installs a load hook, so the method lights up if and when the
  # adapter package is loaded, and costs nothing when it never is. This is what
  # lets zukzeit interoperate with fabletools without importing it — and without
  # defining a competing `as_fable()` generic.
  vctrs::s3_register("fabletools::as_fable", "zuk_forecast")
  # `model_sum()` is fabletools' own generic, so it can only be registered
  # lazily. `forecast()` needs no hook: it comes from generics, which zukzeit
  # imports and re-exports, and fabletools dispatches on that same generic.
  vctrs::s3_register("fabletools::model_sum", "model_tsfm")

  # Register the parsnip engine when parsnip is available, mirroring how
  # brulee/modeltime engines register lazily so the core package stays light.
  # Best-effort: registration must never block package load.
  if (requireNamespace("parsnip", quietly = TRUE)) {
    tryCatch(
      make_zuk_reg(),
      error = function(e) {
        cli::cli_warn(c(
          "Could not register the {.val zukzeit} parsnip engine.",
          "i" = "The engine will be unavailable; everything else still works.",
          "x" = conditionMessage(e)
        ))
      }
    )
  }
}
