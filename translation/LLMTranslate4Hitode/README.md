# LLMTranslate4Hitode

Shiny application supplied by **Yoshihiko Kunisato / 国里愛彦**, adapted from **Jonas R. Kunst's LLMTranslate**. The supplied `app.R` is retained without code changes; its hash is in [the source notes](../../docs/sources.md).

The app keeps both original English and an existing Japanese forward translation as fixed inputs. It skips the forward-translation call and performs only back-translation and reconciliation. This supports the study's human translation condition without losing the English original used for comparison.

Install dependencies from the repository root with `Rscript translation/install.R`, then run `Rscript translation/run.R human`. The Excel input requires original English in column A and existing Japanese translation in column B. The bundled LLMTranslate 0.3.0 supplies its internal helpers.

The survey used the human teams' original Japanese translation, not this app's reconciled output. Retain [the shared MIT notice](../../LICENSE) when redistributing this adaptation.
