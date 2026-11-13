# Translation workflows

The study used LLMTranslate for the LLM condition and LLMTranslate4Hitode for the human condition. Both R applications are included here; study materials are supplied separately through [OSF](https://osf.io/hn4fr/).

From the repository root:

```sh
Rscript translation/install.R
Rscript translation/check.R
Rscript translation/run.R llm
# or: Rscript translation/run.R human
```

Hitode accepts original English in Excel column A and fixed human Japanese in column B. It skips forward generation while retaining the original for comparison. It relies on internal helpers from the bundled LLMTranslate 0.3.0. API keys are entered locally; translation requests are submitted manually through the interface.

The survey used the **initial Japanese translations**, before reconciliation. See the [study procedure](../docs/translation.md), local `model_assignments.csv`, [materials](materials/README.md), and shared [MIT license](../LICENSE).
