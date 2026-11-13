# Replication guide

Study data and materials will be released through the [OSF project](https://osf.io/hn4fr/); the analysis-data upload is pending. GitHub contains code, rendered reports, and final figures. It excludes raw and processed data, numerical result tables, workbooks, embedding caches, and fitted-model objects.

## Local inputs

After downloading the OSF release outside the repository, run from the repository root:

```sh
python scripts/attach_local_data.py --source /path/to/osf-download
```

The helper attaches available files using the paths below, without copying data into Git. An existing coauthor download using `Without_Exclusion_Criteria_3` and `With_Exclusion_Criteria_3` is also supported. It never overwrites an existing destination.

| Local destination | Required contents |
|---|---|
| `analysis/ratings/primary/data/` | Raw, processed, and long-form H1–H3 survey inputs |
| `analysis/ratings/primary/results/` | H1–H3 aggregate summaries for figure export and saved-result checks |
| `analysis/ratings/primary/R/*.csv` | H2 item/factor keys, reference coefficients, and exploratory item selections |
| `analysis/ratings/sensitivity/data/` | Optional criterion-3 sensitivity inputs |
| `analysis/ratings/supplementary/results/` | Saved supplementary model summaries |
| `translation/materials/` | `ManyScales対象尺度.xlsx`, `尺度対応表.xlsx`, and the scale catalog |
| `analysis/scale_names.csv` | Approved abbreviations keyed by the original 1–22 IDs |
| `translation/model_assignments.csv` | Scale-level translation-model assignments |
| `analysis/h4/data/published/` | H4 aggregate tables, frozen token data, and projection coordinates |
| `analysis/h4/data/survey/` | Local respondent inputs for H4 exploration; optional questionnaire QSF files under `questionnaires/` |
| `figures/h1_ratings.csv`, `figures/h3_ratings.csv` | Saved plotted values for result verification |
| `figures/embeddings/*.csv` | Coordinates and centers for the three descriptive 3D views |

These are the code's expected paths, not a claim that the pending OSF release already contains every file. If the release uses a different layout, arrange the local inputs accordingly. All data formats and runtime directories are ignored by Git.

## H1–H3 and supplementary models

Install R 4.2+, Quarto, and the packages named in the sources. Primary H1/H3 use `brms` with `cmdstanr`/CmdStan; H2 uses `lavaan`, `semTools`, and `exametrika`. Supplementary models use `rstan`, `bridgesampling`, and the other packages loaded at the top of their scripts.

From `analysis/ratings/primary/`, render `h1.qmd`, `h2.qmd`, or `h3.qmd` with Quarto. These sources include full fitting chunks, which can take substantial time. Preprocessing sources are under `preparation/`.

For supplementary refits, work from `analysis/ratings/` and run `Rscript supplementary/R/run_h1rep_main.R` or `Rscript supplementary/R/run_h3rep_main.R` with the task arguments documented at the top of each script. Their `report.qmd` reads saved summaries.

## H4

From the repository root:

```sh
python -m pip install -e 'analysis/h4[dev,embeddings]'
cd analysis/h4
python scripts/00_extract_scales.py
python -m h4.pipeline --config configs/analysis_no_tivs.yaml
```

Model weights are downloaded separately. `configs/analysis_no_tivs.yaml` is the 21-scale analysis; `configs/analysis.yaml` retains scale 11 as a robustness check. H4 unit tests construct synthetic data in temporary directories and do not require the study download.

Exploratory respondent models require the local survey tables and cached item-similarity/token-distance tables. With the two-rule citizen cut and expert tables attached, run from `analysis/h4/`:

```sh
python scripts/prepare_own_level.py results/own_level/frames_n1859 data/survey/4_analysis_n1859/citizen_scale_judgment.parquet
Rscript scripts/fit_raw_outcome_models.R results/own_level/frames_n1859 results/own_level/models_n1859 data/published
```

`build_n1859.py` constructs the two-rule citizen cut from local flag tables when needed. The detailed R walkthrough is `notebooks/h4_replication.Rmd`.

## Figures and reports

From the repository root, after attaching matching saved results:

```sh
make figures
make reports
python scripts/redraw_embeddings.py
python scripts/check.py --results
```

Figure export requires `dplyr`, `ggplot2`, `ragg`, `svglite`, `systemfonts`, and a Japanese font. `make figures` preserves the H1/H3 posterior estimates and intervals. `make reports` renders both H4 reports and the supplementary report from local aggregates. It does not refit those models. Use `--report h4-ja`, `h4-en`, or `supplementary` with `scripts/render_reports.py` to render one report.

## Analysis conventions

- Primary H1/H3: human minus LLM, Bayesian 95% CrI. Supplementary models use LLM minus human and retain their own priors, HDI/CrI, BF, and ROPE definitions.
- H4: human minus LLM MPNet cosine gap, REML 95% CI. Exploratory rating and checkbox models are separate analyses.
- Primary analyses exclude scale ID 11; original IDs are retained.
- OCQ non-consumers (`C08_chocolate_frequency = 6`) are excluded only from OCQ responses. The main OCQ sample is 363 → 330; other scale responses remain, and the citizen total stays 1,859. The criterion-3 sensitivity sample is 1,211, with OCQ 198 → 192.
- Exported cultural-validity ratings already run from low to high validity; no additional reversal is applied. Cultural-problem checkboxes remain 0/1, with 1 meaning a reported problem.

The rendered H4 reports retain the corrected interpretations of cultural validity and response-option fit. The repository cleanup does not change their numerical results.
