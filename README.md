# ManyScales

Analysis code and translation tools for comparing human and LLM translations of psychological scales.

**Paper:** Yamada et al. (2026), 「大規模言語モデルの翻訳を評価する大規模共同研究：22心理尺度の人手-機械翻訳間の比較」. [Jxiv preprint](https://jxiv.jst.go.jp/index.php/jxiv/preprint/view/2056) · [DOI](https://doi.org/10.51094/jxiv.2056).

**Data and study materials:** [LLMTranslate-Psych-Japan — OSF project](https://osf.io/hn4fr/). The analysis-data upload is pending. Data, workbooks, embeddings, and fitted-model files are kept outside GitHub; replication requires the corresponding OSF release.

## Repository

| Folder | Contents |
|---|---|
| [analysis](analysis/README.md) | H1–H4 analysis sources and tests |
| [translation](translation/README.md) | LLMTranslate and LLMTranslate4Hitode R applications |
| [reports](reports/README.md) | Rendered analysis reports |
| [figures](figures/README.md) | Manuscript figures |

## Replicate the analyses

1. Install Python 3.11+, R 4.2+, and Quarto. Create a Python environment from the repository root:

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e 'analysis/h4[dev,embeddings]'
   ```

2. Once the data release is uploaded to [OSF](https://osf.io/hn4fr/), download it outside this repository. Preserve its directory structure and attach the local files:

   ```sh
   python scripts/attach_local_data.py --source /path/to/osf-download
   ```

3. For H1–H3, install the R packages listed in the sources. H1/H3 additionally require `brms`, `cmdstanr`, and CmdStan; H2 uses `lavaan`, `semTools`, and `exametrika`. Run from the primary analysis directory:

   ```sh
   cd analysis/ratings/primary
   quarto render h1.qmd
   quarto render h2.qmd
   quarto render h3.qmd
   ```

4. For H4, run from `analysis/h4/`:

   ```sh
   python scripts/00_extract_scales.py
   python -m h4.pipeline --config configs/analysis_no_tivs.yaml
   ```

From the repository root, `make figures` redraws H1/H3 and synchronizes the H4 figure; `make reports` renders H4 and supplementary reports. Both require the matching local result tables. See [the replication guide](docs/reproduce.md) for input paths, supplementary models, exploratory fits, and coding rules.

Code checks use synthetic fixtures and require no study data:

```sh
python scripts/check.py
(cd analysis/h4 && python -m pytest -q)
```

## Cite and reuse

Please cite the [ManyScales paper](https://doi.org/10.51094/jxiv.2056) and [OSF project](https://osf.io/hn4fr/) when using this work. [CITATION.cff](CITATION.cff) and [BibTeX entries](docs/references.bib) are provided.

Software is distributed under one [MIT license](LICENSE), retaining the upstream and project copyright notices. Scale materials retain their original rights. [CONTRIBUTORS.md](CONTRIBUTORS.md) records project credits.
