# LLM Translate

Python toolkit for survey translation using Large Language Models (OpenAI, Gemini) and translation quality analysis via multilingual embeddings. Includes TRAPD/ISPOR-aligned prompt templates, back-translation, and reconciliation.

## Features
- Forward translation, optional blind back-translation, and reconciliation
- Batch translation mode with JSON outputs (forward/back/recon)
- TRAPD/ISPOR-aligned default prompts; override via external YAML
- Multiple providers: OpenAI (GPT) and Google Gemini (2.0/2.5 families)
- Analysis: ZCA whitening, Procrustes alignment, KL≈½·||Δ||², stats, visuals
- CLI scripts + example configs + sample data

## Install
```bash
pip install -r requirements.txt
```

Set API keys (env or config):
```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

## Repository Structure
```
llm_translate/
├── llm_translate/            # Package code
│   ├── translation/          # LLM translation logic
│   ├── analysis/             # Embedding + metrics + stats + visuals
│   ├── config/               # YAML loading + validation
│   └── utils/                # Helpers, IO, logging
├── scripts/                  # CLI scripts
│   ├── translate.py
│   └── analyze.py
├── configs/                  # Configs and prompt templates
│   ├── translation_config.yaml
│   ├── analysis_config.yaml
│   └── prompts_template.yaml
├── data/                     # Sample input data
├── outputs/                  # Script outputs (gitignored)
├── notebook/                 # Jupyter notebooks (usage demo)
├── requirements.txt
└── LICENSE                   # MIT
```

## Usage
### 1) Translation
Edit `configs/translation_config.yaml`:
```yaml
model: "gemini-2.5-flash"
source_lang: "en"
target_lang: "ja"
input_file: "data/sample_items.csv"
output_file: "outputs/translations_with_back_and_recon.csv"

do_back: true
do_recon: true
batch_mode: false
batch_size: 20

# Option A: inline prompts (optional)
# forward_prompt: |
#   ...
# back_prompt: |
#   ...
# recon_prompt: |
#   ...

# Batch prompt overrides (used only when batch_mode is true)
# batch_forward_prompt: |
#   ...
# batch_back_prompt: |
#   ...
# batch_recon_prompt: |
#   ...

# Option B: external prompt file (recommended)
prompt_file: "configs/prompts.yaml"
```
Run:
```bash
python scripts/translate.py --config configs/translation_config.yaml
```
Output CSV columns:
- `forward`; optional `back`; optional `reconciled`, `recon_explanation`

### 2) Batch Translation Mode
Enable batch mode to translate multiple items per LLM call and return JSON:
```yaml
batch_mode: true
batch_size: 25
```
When `batch_mode` is enabled, the script uses the batch prompt keys (`batch_*_prompt`) and expects JSON arrays keyed by `item_number`.

### 3) External Prompt File
Use the provided template and customize:
```bash
cp configs/prompts_template.yaml configs/prompts.yaml
# edit configs/prompts.yaml
```
Then set `prompt_file` in the translation config.

### 4) Analysis
```bash
python scripts/analyze.py --config configs/analysis_config.yaml
```
Generates metrics tables and figures under `outputs/`.

### 5) Jupyter Notebook (Translation Walkthrough)
A usage notebook is provided at `notebook/Translate_Usage.ipynb` showing:
- Loading config and optional external prompts
- Running forward/back/recon programmatically
- Saving output CSV

## Functional Updates
- Added batch translation mode with JSON outputs and batch prompt overrides.

## Attribution
- This Python implementation replicates and adapts concepts and prompt designs from the R package `LLMTranslate` by Jonas R. Kunst. See the CRAN page: https://CRAN.R-project.org/package=LLMTranslate
- License: MIT (this repository). Additional attribution included in `LICENSE`.

## Troubleshooting
- 404 on Gemini: ensure your model name is supported (see `scripts/list_gemini_models.py`).
- API errors: verify keys and quotas; you can switch providers/models in config.
- CSV commas: sample files are quoted; use `scripts/validate_csv.py` to fix your CSVs.

## License
MIT License. See `LICENSE`.
