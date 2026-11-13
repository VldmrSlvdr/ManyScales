
# LLMTranslate

<!-- badges: start -->
<!-- badges: end -->

**LLMTranslate** is an R package that wraps a Shiny application for TRAPD/ISPOR-style survey translation. It automates forward translation, optional back-translation, and reconciliation using large language models from multiple providers:

- **OpenAI**: GPT-4o, GPT-4.1, GPT-5 series, and o-series reasoning models
- **Google Gemini**: Gemini 2.5 Pro, 2.5 Flash, 2.0 Flash
- **Anthropic Claude**: Claude Sonnet 4.5, Claude Haiku 4.5, Claude Opus 4.1, and more

## Key Features (v0.3.0)

- **Two Translation Modes**:
  - **Batch Translation**: Translates all items in one LLM call for speed and context-aware consistency
  - **Item-by-item Translation**: Processes each item separately, ideal for very long instruments
- **Multi-Sheet Excel Support**: Translate individual sheets or all sheets at once
- **Custom Model Support**: Enter any model name, not just predefined ones
- **Smart Empty Row Handling**: Automatically skips empty rows while maintaining alignment
- **Comprehensive Logging**: Download includes Model Selection Log, Prompt Log, and translation mode used
- **Progress Tracking**: Visual progress bars with sheet-specific status updates

## Installation

Install from CRAN:

``` r
install.packages("LLMTranslate")
```

## Usage

Launch the Shiny app:

``` r
library(LLMTranslate)
run_app()
```

### Quick Start

1. **Set up API keys** in the "API Keys & Models" tab (or set environment variables)
2. **Choose translation mode**: Batch Translation (recommended) or Item-by-item
3. **Upload Excel file** with one item per row
4. **Select sheet(s)** to translate (or choose "All sheets")
5. **Choose column** containing original items
6. **Adjust prompts** if needed (defaults follow TRAPD/ISPOR best practices)
7. **Start translation** and monitor progress
8. **Download results** as Excel with all sheets and logs

### Excel File Preparation

- One survey item per row
- Dedicated column with original text
- No merged cells
- Supports .xlsx and .xls formats
- Multiple sheets supported

## Translation Modes

**Batch Translation** (recommended for most surveys):
- Faster processing (3 LLM calls total vs N×3 for item-by-item)
- Better context-aware translations
- Maintains terminology consistency across items
- Best for instruments with <100 items

**Item-by-item Translation**:
- Works with any instrument length
- Can pause and resume translation
- Better for 100+ items or rate limit concerns
- Fine-grained error handling per item

