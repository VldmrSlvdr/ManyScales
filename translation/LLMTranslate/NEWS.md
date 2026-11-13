# LLMTranslate 0.3.0

## Major Features

* **Batch Translation Mode**: New translation mode that sends all items in a single LLM call per stage (forward/back/recon) for faster processing and better context-aware translations
* **Multi-Sheet Excel Support**: Can now select and translate individual sheets or all sheets at once from Excel files with multiple sheets
* **Custom Model Input**: Model selection fields now accept custom model names typed by users, allowing use of newly released models without app updates
* **Empty Row Handling**: Automatically filters and skips empty rows during translation while maintaining correct row alignment
* **Sheet-Specific Status**: Status notifications now indicate which sheet is being processed during multi-sheet translations

## UI/UX Improvements

* Batch Translation tab now appears before Item-by-item Translation as the recommended default
* Added sheet selector dropdown for viewing results from different sheets after multi-sheet translation
* Added "Translation Mode" column to Model Selection Log indicating whether Batch or Item-by-item mode was used
* Added comprehensive Excel file preparation guide in Help tab
* Model selection dropdowns now support typing custom model names with autocomplete
* Custom models trigger informative warnings instead of blocking translation
* Progress text shows current sheet name during multi-sheet processing
* Multi-sheet downloads include all translated sheets plus Model Selection and Prompt logs

## Translation Quality Improvements

* Batch mode prompts emphasize context and terminology consistency across items
* Fixed item number parsing in batch responses to prevent row misalignment
* Parser now uses `item_number` field from LLM responses for accurate mapping
* Improved error handling and logging for batch translation operations

## Bug Fixes

* Fixed preview table error when displaying multi-sheet translation results
* Fixed row mapping issue where translations were offset by one row
* Fixed tryCatch structure in batch translation for proper error handling
* Corrected indentation and control flow in multi-sheet processing loop

# LLMTranslate 0.2.0

## Major Features

* Added support for Anthropic Claude models (Sonnet 4.5, Haiku 4.5, Opus 4.1, and others)
* Updated Google Gemini models to latest versions (Gemini 2.5 Pro, 2.5 Flash, 2.0 Flash)
* Added Model Selection Log sheet to Excel output
* Added Prompt Log sheet to Excel output for full reproducibility

## UI/UX Improvements

* Added visual progress bar showing item X of Y with percentage
* Added "Test Connection" buttons for all API providers with success/error feedback
* Added resume functionality - can continue after stopping mid-batch
* Added warning notification for large batches (500+ items)
* Added real-time preview of first completed translation
* Added "Reset" button to clear all data and start fresh
* Added comprehensive "API Keys Guide" tab with step-by-step instructions for obtaining keys
* Improved Help tab with updated instructions

## Bug Fixes

* Fixed Unicode display issues in API status messages
* Fixed deprecated Gemini model names (updated fallback logic)

# LLMTranslate 0.1.3

# LLMTranslate 0.1.2


# LLMTranslate 0.1.1

* Address CRAN feedback: quotes around software names, expanded acronyms, added references, \value in Rd, minimal test.

# LLMTranslate 0.1.0 (2025-07-24)

* Initial CRAN release.
