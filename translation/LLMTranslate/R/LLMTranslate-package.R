#' LLMTranslate: Shiny App for TRAPD/ISPOR-style Survey Translation with LLMs
#'
#' LLMTranslate wraps a Shiny application that automates forward translation,
#' optional blind back-translation, and reconciliation (TRAPD/ISPOR workflow)
#' using large language models (OpenAI GPT-4o / GPT-4.1 / o-series; Google Gemini).
#'
#' The main exported function is [run_app()], which launches the UI.
#' All other functions/objects are internal but documented below for clarity.
#'
#' @section Launching the app:
#' ```r
#' LLMTranslate::run_app()
#' ```
#' Pass arguments through to `shiny::runApp()`, e.g. `host`, `port`, `launch.browser`.
#'
#' @section UI overview (tabs and key inputs/outputs):
#' \itemize{
#'   \item \strong{API Keys & Models}:
#'     \itemize{
#'       \item \code{openai_key}, \code{gemini_key}: read from environment if empty.
#'       \item \code{forward_model}, \code{back_model}, \code{recon_model}: chosen from \code{MODEL_SPEC}.
#'       \item \code{forward_temp}, \code{back_temp}, \code{recon_temp}: temperature sliders (ignored for o1/o3).
#'       \item \code{reasoning_effort}: only for OpenAI o-series reasoning models.
#'     }
#'   \item \strong{Translation}:
#'     \itemize{
#'       \item \code{lang_from}, \code{lang_to}
#'       \item \code{file}: Excel upload (.xlsx/.xls)
#'       \item \code{orig_col}: original text column selector
#'       \item \code{forward_prompt}, \code{back_prompt}, \code{recon_prompt}
#'       \item \code{do_back}, \code{do_recon}, \code{debug}
#'       \item Buttons: \code{run}, \code{stop}; download button appears after completion
#'       \item Outputs: \code{table} (DT), \code{debug_log} (verbatim)
#'     }
#'   \item \strong{Help}: instructions, troubleshooting.
#'   \item \strong{Citation}: citation text & BibTeX.
#'   \item \strong{About}: brief description and credits.
#' }
#'

#' @section Citation:
#' See the Citation tab or \code{citation("LLMTranslate")}. Also cite specific LLMs and TRAPD/ISPOR frameworks.
#'
#' @seealso [run_app()]
#'
#' @examples
#' \donttest{
#' if (interactive()) {
#'   LLMTranslate::run_app()
#' }
#' }
#'
#' @name LLMTranslate
#' @aliases LLMTranslate-package
#' @import shiny
"_PACKAGE"
