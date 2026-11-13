# Launch one of the study's R apps. API submission remains a manual UI action.
args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args)) args[[1]] else "llm"
if (!mode %in% c("llm", "human")) stop("Use: Rscript translation/run.R llm|human")
root <- normalizePath(".")
.libPaths(c(file.path(root, ".Rlib"), .libPaths()))
if (!requireNamespace("LLMTranslate", quietly = TRUE) || as.character(packageVersion("LLMTranslate")) != "0.3.0") {
  stop("Run Rscript translation/install.R from the repository root to install bundled LLMTranslate 0.3.0.")
}
if (mode == "llm") LLMTranslate::run_app() else shiny::runApp(file.path(root, "translation/LLMTranslate4Hitode"))
