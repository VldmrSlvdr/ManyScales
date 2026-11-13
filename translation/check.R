# Offline compatibility check: construct the app without launching or calling an API.
root <- normalizePath(".")
.libPaths(c(file.path(root, ".Rlib"), .libPaths()))
stopifnot(requireNamespace("LLMTranslate", quietly = TRUE), as.character(packageVersion("LLMTranslate")) == "0.3.0")
app_path <- file.path(root, "translation/LLMTranslate4Hitode/app.R")
invisible(parse(file = app_path))
app <- source(app_path, local = new.env(parent = globalenv()))$value
stopifnot(inherits(app, "shiny.appobj"))
cat("LLMTranslate 0.3.0 and Hitode app construction passed; no API calls made.\n")
