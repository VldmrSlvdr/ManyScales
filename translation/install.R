# Install the bundled package and UI dependencies; never makes translation requests.
args <- commandArgs(trailingOnly = TRUE)
root <- normalizePath(if (length(args)) args[[1]] else ".")
lib <- file.path(root, ".Rlib")
dir.create(lib, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
deps <- c("shiny", "DT", "readxl", "writexl", "openxlsx", "httr2", "jsonlite", "glue", "later")
missing <- deps[!vapply(deps, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, lib = lib, repos = "https://cloud.r-project.org")
package_dir <- tempfile("LLMTranslate-build-")
dir.create(package_dir)
source_dir <- file.path(root, "translation/LLMTranslate")
stopifnot(all(file.copy(list.files(source_dir, full.names = TRUE, all.files = TRUE, no.. = TRUE),
                       package_dir, recursive = TRUE)))
stopifnot(file.copy(file.path(root, "LICENSE"), file.path(package_dir, "LICENSE")))
tryCatch(
  install.packages(package_dir, repos = NULL, type = "source", lib = lib),
  finally = unlink(package_dir, recursive = TRUE)
)
stopifnot(as.character(packageVersion("LLMTranslate", lib.loc = lib)) == "0.3.0")
cat("Bundled LLMTranslate 0.3.0 installed in", lib, "\n")
