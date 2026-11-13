#' Launch the LLM Survey Translator shiny app
#'
#' @param ... Passed to shiny::runApp.
#' @return No return value, called for side effects (launches a shiny app).
#' @export
#'
#' @examples
#' if (interactive()) {
#'   run_app()
#' }
run_app <- function(...) {
  needed <- c("DT","readxl","writexl","openxlsx","httr2","jsonlite","glue","later")
  miss <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(miss)) stop("Install missing packages: ", paste(miss, collapse = ", "))
  app_dir <- system.file("app", package = "LLMTranslate")
  shiny::runApp(app_dir, ...)
}
