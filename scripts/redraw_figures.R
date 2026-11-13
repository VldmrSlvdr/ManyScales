# Exact reuse of the co-author's H1/H3 posterior forest functions.
# Contract: quantitative grid, 21 scales plus equal-scale posterior average.
# All numerical estimates and 95% CrI are read from committed aggregate RDS files.
# Only scale names and AI/LLM display wording change; no model is fitted.
# R-only graphics. Report-sized PNG 300 dpi, PDF and editable SVG; no figure note.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
root <- normalizePath(if (length(args)) args[[1]] else ".")
repo <- root
base <- file.path(repo, "analysis/ratings/primary")
out <- file.path(root, "figures")
dir.create(out, showWarnings = FALSE)

source(file.path(base, "R/hypothesis_1_functions.R"))
source(file.path(base, "R/hypothesis_3_functions.R"))
results_h1 <- readRDS(file.path(base, "results/h1/ordinal_results.rds"))
results_h3 <- readRDS(file.path(base, "results/h3/ordinal_results.rds"))

# Read only the specified plot-definition chunk, never the MCMC/setup chunks.
eval_chunk <- function(path, label) {
  lines <- readLines(path, warn = FALSE)
  hit <- which(trimws(lines) == paste0("#| label: ", label))
  stopifnot(length(hit) == 1)
  end <- hit + which(trimws(lines[(hit + 1):length(lines)]) == "```")[[1]]
  eval(parse(text = lines[(hit + 1):(end - 1)]), envir = .GlobalEnv)
}

names <- read.csv(file.path(root, "analysis/scale_names.csv"),
                  colClasses = "character", check.names = FALSE)
stopifnot(identical(names$scale_id, sprintf("%02d", 1:22)))
h1_scale_labels <- h3_scale_labels <- sprintf("%02d  %s", 1:22, names$scale_name)
h1_scale_levels <- h3_scale_levels <- h1_scale_labels[-11]
h1_aspect_labels <- c(semantic = "意味的忠実性", naturalness = "表現の自然さ", cultural_validity = "文化的妥当性")

h1_pop <- h1_population_table(results_h1)
h3_pop <- h3_population_table(results_h3)
eval_chunk(file.path(base, "h1.qmd"), "h1-scale-data")
eval_chunk(file.path(base, "h3.qmd"), "h3-forest-fn")
h3_scale_pt <- h3_scale_table(results_h3, "scale_points")

overall_h1 <- h1_pop |> transmute(aspect, mean = points_mean, lower95 = points_lower95, upper95 = points_upper95)
overall_h3 <- h3_pop |> transmute(mean = points_mean, lower95 = points_lower95, upper95 = points_upper95)
p1 <- h1_forest(h1_scale_pt, overall_h1, "人手版 − LLM版（評定点）")
p3 <- h3_forest(h3_scale_pt, overall_h3, "人手版 − LLM版（評定点）")

# Keep original plot geometry, interval coloring and overall diamond marks.
# Update the AI wording in the legend to match the manuscript's LLM wording.
display_llm <- function(p) {
  labels <- c("AI版が高い" = "LLM版が高い", "人手版が高い" = "人手版が高い", "0を含む" = "0を含む")
  color_scale <- p$scales$get_scales("colour")
  color_scale$labels <- labels
  family <- if ("Hiragino Sans" %in% systemfonts::system_fonts()$family) "Hiragino Sans" else "Arial"
  p + theme(text = element_text(family = family))
}

audit_numeric <- function(p, scales, overall, aspects = NULL) {
  expected <- bind_rows(scales |> mutate(overall = FALSE), overall |> mutate(scale_id = NA_integer_, overall = TRUE))
  actual <- p$data
  # h1_forest/h3_forest bind the exact provided rows, retaining their order.
  stopifnot(identical(actual$mean, expected$mean), identical(actual$lower95, expected$lower95),
            identical(actual$upper95, expected$upper95), identical(actual$overall, expected$overall))
  expected$scale_name <- ifelse(expected$overall, "全体", names$scale_name[match(sprintf("%02d", expected$scale_id), names$scale_id)])
  stopifnot(!anyNA(expected$scale_name))
  expected$direction <- ifelse(expected$lower95 > 0, "人手版が高い", ifelse(expected$upper95 < 0, "LLM版が高い", "0を含む"))
  expected
}
t1 <- audit_numeric(p1, h1_scale_pt, overall_h1)
t3 <- audit_numeric(p3, h3_scale_pt, overall_h3)
stopifnot(nrow(t1) == 66L, nrow(t3) == 22L,
          !11L %in% h1_scale_pt$scale_id, !11L %in% h3_scale_pt$scale_id,
          sum(h3_scale_pt$lower95 > 0) == 9L, sum(h3_scale_pt$upper95 < 0) == 1L,
          sprintf("%.3f", h3_pop$points_mean) == "0.094",
          sprintf("%.3f", h3_pop$points_lower95) == "0.071",
          sprintf("%.3f", h3_pop$points_upper95) == "0.116")
write.csv(t1, file.path(out, "h1_ratings.csv"), row.names = FALSE)
write.csv(t3, file.path(out, "h3_ratings.csv"), row.names = FALSE)

export_plot <- function(p, stem, width, height) {
  original_geometry <- ggplot_build(p)$data
  p <- display_llm(p)
  stopifnot(identical(ggplot_build(p)$data, original_geometry))
  filename <- file.path(out, stem)
  ggsave(paste0(filename, ".png"), p, device = ragg::agg_png, width = width, height = height, dpi = 300, bg = "white")
  svglite::svglite(paste0(filename, ".svg"), width = width, height = height, bg = "white")
  print(p)
  grDevices::dev.off()
  grDevices::cairo_pdf(paste0(filename, ".pdf"), width = width, height = height, family = "Hiragino Sans", bg = "white")
  print(p)
  grDevices::dev.off()
  invisible(p)
}
export_plot(p1, "h1_ratings", 12, 8)
export_plot(p3, "h3_ratings", 8, 8)

# Synchronize the published H4 figure without altering its numerical inputs.
for (ext in c("png", "pdf", "svg")) {
  stopifnot(file.copy(file.path(repo, "analysis/h4/figures", paste0("mean_item_gap_ja_ci.", ext)),
                     file.path(out, paste0("h4_similarity.", ext)), overwrite = TRUE))
}
cat("H1 rows:", nrow(t1), "; H3 rows:", nrow(t3), "; all posterior means/intervals preserved exactly.\n")
print(h3_pop[, c("points_mean", "points_lower95", "points_upper95", "scale_sd_median")])
