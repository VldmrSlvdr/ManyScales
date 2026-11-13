suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)
fit_dir <- args[[1]]
figure_path <- args[[2]]

ratings <- read.csv(file.path(fit_dir, "raw_rating_ordinal.csv"), stringsAsFactors = FALSE)
choices <- read.csv(file.path(fit_dir, "raw_choice_logit.csv"), stringsAsFactors = FALSE)
flags <- read.csv(file.path(fit_dir, "raw_flag_logit.csv"), stringsAsFactors = FALSE)

ratings <- ratings[ratings$term %in% c("sim_human", "sim_llm", "x:sim"), ]
ratings$panel <- "Ordinal ratings"
term_label <- c(
  sim_human = "human-text similarity",
  sim_llm = "LLM-text similarity",
  "x:sim" = "human minus LLM slope"
)
ratings$label <- paste(ratings$audience, ratings$outcome, term_label[ratings$term], sep = " — ")

choices <- choices[choices$term %in% c("sim_human", "sim_llm"), ]
choices$panel <- "Binary choices"
choices$label <- paste(choices$audience, choices$outcome, term_label[choices$term], sep = " — ")

flags <- flags[flags$term %in% c("sim_human", "sim_llm", "x:sim"), ]
flags$panel <- "Expert problem flags"
flags$label <- paste(flags$outcome, term_label[flags$term], sep = " — ")

plot_frame <- rbind(
  ratings[, c("panel", "label", "term", "estimate", "ci_low", "ci_high")],
  choices[, c("panel", "label", "term", "estimate", "ci_low", "ci_high")],
  flags[, c("panel", "label", "term", "estimate", "ci_low", "ci_high")]
)
plot_frame$excludes_zero <- plot_frame$ci_low > 0 | plot_frame$ci_high < 0
plot_frame$label <- factor(plot_frame$label, levels = rev(plot_frame$label))

figure <- ggplot(plot_frame, aes(estimate, label, color = excludes_zero)) +
  geom_vline(xintercept = 0, linewidth = 0.3) +
  geom_pointrange(aes(xmin = ci_low, xmax = ci_high), linewidth = 0.4) +
  facet_wrap(~panel, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c("TRUE" = "#1B1B1B", "FALSE" = "#4C78A8"), guide = "none") +
  labs(
    x = "Log-odds slope per SD of that translation's similarity",
    y = NULL
  ) +
  theme_classic(base_size = 11, base_family = "sans") +
  theme(strip.background = element_blank(), axis.text.y = element_text(size = 7.5))

figure_path <- sub("\\.[^.]+$", ".png", figure_path)
ggsave(figure_path, figure, width = 8.4, height = 10.2, dpi = 300)
base_path <- sub("\\.png$", "", figure_path)
svglite::svglite(paste0(base_path, ".svg"), width=8.4, height=10.2)
print(figure)
dev.off()
grDevices::cairo_pdf(paste0(base_path, ".pdf"), width=8.4, height=10.2, family="sans")
print(figure)
dev.off()
