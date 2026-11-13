# Similarity interactions for the raw outcomes.
# Naturalness slopes are read from naturalness_raw.csv.
# Preference and the guess use both versions' own similarities.
# Expert flags use own item similarity times human versus LLM.
# Citizen-versus-expert is not fit for the flags: only experts marked them.

suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)
frame_dir <- args[[1]]
fit_dir <- args[[2]]
figure_path <- args[[3]]

choices <- read.csv(file.path(frame_dir, "choice.csv"), stringsAsFactors = FALSE)
flags <- read.csv(file.path(frame_dir, "flag_long.csv"), stringsAsFactors = FALSE)
choices$scale_id <- sprintf("%02d", as.integer(choices$scale_id))
flags$scale_id <- sprintf("%02d", as.integer(flags$scale_id))

empty <- data.frame(
  outcome = character(), question = character(), group = character(),
  estimate = numeric(), se = numeric(), ci_low = numeric(), ci_high = numeric(),
  p = numeric(), stringsAsFactors = FALSE
)

row_of <- function(outcome, question, group, estimate, se) {
  data.frame(
    outcome = outcome, question = question, group = group,
    estimate = estimate, se = se,
    ci_low = estimate - 1.96 * se, ci_high = estimate + 1.96 * se,
    p = 2 * pnorm(-abs(estimate / se)),
    stringsAsFactors = FALSE
  )
}

from_fit <- function(fit, outcome, question, group, terms, signs = NULL) {
  beta <- fixef(fit)
  variance <- as.matrix(vcov(fit))
  if (is.null(signs)) signs <- rep(1, length(terms))
  estimate <- sum(signs * beta[terms])
  se <- sqrt(as.numeric(t(signs) %*% variance[terms, terms, drop = FALSE] %*% signs))
  row_of(outcome, question, group, estimate, se)
}

natural <- read.csv(file.path(fit_dir, "naturalness_raw.csv"), stringsAsFactors = FALSE)
natural <- natural[natural$predictor == "cosine", ]
pick <- function(model, term) natural[natural$model == model & natural$term == term, ]
records <- empty
citizen <- pick("by_expert", "sim")
expert_slope <- pick("by_expert_slope", "sim + sim:expert")
who_gap <- pick("by_expert", "sim:expert")
llm_slope <- pick("by_version", "sim")
human_slope <- pick("by_version_human_slope", "sim + sim:version_human")
version_gap <- pick("by_version", "sim:version_human")
records <- rbind(
  records,
  row_of("Naturalness", "who", "Citizen", citizen$estimate, citizen$se),
  row_of("Naturalness", "who", "Expert", expert_slope$estimate, expert_slope$se),
  row_of("Naturalness", "who", "Expert minus citizen", who_gap$estimate, who_gap$se),
  row_of("Naturalness", "version", "Human text", human_slope$estimate, human_slope$se),
  row_of("Naturalness", "version", "LLM text", llm_slope$estimate, llm_slope$se),
  row_of("Naturalness", "version", "Version difference", version_gap$estimate, version_gap$se)
)

fit_binomial <- function(frame, formula) {
  glmer(
    formula, data = frame, family = binomial(link = "logit"), nAGQ = 0,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  )
}

for (spec in list(
  list(outcome = "Preference", column = "prefer_human"),
  list(outcome = "LLM guess", column = "guess_llm")
)) {
  base <- choices[!is.na(choices[[spec$column]]), ]
  base$y <- base[[spec$column]]
  base$sim_human <- base$cosine_human_z
  base$sim_llm <- base$cosine_llm_z
  version_fit <- fit_binomial(
    base,
    y ~ sim_human + sim_llm + ai_first + expert + (1 | respondent_id) + (1 | scale_id)
  )
  who_fit <- fit_binomial(
    base,
    y ~ sim_human * expert + sim_llm * expert + ai_first + (1 | respondent_id) + (1 | scale_id)
  )
  records <- rbind(
    records,
    from_fit(version_fit, spec$outcome, "version", "Human similarity", "sim_human"),
    from_fit(version_fit, spec$outcome, "version", "LLM similarity", "sim_llm"),
    from_fit(version_fit, spec$outcome, "version", "Human minus LLM slope", c("sim_human", "sim_llm"), c(1, -1)),
    from_fit(who_fit, spec$outcome, "who", "Citizen, human similarity", "sim_human"),
    from_fit(who_fit, spec$outcome, "who", "Expert, human similarity", c("sim_human", "sim_human:expert")),
    from_fit(who_fit, spec$outcome, "who", "Citizen, LLM similarity", "sim_llm"),
    from_fit(who_fit, spec$outcome, "who", "Expert, LLM similarity", c("sim_llm", "expert:sim_llm")),
    from_fit(who_fit, spec$outcome, "who", "Human similarity × expert", "sim_human:expert"),
    from_fit(who_fit, spec$outcome, "who", "LLM similarity × expert", "expert:sim_llm")
  )
}

flag_names <- c(semantic = "Semantic flag", understandability = "Understandability flag", cultural = "Cultural flag")
for (dimension in names(flag_names)) {
  part <- flags[flags$dimension == dimension, ]
  part$sim <- part$cosine_z
  part$version_human <- as.integer(part$version == "human")
  fit <- fit_binomial(
    part,
    flagged ~ sim * version_human + ai_first + (1 | respondent_id) + (1 | scale_id) + (1 | item_uid)
  )
  records <- rbind(
    records,
    from_fit(fit, flag_names[[dimension]], "version", "Human text", c("sim", "sim:version_human")),
    from_fit(fit, flag_names[[dimension]], "version", "LLM text", "sim"),
    from_fit(fit, flag_names[[dimension]], "version", "Version difference", "sim:version_human")
  )
}

records$excludes_zero <- records$ci_low > 0 | records$ci_high < 0
write.csv(records, file.path(fit_dir, "similarity_interactions.csv"), row.names = FALSE)

plot_frame <- records
plot_frame$label <- paste(plot_frame$outcome, plot_frame$group, sep = " — ")
plot_frame$panel <- ifelse(plot_frame$question == "who", "Similarity × citizen versus expert", "Similarity × version")
who_levels <- plot_frame$label[plot_frame$question == "who"]
version_levels <- plot_frame$label[plot_frame$question == "version"]
plot_frame$label <- factor(plot_frame$label, levels = rev(c(who_levels, version_levels)))
plot_frame$group <- factor(plot_frame$group, levels = unique(plot_frame$group))

figure <- ggplot(plot_frame, aes(estimate, label, color = excludes_zero)) +
  geom_vline(xintercept = 0, linewidth = 0.3) +
  geom_pointrange(aes(xmin = ci_low, xmax = ci_high), linewidth = 0.4) +
  facet_wrap(~panel, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c("TRUE" = "#1B1B1B", "FALSE" = "#4C78A8"), guide = "none") +
  labs(x = "Cosine slope (raw outcome; 1 SD of own similarity)", y = NULL) +
  theme_classic(base_size = 11) +
  theme(strip.background = element_blank(), axis.text.y = element_text(size = 8))
ggsave(figure_path, figure, width = 8.2, height = 9.2, dpi = 160)
print(records)
