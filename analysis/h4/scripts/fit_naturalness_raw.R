# Naturalness on the raw 1-5 rating of one version.
# rating ~ similarity + order + expert + (1 | person) + (1 | scale)
# Then one interaction at a time: similarity by version, by order, and by expert.
# Similarity is the scale-mean back-translation score of that version, z-scored
# on both versions together. It is not an item-level slope.

suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
frame_dir <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ratings <- read.csv(file.path(frame_dir, "rating_long.csv"), stringsAsFactors = FALSE)
nat <- ratings[ratings$outcome == "naturalness", ]
nat$version_human <- as.integer(nat$version == "human")
nat$scale_id <- sprintf("%02d", as.integer(nat$scale_id))

wald <- function(fit) {
  tab <- as.data.frame(coef(summary(fit)))
  tab$term <- rownames(tab)
  tab$p <- 2 * pnorm(-abs(tab[["t value"]]))
  names(tab)[names(tab) == "t value"] <- "z"
  names(tab)[names(tab) == "Estimate"] <- "estimate"
  names(tab)[names(tab) == "Std. Error"] <- "se"
  tab$ci_low <- tab$estimate - 1.96 * tab$se
  tab$ci_high <- tab$estimate + 1.96 * tab$se
  tab[, c("term", "estimate", "se", "z", "p", "ci_low", "ci_high")]
}

slope <- function(fit, terms) {
  beta <- fixef(fit)
  variance <- as.matrix(vcov(fit))
  estimate <- sum(beta[terms])
  se <- sqrt(sum(variance[terms, terms]))
  data.frame(
    term = paste(terms, collapse = " + "),
    estimate = estimate,
    se = se,
    z = estimate / se,
    p = 2 * pnorm(-abs(estimate / se)),
    ci_low = estimate - 1.96 * se,
    ci_high = estimate + 1.96 * se
  )
}

forms <- list(
  main = rating ~ sim + ai_first + expert + (1 | respondent_id) + (1 | scale_id),
  by_version = rating ~ sim * version_human + ai_first + expert + (1 | respondent_id) + (1 | scale_id),
  by_order = rating ~ sim * ai_first + expert + (1 | respondent_id) + (1 | scale_id),
  by_expert = rating ~ sim * expert + ai_first + (1 | respondent_id) + (1 | scale_id)
)

rows <- list()
for (metric in c("cosine", "w1", "w2")) {
  nat$sim <- nat[[paste0("sim_", metric)]]
  for (name in names(forms)) {
    fit <- lmer(forms[[name]], data = nat, REML = TRUE, control = lmerControl(optimizer = "bobyqa"))
    tab <- wald(fit)
    tab$model <- name
    tab$predictor <- metric
    tab$singular <- isSingular(fit, tol = 1e-4)
    tab$n <- nrow(nat)
    tab$n_people <- length(unique(nat$respondent_id))
    tab$n_scales <- length(unique(nat$scale_id))
    rows[[length(rows) + 1]] <- tab
    if (name == "by_version") {
      extra <- slope(fit, c("sim", "sim:version_human"))
      extra$model <- "by_version_human_slope"
      extra$predictor <- metric
      extra$singular <- isSingular(fit, tol = 1e-4)
      extra$n <- nrow(nat)
      extra$n_people <- length(unique(nat$respondent_id))
      extra$n_scales <- length(unique(nat$scale_id))
      rows[[length(rows) + 1]] <- extra
    }
    if (name == "by_order") {
      extra <- slope(fit, c("sim", "sim:ai_first"))
      extra$model <- "by_order_ai_first_slope"
      extra$predictor <- metric
      extra$singular <- isSingular(fit, tol = 1e-4)
      extra$n <- nrow(nat)
      extra$n_people <- length(unique(nat$respondent_id))
      extra$n_scales <- length(unique(nat$scale_id))
      rows[[length(rows) + 1]] <- extra
    }
    if (name == "by_expert") {
      extra <- slope(fit, c("sim", "sim:expert"))
      extra$model <- "by_expert_slope"
      extra$predictor <- metric
      extra$singular <- isSingular(fit, tol = 1e-4)
      extra$n <- nrow(nat)
      extra$n_people <- length(unique(nat$respondent_id))
      extra$n_scales <- length(unique(nat$scale_id))
      rows[[length(rows) + 1]] <- extra
    }
  }
}

result <- do.call(rbind, rows)
write.csv(result, file.path(out_dir, "naturalness_raw.csv"), row.names = FALSE)
print(result[, c("model", "predictor", "term", "estimate", "ci_low", "ci_high", "p", "singular")])
