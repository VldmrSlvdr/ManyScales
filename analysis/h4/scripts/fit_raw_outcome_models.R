# Exploratory H4 models on observed outcomes rather than difference scores.
#
# The rating structure follows the co-author H1/H3 reports in `refers/`:
# one row per respondent, scale, and translation; effect-coded translation;
# respondent- and scale-specific translation effects; and a pair intercept.
# H1/H3 use Bayesian cumulative logits. Here the exploratory association is
# estimated with a cumulative-link mixed model (Laplace approximation), so the
# outcome remains the observed 1--5 ordinal response.

suppressPackageStartupMessages({
  library(lme4)
  library(ordinal)
})

args <- commandArgs(trailingOnly = TRUE)
frame_dir <- args[[1]]
out_dir <- args[[2]]
publish_dir <- if (length(args) >= 3) args[[3]] else ""
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (nzchar(publish_dir)) dir.create(publish_dir, recursive = TRUE, showWarnings = FALSE)

wald_rows <- function(fit, model, outcome, audience, family, formula_text, n, n_people, n_scales) {
  tab <- as.data.frame(coef(summary(fit)))
  tab$term <- rownames(tab)
  estimate_name <- if ("Estimate" %in% names(tab)) "Estimate" else "estimate"
  se_name <- if ("Std. Error" %in% names(tab)) "Std. Error" else "se"
  z_name <- if ("z value" %in% names(tab)) "z value" else "z"
  tab$estimate <- tab[[estimate_name]]
  tab$se <- tab[[se_name]]
  tab$z <- tab[[z_name]]
  tab$p <- 2 * pnorm(-abs(tab$z))
  tab$ci_low <- tab$estimate - 1.96 * tab$se
  tab$ci_high <- tab$estimate + 1.96 * tab$se
  tab$model <- model
  tab$outcome <- outcome
  tab$audience <- audience
  tab$family <- family
  tab$formula <- formula_text
  tab$n <- n
  tab$n_people <- n_people
  tab$n_scales <- n_scales
  tab[, c(
    "model", "outcome", "audience", "family", "formula", "term",
    "estimate", "se", "z", "p", "ci_low", "ci_high",
    "n", "n_people", "n_scales"
  )]
}

max_gradient <- function(fit) {
  gradient <- tryCatch(
    {
      if (inherits(fit, "clmm")) fit$gradient else fit@optinfo$derivs$gradient
    },
    error = function(e) NULL
  )
  if (is.null(gradient) || !is.numeric(gradient) || length(gradient) == 0) {
    return(NA_real_)
  }
  max(abs(gradient))
}

version_similarity_rows <- function(fit, tab) {
  terms <- c("sim", "x:sim")
  beta <- if (inherits(fit, "clmm")) coef(fit) else fixef(fit)
  variance <- as.matrix(vcov(fit))
  make_row <- function(label, weights) {
    estimate <- sum(weights * beta[terms])
    se <- sqrt(as.numeric(t(weights) %*% variance[terms, terms, drop = FALSE] %*% weights))
    row <- tab[tab$term == "sim", , drop = FALSE]
    row$term <- label
    row$estimate <- estimate
    row$se <- se
    row$z <- estimate / se
    row$p <- 2 * pnorm(-abs(row$z))
    row$ci_low <- estimate - 1.96 * se
    row$ci_high <- estimate + 1.96 * se
    row
  }
  rbind(
    make_row("sim_llm", c(1, -0.5)),
    make_row("sim_human", c(1, 0.5))
  )
}

rating_formula <- ordered_rating ~
  x * sim + ai_first +
  (1 | respondent_id) + (0 + x | respondent_id) +
  (1 | scale_id) + (0 + x | scale_id) +
  (1 | pair_id)

rating_formula_text <- paste(
  "ordered_rating ~ x * own_similarity_z + ai_first +",
  "(1 | respondent_id) + (0 + x | respondent_id) +",
  "(1 | scale_id) + (0 + x | scale_id) + (1 | pair_id)"
)

ratings <- read.csv(file.path(frame_dir, "rating_long.csv"), stringsAsFactors = FALSE)
ratings$x <- ifelse(ratings$version == "human", 0.5, -0.5)
ratings$sim <- ratings$sim_cosine
# cultural_validity export codes are already high = better; do not reverse again.
ratings$ordered_rating <- ordered(as.integer(ratings$rating), levels = 1:5)
ratings$pair_id <- interaction(
  ratings$respondent_id, ratings$scale_id, ratings$outcome, drop = TRUE
)
ratings$respondent_id <- factor(ratings$respondent_id)
ratings$scale_id <- factor(sprintf("%02d", as.integer(ratings$scale_id)))

rating_specs <- list(
  list(outcome = "naturalness", audience = "citizen", expert = 0),
  list(outcome = "naturalness", audience = "expert", expert = 1),
  list(outcome = "semantic", audience = "expert", expert = 1),
  list(outcome = "cultural_validity", audience = "expert", expert = 1),
  list(outcome = "choice_fit", audience = "expert", expert = 1)
)

rating_records <- list()
for (spec in rating_specs) {
  part <- ratings[
    ratings$outcome == spec$outcome & ratings$expert == spec$expert &
      !is.na(ratings$ordered_rating) & !is.na(ratings$sim),
  ]
  part <- droplevels(part)
  fit <- clmm(
    rating_formula,
    data = part,
    Hess = TRUE,
    nAGQ = 1,
    control = clmm.control(maxIter = 150, gradTol = 1e-3)
  )
  tab <- wald_rows(
    fit,
    "raw_ordinal_rating",
    spec$outcome,
    spec$audience,
    "cumulative_logit",
    rating_formula_text,
    nrow(part),
    length(unique(part$respondent_id)),
    length(unique(part$scale_id))
  )
  tab <- rbind(tab, version_similarity_rows(fit, tab))
  tab <- tab[tab$term %in% c("x", "sim", "sim_llm", "sim_human", "ai_first", "x:sim"), ]
  tab$convergence <- fit$optRes$convergence
  tab$max_gradient <- max_gradient(fit)
  rating_records[[length(rating_records) + 1]] <- tab
  write.csv(
    do.call(rbind, rating_records),
    file.path(out_dir, "raw_rating_ordinal.csv"),
    row.names = FALSE
  )
}
rating_result <- do.call(rbind, rating_records)
if (nzchar(publish_dir)) {
  write.csv(
    rating_result,
    file.path(publish_dir, "raw_rating_ordinal_n1859.csv"),
    row.names = FALSE
  )
}

fit_binomial <- function(frame, formula) {
  glmer(
    formula,
    data = frame,
    family = binomial(link = "logit"),
    nAGQ = 0,
    control = glmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = 2e5),
      check.conv.singular = .makeCC(action = "message", tol = 1e-4)
    )
  )
}

choice_formula <- y ~
  sim_human + sim_llm + ai_first +
  (1 | respondent_id) + (1 | scale_id)
choice_formula_text <- paste(
  "outcome ~ human_similarity_z + llm_similarity_z + ai_first +",
  "(1 | respondent_id) + (1 | scale_id)"
)

choices <- read.csv(file.path(frame_dir, "choice.csv"), stringsAsFactors = FALSE)
choices$respondent_id <- factor(choices$respondent_id)
choices$scale_id <- factor(sprintf("%02d", as.integer(choices$scale_id)))
choices$sim_human <- choices$cosine_human_z
choices$sim_llm <- choices$cosine_llm_z

choice_records <- list()
for (audience_value in c(0, 1)) {
  audience_name <- if (audience_value == 0) "citizen" else "expert"
  for (spec in list(
    list(outcome = "preference", column = "prefer_human"),
    list(outcome = "guess", column = "guess_llm")
  )) {
    part <- choices[choices$expert == audience_value & !is.na(choices[[spec$column]]), ]
    part$y <- part[[spec$column]]
    fit <- fit_binomial(part, choice_formula)
    tab <- wald_rows(
      fit,
      "raw_binary_choice",
      spec$outcome,
      audience_name,
      "binomial_logit",
      choice_formula_text,
      nrow(part),
      length(unique(part$respondent_id)),
      length(unique(part$scale_id))
    )
    tab <- tab[tab$term %in% c("sim_human", "sim_llm", "ai_first"), ]
    tab$convergence <- if (is.null(fit@optinfo$conv$lme4$messages)) 0 else 1
    tab$max_gradient <- max_gradient(fit)
    choice_records[[length(choice_records) + 1]] <- tab
  }
}
choice_result <- do.call(rbind, choice_records)
write.csv(choice_result, file.path(out_dir, "raw_choice_logit.csv"), row.names = FALSE)
if (nzchar(publish_dir)) {
  write.csv(
    choice_result,
    file.path(publish_dir, "raw_choice_logit_n1859.csv"),
    row.names = FALSE
  )
}

flag_formula <- flagged ~
  x * sim + ai_first +
  (1 | respondent_id) + (0 + x | respondent_id) +
  (1 | scale_id) + (0 + x | scale_id) +
  (1 | item_uid) + (1 | pair_id)
flag_formula_text <- paste(
  "flagged ~ x * own_similarity_z + ai_first +",
  "(1 | respondent_id) + (0 + x | respondent_id) +",
  "(1 | scale_id) + (0 + x | scale_id) +",
  "(1 | item_uid) + (1 | respondent_id:item_uid)"
)

flags <- read.csv(file.path(frame_dir, "flag_long.csv"), stringsAsFactors = FALSE)
flags$x <- ifelse(flags$version == "human", 0.5, -0.5)
flags$sim <- flags$cosine_z
flags$respondent_id <- factor(flags$respondent_id)
flags$scale_id <- factor(sprintf("%02d", as.integer(flags$scale_id)))
flags$item_uid <- factor(flags$item_uid)
flags$pair_id <- interaction(flags$respondent_id, flags$item_uid, drop = TRUE)

flag_records <- list()
for (dimension in c("semantic", "understandability", "cultural")) {
  part <- droplevels(flags[flags$dimension == dimension, ])
  fit <- fit_binomial(part, flag_formula)
  tab <- wald_rows(
    fit,
    "raw_binary_flag",
    dimension,
    "expert",
    "binomial_logit",
    flag_formula_text,
    nrow(part),
    length(unique(part$respondent_id)),
    length(unique(part$scale_id))
  )
  tab <- rbind(tab, version_similarity_rows(fit, tab))
  tab <- tab[tab$term %in% c("x", "sim", "sim_llm", "sim_human", "ai_first", "x:sim"), ]
  tab$convergence <- if (is.null(fit@optinfo$conv$lme4$messages)) 0 else 1
  tab$max_gradient <- max_gradient(fit)
  flag_records[[length(flag_records) + 1]] <- tab
  write.csv(
    do.call(rbind, flag_records),
    file.path(out_dir, "raw_flag_logit.csv"),
    row.names = FALSE
  )
}
flag_result <- do.call(rbind, flag_records)
if (nzchar(publish_dir)) {
  write.csv(
    flag_result,
    file.path(publish_dir, "raw_flag_logit_n1859.csv"),
    row.names = FALSE
  )
}
