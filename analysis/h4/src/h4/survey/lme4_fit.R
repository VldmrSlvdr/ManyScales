# Crossed random-intercept fit for one prepared frame.
# Args: data.csv formula family out_prefix nAGQ
suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
data_path <- args[[1]]
formula_text <- args[[2]]
family_name <- args[[3]]
out_prefix <- args[[4]]
n_agq <- as.integer(args[[5]])
use_ml <- length(args) >= 6 && args[[6]] == "ML"

frame <- read.csv(data_path, stringsAsFactors = FALSE)
form <- as.formula(formula_text)
warnings <- character()
fit <- withCallingHandlers(
  if (family_name == "gaussian") {
    lmer(form, data = frame, REML = !use_ml, control = lmerControl(optimizer = "bobyqa"))
  } else {
    glmer(
      form,
      data = frame,
      family = binomial(link = "logit"),
      nAGQ = n_agq,
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )
  },
  warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)

summary_fit <- summary(fit)
coef_table <- as.data.frame(coef(summary_fit))
coef_table$term <- rownames(coef_table)
if (!("Pr(>|z|)" %in% names(coef_table))) {
  stat <- coef_table[["t value"]]
  coef_table[["Pr(>|z|)"]] <- 2 * pnorm(-abs(stat))
  names(coef_table)[names(coef_table) == "t value"] <- "z"
  names(coef_table)[names(coef_table) == "Std. Error"] <- "se"
  names(coef_table)[names(coef_table) == "Estimate"] <- "estimate"
} else {
  names(coef_table)[names(coef_table) == "Std. Error"] <- "se"
  names(coef_table)[names(coef_table) == "Estimate"] <- "estimate"
  names(coef_table)[names(coef_table) == "z value"] <- "z"
  names(coef_table)[names(coef_table) == "Pr(>|z|)"] <- "p"
}
if (!("p" %in% names(coef_table)) && "Pr(>|z|)" %in% names(coef_table)) {
  names(coef_table)[names(coef_table) == "Pr(>|z|)"] <- "p"
}
coef_table$ci_low <- coef_table$estimate - 1.96 * coef_table$se
coef_table$ci_high <- coef_table$estimate + 1.96 * coef_table$se

var_rows <- as.data.frame(VarCorr(fit))
var_rows$term <- var_rows$grp
write.csv(coef_table[, c("term", "estimate", "se", "z", "p", "ci_low", "ci_high")], paste0(out_prefix, "_coef.csv"), row.names = FALSE)
write.csv(var_rows, paste0(out_prefix, "_var.csv"), row.names = FALSE)
status <- data.frame(
  n = nrow(frame),
  singular = isSingular(fit),
  aic = AIC(fit),
  bic = BIC(fit),
  loglik = as.numeric(logLik(fit)),
  warnings = paste(unique(warnings), collapse = " | "),
  stringsAsFactors = FALSE
)
write.csv(status, paste0(out_prefix, "_status.csv"), row.names = FALSE)
