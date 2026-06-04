#!/usr/bin/env Rscript
# Survival analysis — Kaplan-Meier curves + log-rank test
# Usage: Rscript survival.R <data_path> <time_col> <event_col> <group_col> <output_dir>

suppressPackageStartupMessages({
  library(survival)
  library(ggplot2)
  library(dplyr)
})

args       <- commandArgs(trailingOnly = TRUE)
data_path  <- args[1]
time_col   <- ifelse(length(args) >= 2, args[2], "time")
event_col  <- ifelse(length(args) >= 3, args[3], "event")
group_col  <- ifelse(length(args) >= 4, args[4], "group")
output_dir <- ifelse(length(args) >= 5, args[5], "output")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

df      <- read.csv(data_path, check.names = FALSE)
surv_obj <- Surv(df[[time_col]], df[[event_col]])
fit      <- survfit(surv_obj ~ df[[group_col]])

# Log-rank test
lr_test  <- survdiff(surv_obj ~ df[[group_col]])
p_val    <- pchisq(lr_test$chisq, df = length(lr_test$n) - 1, lower.tail = FALSE)

# KM plot
png(file.path(output_dir, "kaplan_meier.png"), width = 900, height = 700, res = 130)
plot(fit,
     col    = seq_along(levels(factor(df[[group_col]]))),
     lwd    = 2,
     xlab   = "Time",
     ylab   = "Survival probability",
     main   = sprintf("Kaplan-Meier — log-rank p = %.4f", p_val))
legend("topright", levels(factor(df[[group_col]])),
       col = seq_along(levels(factor(df[[group_col]]))), lwd = 2)
dev.off()

# Summary table
sum_df <- data.frame(
  group    = names(fit$n),
  n        = fit$n,
  events   = fit$n - fit$n * fit$surv[nrow(fit$surv), ]
)
write.csv(sum_df, file.path(output_dir, "survival_summary.csv"), row.names = FALSE)

cat(sprintf("Survival analysis complete. Log-rank p-value: %.4f\n", p_val))
cat(sprintf("Groups: %s\n", paste(names(fit$n), collapse = ", ")))
