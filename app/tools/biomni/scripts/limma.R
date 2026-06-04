#!/usr/bin/env Rscript
# limma differential expression (microarray or RNA-seq with voom)
# Usage: Rscript limma.R <expression_path> <metadata_path> <condition_col> <reference_level> <output_dir>

suppressPackageStartupMessages({
  library(limma)
  library(ggplot2)
})

args          <- commandArgs(trailingOnly = TRUE)
expr_path     <- args[1]
metadata_path <- args[2]
condition_col <- ifelse(length(args) >= 3, args[3], "condition")
ref_level     <- ifelse(length(args) >= 4 && args[4] != "NULL", args[4], NULL)
output_dir    <- ifelse(length(args) >= 5, args[5], "output")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

expr     <- read.csv(expr_path,     row.names = 1, check.names = FALSE)
metadata <- read.csv(metadata_path, row.names = 1, check.names = FALSE)

common   <- intersect(colnames(expr), rownames(metadata))
expr     <- as.matrix(expr[, common])
metadata <- metadata[common, , drop = FALSE]

group <- factor(metadata[[condition_col]])
if (!is.null(ref_level)) group <- relevel(group, ref = ref_level)

design <- model.matrix(~ group)

# voom transform if integer counts
if (all(expr == floor(expr))) {
  v      <- voom(expr, design, plot = FALSE)
  fit    <- lmFit(v, design)
} else {
  fit    <- lmFit(expr, design)
}

fit2   <- eBayes(fit)
res_df <- topTable(fit2, coef = 2, number = Inf, sort.by = "P")
write.csv(res_df, file.path(output_dir, "limma_results.csv"))

# Volcano plot
res_df$significance <- "NS"
res_df$significance[res_df$adj.P.Val < 0.05 & res_df$logFC >  1] <- "Up"
res_df$significance[res_df$adj.P.Val < 0.05 & res_df$logFC < -1] <- "Down"

p <- ggplot(res_df, aes(logFC, -log10(P.Value), colour = significance)) +
  geom_point(alpha = 0.4, size = 1.5) +
  scale_colour_manual(values = c("Up" = "#e74c3c", "Down" = "#3498db", "NS" = "grey70")) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
  labs(title = "Volcano plot — limma", x = "logFC", y = "-log10(P-value)") +
  theme_bw(base_size = 13)
ggsave(file.path(output_dir, "volcano_plot.png"), p, width = 8, height = 6, dpi = 150)

n_up   <- sum(res_df$adj.P.Val < 0.05 & res_df$logFC >  1)
n_down <- sum(res_df$adj.P.Val < 0.05 & res_df$logFC < -1)
cat(sprintf("limma complete: %d up, %d down (adj.P.Val<0.05, |logFC|>1)\n", n_up, n_down))
cat(sprintf("Top gene: %s (logFC=%.2f, adj.P=%.2e)\n",
    rownames(res_df)[1], res_df$logFC[1], res_df$adj.P.Val[1]))
