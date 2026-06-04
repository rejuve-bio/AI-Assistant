#!/usr/bin/env Rscript
# DESeq2 differential expression analysis
# Usage: Rscript deseq2.R <counts_path> <metadata_path> <condition_col> <reference_level> <output_dir>

suppressPackageStartupMessages({
  library(DESeq2)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
counts_path    <- args[1]
metadata_path  <- args[2]
condition_col  <- ifelse(length(args) >= 3, args[3], "condition")
ref_level      <- ifelse(length(args) >= 4 && args[4] != "NULL", args[4], NULL)
output_dir     <- ifelse(length(args) >= 5, args[5], "output")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Load data
counts   <- read.csv(counts_path, row.names = 1, check.names = FALSE)
metadata <- read.csv(metadata_path, row.names = 1, check.names = FALSE)

# Align samples
common <- intersect(colnames(counts), rownames(metadata))
counts   <- counts[, common]
metadata <- metadata[common, , drop = FALSE]
counts   <- round(as.matrix(counts))

# Set reference level
if (!is.null(ref_level)) {
  metadata[[condition_col]] <- relevel(factor(metadata[[condition_col]]), ref = ref_level)
}

# Build DESeq2 dataset
formula <- as.formula(paste("~", condition_col))
dds <- DESeqDataSetFromMatrix(countData = counts, colData = metadata, design = formula)
dds <- dds[rowSums(counts(dds)) >= 10, ]
dds <- DESeq(dds)

# Results
res <- results(dds, alpha = 0.05)
res_df <- as.data.frame(res)
res_df <- res_df[order(res_df$padj, na.last = TRUE), ]
write.csv(res_df, file.path(output_dir, "deseq2_results.csv"))

# Volcano plot
res_df$significance <- "Not significant"
res_df$significance[res_df$padj < 0.05 & res_df$log2FoldChange >  1] <- "Up"
res_df$significance[res_df$padj < 0.05 & res_df$log2FoldChange < -1] <- "Down"

volcano <- ggplot(na.omit(res_df), aes(x = log2FoldChange, y = -log10(pvalue), colour = significance)) +
  geom_point(alpha = 0.4, size = 1.5) +
  scale_colour_manual(values = c("Up" = "#e74c3c", "Down" = "#3498db", "Not significant" = "grey70")) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", colour = "grey40") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", colour = "grey40") +
  labs(title = "Volcano plot — DESeq2", x = "log2 Fold Change", y = "-log10(p-value)") +
  theme_bw(base_size = 13)
ggsave(file.path(output_dir, "volcano_plot.png"), volcano, width = 8, height = 6, dpi = 150)

# MA plot
png(file.path(output_dir, "ma_plot.png"), width = 800, height = 600, res = 120)
plotMA(res, main = "MA plot — DESeq2")
dev.off()

# Summary
n_up   <- sum(res_df$padj < 0.05 & res_df$log2FoldChange >  1, na.rm = TRUE)
n_down <- sum(res_df$padj < 0.05 & res_df$log2FoldChange < -1, na.rm = TRUE)
cat(sprintf("DESeq2 complete: %d upregulated, %d downregulated (padj<0.05, |LFC|>1)\n", n_up, n_down))
cat(sprintf("Top gene: %s (LFC=%.2f, padj=%.2e)\n",
    rownames(res_df)[1], res_df$log2FoldChange[1], res_df$padj[1]))
