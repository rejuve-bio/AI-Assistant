"""
Biomni visualization tools.
All tools use matplotlib, seaborn, networkx directly — no LLM code generation.
Each tool generates one or more image files in output_dir and returns their paths.
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")


# ─────────────────────────────────────────────────────────────────────────────
# PPI / interaction network visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_ppi_network(gene_name: str, species: int = 9606,
                     min_score: int = 700, output_dir: str = "output/") -> dict:
    """
    Fetch protein-protein interactions from STRING and draw a network graph.

    Args:
        gene_name: Central gene/protein (e.g. 'BRCA1', 'FOXO3')
        species: NCBI taxonomy ID (9606 = human)
        min_score: Minimum STRING confidence score (0-1000, default 700 = high)
        output_dir: Where to save the PNG

    Returns:
        dict with network stats and output_files (PNG path)
    """
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from .database_connectors import query_string

        os.makedirs(output_dir, exist_ok=True)

        # Get interaction data
        data = query_string(gene_name, species=species, limit=25, min_score=min_score)
        if data.get("error"):
            return {"error": data["error"], "source": "PPI network"}

        interactions = data.get("interactions", [])
        if not interactions:
            return {"gene": gene_name, "error": "No interactions found above threshold", "source": "STRING"}

        # Build graph
        G = nx.Graph()
        G.add_node(gene_name, central=True)
        for i in interactions:
            partner = i["partner"]
            score   = i.get("score", 0)
            G.add_node(partner, central=False)
            G.add_edge(gene_name, partner, weight=score)

        # Layout + draw
        pos    = nx.spring_layout(G, seed=42, k=2.5)
        scores = [G[gene_name][n].get("weight", 0) for n in G.neighbors(gene_name)]
        norm   = plt.Normalize(vmin=min(scores), vmax=max(scores)) if scores else plt.Normalize()

        fig, ax = plt.subplots(figsize=(12, 10))
        # Draw edges coloured by score
        for u, v, d in G.edges(data=True):
            weight = d.get("weight", 0)
            color  = cm.Blues(norm(weight))
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=[color],
                                   width=2, ax=ax, alpha=0.7)
        # Central node larger
        nx.draw_networkx_nodes(G, pos, nodelist=[gene_name],
                               node_color="#e74c3c", node_size=800, ax=ax)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=[n for n in G.nodes if n != gene_name],
                               node_color="#3498db", node_size=400, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

        sm = cm.ScalarMappable(cmap=cm.Blues, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="STRING confidence score")
        ax.set_title(f"Protein-protein interaction network — {gene_name}", fontsize=14)
        ax.axis("off")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"ppi_network_{gene_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        return {
            "gene": gene_name,
            "n_interactions": len(interactions),
            "top_interactors": [i["partner"] for i in interactions[:5]],
            "output_files": [out_path],
            "source": "STRING v12",
        }
    except ImportError as e:
        return {"error": f"Missing dependency: {e}. Run: pip install networkx matplotlib", "source": "PPI network"}
    except Exception as e:
        logger.error(f"plot_ppi_network error: {e}")
        return {"error": str(e), "source": "PPI network"}


# ─────────────────────────────────────────────────────────────────────────────
# Gene expression heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_expression_heatmap(expression_path: str, top_n: int = 50,
                            output_dir: str = "output/") -> dict:
    """
    Plot a heatmap of gene expression data (genes × samples).

    Args:
        expression_path: CSV/TSV file with genes as rows, samples as columns
        top_n: Number of most variable genes to show
        output_dir: Where to save the PNG

    Returns:
        dict with output_files
    """
    try:
        import pandas as pd
        import seaborn as sns
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(expression_path, index_col=0, sep=None, engine="python")

        # Select top N most variable genes
        var = df.var(axis=1).nlargest(top_n)
        df_top = df.loc[var.index]

        # Normalise per-gene (z-score)
        df_z = df_top.sub(df_top.mean(axis=1), axis=0).div(df_top.std(axis=1).replace(0, 1), axis=0)

        fig_h = max(8, top_n * 0.25)
        fig, ax = plt.subplots(figsize=(min(20, len(df.columns) * 0.8 + 4), fig_h))
        sns.heatmap(df_z, cmap="RdBu_r", center=0, linewidths=0,
                    yticklabels=True, xticklabels=True, ax=ax)
        ax.set_title(f"Expression heatmap — top {top_n} variable genes (z-score)")
        plt.tight_layout()

        out_path = os.path.join(output_dir, "expression_heatmap.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        return {
            "n_genes_shown": top_n,
            "n_samples": len(df.columns),
            "output_files": [out_path],
            "source": "matplotlib + seaborn",
        }
    except ImportError as e:
        return {"error": f"Missing dependency: {e}. Run: pip install seaborn pandas", "source": "heatmap"}
    except Exception as e:
        logger.error(f"plot_expression_heatmap error: {e}")
        return {"error": str(e), "source": "heatmap"}


# ─────────────────────────────────────────────────────────────────────────────
# Manhattan plot for GWAS
# ─────────────────────────────────────────────────────────────────────────────

def plot_manhattan(sumstats_path: str, chrom_col: str = "chr",
                   pos_col: str = "pos", pval_col: str = "pval",
                   snp_col: str = "snp", output_dir: str = "output/") -> dict:
    """
    Generate a Manhattan plot from GWAS summary statistics.

    Args:
        sumstats_path: TSV/CSV with chr, pos, pval columns
        chrom_col, pos_col, pval_col, snp_col: column names
        output_dir: Where to save the PNG

    Returns:
        dict with genome-wide significant hits and output_files
    """
    try:
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(sumstats_path, sep=None, engine="python")
        df[chrom_col] = df[chrom_col].astype(str).str.replace("chr", "", case=False)
        df[pval_col]  = pd.to_numeric(df[pval_col], errors="coerce")
        df = df.dropna(subset=[pval_col]).copy()
        df["-log10p"] = -np.log10(df[pval_col].clip(lower=1e-300))

        # Compute cumulative positions
        chromosomes = sorted(df[chrom_col].unique(), key=lambda x: int(x) if x.isdigit() else 99)
        offsets, tick_pos, tick_labels = {}, [], []
        offset = 0
        for chrom in chromosomes:
            sub = df[df[chrom_col] == chrom]
            offsets[chrom] = offset
            tick_pos.append(offset + (sub[pos_col].max() - sub[pos_col].min()) / 2)
            tick_labels.append(chrom)
            offset += sub[pos_col].max() - sub[pos_col].min() + 1e7

        df["x"] = df.apply(lambda r: offsets[r[chrom_col]] + r[pos_col], axis=1)

        # Plot
        colors = ["#2980b9", "#e74c3c"]
        fig, ax = plt.subplots(figsize=(18, 6))
        for i, chrom in enumerate(chromosomes):
            sub = df[df[chrom_col] == chrom]
            ax.scatter(sub["x"], sub["-log10p"], c=colors[i % 2], s=2, alpha=0.7, linewidths=0)

        ax.axhline(-np.log10(5e-8), color="red",   linestyle="--", linewidth=1, label="Genome-wide (5e-8)")
        ax.axhline(-np.log10(1e-5), color="orange", linestyle="--", linewidth=0.8, label="Suggestive (1e-5)")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("Chromosome")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title("Manhattan Plot")
        ax.legend(fontsize=9)
        plt.tight_layout()

        out_path = os.path.join(output_dir, "manhattan_plot.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        sig_hits = df[df[pval_col] < 5e-8]
        return {
            "n_variants": len(df),
            "n_significant": len(sig_hits),
            "top_hits": sig_hits.nsmallest(5, pval_col)[[snp_col, chrom_col, pos_col, pval_col]].to_dict("records") if not sig_hits.empty else [],
            "output_files": [out_path],
            "source": "matplotlib",
        }
    except ImportError as e:
        return {"error": f"Missing dependency: {e}", "source": "manhattan"}
    except Exception as e:
        logger.error(f"plot_manhattan error: {e}")
        return {"error": str(e), "source": "manhattan"}


# ─────────────────────────────────────────────────────────────────────────────
# R-based differential expression tools
# ─────────────────────────────────────────────────────────────────────────────

def run_deseq2_r(counts_path: str, metadata_path: str,
                 condition_col: str = "condition",
                 reference_level: str = "NULL",
                 output_dir: str = "output/") -> dict:
    """
    Run DESeq2 differential expression using a pre-written R script.
    Generates volcano plot, MA plot, and results CSV.

    Args:
        counts_path: CSV with genes as rows, samples as columns (integer counts)
        metadata_path: CSV with samples as rows, condition column
        condition_col: Column name in metadata for grouping
        reference_level: Reference condition (e.g. 'control', 'young')
        output_dir: Where to save results and plots

    Returns:
        dict with DE statistics and output_files (CSV + PNG plots)
    """
    import shutil
    if not shutil.which("Rscript"):
        return {"error": "Rscript not found. R is required for this tool.", "source": "DESeq2 (R)"}

    os.makedirs(output_dir, exist_ok=True)
    script = os.path.join(_SCRIPTS_DIR, "deseq2.R")

    try:
        result = subprocess.run(
            ["Rscript", script, counts_path, metadata_path,
             condition_col, reference_level or "NULL", output_dir],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            return {"error": f"DESeq2 failed:\n{result.stderr[:1000]}", "source": "DESeq2 (R)"}

        output_files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ]
        return {
            "stdout": result.stdout,
            "output_files": output_files,
            "source": "DESeq2 (R)",
        }
    except subprocess.TimeoutExpired:
        return {"error": "DESeq2 timed out (>10 min)", "source": "DESeq2 (R)"}
    except Exception as e:
        logger.error(f"run_deseq2_r error: {e}")
        return {"error": str(e), "source": "DESeq2 (R)"}


def run_limma(expression_path: str, metadata_path: str,
              condition_col: str = "condition",
              reference_level: str = "NULL",
              output_dir: str = "output/") -> dict:
    """
    Run limma differential expression using a pre-written R script.
    Works for both microarray and RNA-seq data (voom transform for counts).

    Args:
        expression_path: CSV with genes as rows, samples as columns
        metadata_path: CSV with samples as rows, condition column
        condition_col: Column name in metadata for grouping
        reference_level: Reference condition
        output_dir: Where to save results and plots

    Returns:
        dict with DE statistics and output_files (CSV + volcano PNG)
    """
    import shutil
    if not shutil.which("Rscript"):
        return {"error": "Rscript not found. R is required for this tool.", "source": "limma (R)"}

    os.makedirs(output_dir, exist_ok=True)
    script = os.path.join(_SCRIPTS_DIR, "limma.R")

    try:
        result = subprocess.run(
            ["Rscript", script, expression_path, metadata_path,
             condition_col, reference_level or "NULL", output_dir],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            return {"error": f"limma failed:\n{result.stderr[:1000]}", "source": "limma (R)"}

        output_files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ]
        return {
            "stdout": result.stdout,
            "output_files": output_files,
            "source": "limma (R)",
        }
    except Exception as e:
        logger.error(f"run_limma error: {e}")
        return {"error": str(e), "source": "limma (R)"}


def run_survival_analysis(data_path: str, time_col: str = "time",
                          event_col: str = "event", group_col: str = "group",
                          output_dir: str = "output/") -> dict:
    """
    Run Kaplan-Meier survival analysis and log-rank test using R.

    Args:
        data_path: CSV with time, event (0/1), and group columns
        time_col: Column name for survival time
        event_col: Column name for event (1=occurred, 0=censored)
        group_col: Column name for grouping
        output_dir: Where to save results and KM plot

    Returns:
        dict with log-rank p-value and output_files (KM plot PNG + summary CSV)
    """
    import shutil
    if not shutil.which("Rscript"):
        return {"error": "Rscript not found. R is required for this tool.", "source": "survival (R)"}

    os.makedirs(output_dir, exist_ok=True)
    script = os.path.join(_SCRIPTS_DIR, "survival.R")

    try:
        result = subprocess.run(
            ["Rscript", script, data_path, time_col, event_col, group_col, output_dir],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            return {"error": f"Survival analysis failed:\n{result.stderr[:1000]}", "source": "survival (R)"}

        output_files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ]
        return {
            "stdout": result.stdout,
            "output_files": output_files,
            "source": "survival (R)",
        }
    except Exception as e:
        logger.error(f"run_survival_analysis error: {e}")
        return {"error": str(e), "source": "survival (R)"}
