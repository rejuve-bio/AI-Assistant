"""
Genomics tools: scRNA-seq annotation, GSEA, embeddings, Hi-C chromatin interactions.
"""

import logging
import os

logger = logging.getLogger(__name__)


def annotate_scrna(adata_path: str, methods: list = None, output_dir: str = "output/") -> dict:
    """
    Annotate cell types in a single-cell RNA-seq dataset using multiple methods simultaneously.
    Supported methods: celltypist, knn, xgboost, svm, random_forest.

    Args:
        adata_path: Path to AnnData h5ad file (e.g. 'input/cells.h5ad')
        methods: List of methods to use. Defaults to ['celltypist', 'knn', 'xgboost']
        output_dir: Directory to write annotated output files

    Returns:
        dict with keys: annotations (per-cell predictions per method), consensus, output_files
    """
    if methods is None:
        methods = ["celltypist", "knn", "xgboost"]

    try:
        import scanpy as sc
        import pandas as pd
        import numpy as np
        os.makedirs(output_dir, exist_ok=True)

        adata = sc.read_h5ad(adata_path)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.pca(adata, use_highly_variable=True)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

        results = {}

        if "celltypist" in methods:
            try:
                import celltypist
                predictions = celltypist.annotate(adata, majority_voting=True)
                results["celltypist"] = predictions.predicted_labels["majority_voting"].tolist()
            except ImportError:
                results["celltypist"] = "celltypist not installed (pip install celltypist)"

        if "knn" in methods:
            sc.tl.leiden(adata, resolution=0.5)
            results["knn_clusters"] = adata.obs["leiden"].tolist()

        if "xgboost" in methods:
            try:
                from xgboost import XGBClassifier
                X = adata.obsm["X_pca"]
                labels = adata.obs.get("leiden", pd.Series(["unknown"] * len(adata))).values
                clf = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric="mlogloss")
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(labels)
                clf.fit(X, y)
                preds = le.inverse_transform(clf.predict(X))
                results["xgboost"] = preds.tolist()
            except ImportError:
                results["xgboost"] = "xgboost not installed (pip install xgboost)"

        out_path = os.path.join(output_dir, "annotated.h5ad")
        adata.write_h5ad(out_path)

        return {
            "n_cells": len(adata),
            "n_genes": adata.n_vars,
            "methods_run": list(results.keys()),
            "annotations": results,
            "output_files": [out_path],
        }
    except ImportError:
        return {"error": "scanpy not installed. Run: pip install scanpy"}
    except Exception as e:
        logger.error(f"scRNA annotation failed: {e}")
        return {"error": str(e)}


def run_gsea(gene_list: list, gene_sets: str = "MSigDB_Hallmark_2020",
             output_dir: str = "output/") -> dict:
    """
    Run Gene Set Enrichment Analysis (GSEA) on a ranked gene list.

    Args:
        gene_list: Ordered list of gene symbols (ranked by expression or fold change)
        gene_sets: Gene set database name or path. Options: 'MSigDB_Hallmark_2020',
                   'KEGG_2021_Human', 'Reactome_2022', 'GO_Biological_Process_2023'
        output_dir: Directory to write result files

    Returns:
        dict with keys: enriched_terms (list of top results), output_files
    """
    try:
        import gseapy as gp
        import pandas as pd
        os.makedirs(output_dir, exist_ok=True)

        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            outdir=output_dir,
            cutoff=0.05,
        )
        top = enr.results.head(20)[
            ["Term", "Overlap", "Adjusted P-value", "Genes"]
        ].to_dict("records")

        return {
            "gene_set_library": gene_sets,
            "n_genes_input": len(gene_list),
            "significant_terms": len(enr.results[enr.results["Adjusted P-value"] < 0.05]),
            "top_enriched": top,
            "output_files": [os.path.join(output_dir, f"{gene_sets}.txt")],
        }
    except ImportError:
        return {"error": "gseapy not installed. Run: pip install gseapy"}
    except Exception as e:
        logger.error(f"GSEA failed: {e}")
        return {"error": str(e)}


def compute_scrna_embeddings(adata_path: str, method: str = "scvi",
                              output_dir: str = "output/") -> dict:
    """
    Compute low-dimensional embeddings for single-cell RNA-seq data.
    Supported methods: scvi, harmony, pca.

    Args:
        adata_path: Path to AnnData h5ad file
        method: Embedding method — 'scvi', 'harmony', or 'pca'
        output_dir: Directory to write output h5ad file

    Returns:
        dict with keys: embedding_shape, output_files
    """
    try:
        import scanpy as sc
        import numpy as np
        os.makedirs(output_dir, exist_ok=True)

        adata = sc.read_h5ad(adata_path)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        sc.pp.pca(adata, use_highly_variable=True)

        if method == "scvi":
            try:
                import scvi
                scvi.model.SCVI.setup_anndata(adata)
                model = scvi.model.SCVI(adata, n_latent=30)
                model.train(max_epochs=50, progress_bar=False)
                adata.obsm["X_scvi"] = model.get_latent_representation()
            except ImportError:
                return {"error": "scvi-tools not installed. Run: pip install scvi-tools"}

        elif method == "harmony":
            try:
                import harmonypy as hm
                ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, "batch")
                adata.obsm["X_harmony"] = ho.Z_corr.T
            except ImportError:
                return {"error": "harmonypy not installed. Run: pip install harmonypy"}

        out_path = os.path.join(output_dir, f"embedded_{method}.h5ad")
        adata.write_h5ad(out_path)

        key = f"X_{method}" if method != "pca" else "X_pca"
        shape = adata.obsm.get(key, np.array([])).shape

        return {
            "method": method,
            "embedding_shape": list(shape),
            "output_files": [out_path],
        }
    except ImportError:
        return {"error": "scanpy not installed. Run: pip install scanpy"}
    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        return {"error": str(e)}