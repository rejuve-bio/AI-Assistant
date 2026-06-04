"""
Biomni data lake access layer.

Each function tries a live REST API first. If BIOMNI_DATA_LAKE is set and the
relevant parquet file exists there, the local file is used (faster, offline).
This means the system works with zero local data — APIs are the default.
"""

import logging
import os
import requests

logger = logging.getLogger(__name__)
_BASE = os.environ.get("BIOMNI_DATA_LAKE", "/data/biomni")
DATA_LAKE = os.path.join(_BASE, "external")   # external API data lives here
_TIMEOUT = 15


def _try_parquet(filename: str):
    """Load a local parquet if it exists, else return None."""
    try:
        import pandas as pd
        path = os.path.join(DATA_LAKE, filename)
        if os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DepMap — cancer cell line gene dependency scores
# ─────────────────────────────────────────────────────────────────────────────

def query_depmap(gene_name: str, top_k: int = 20) -> dict:
    """
    Get cancer cell line dependency scores for a gene from DepMap.
    Higher (more negative) score = gene is more essential in that cell line.
    """
    df = _try_parquet("depmap_gene_dependency.parquet")
    if df is not None and gene_name in df.columns:
        col = df[gene_name].dropna().sort_values()
        return {
            "gene": gene_name,
            "mean_dependency_score": round(float(col.mean()), 4),
            "n_cell_lines": len(col),
            "most_dependent_lines": col.head(top_k).to_dict(),
            "source": "DepMap (local)",
        }

    # DepMap REST API
    try:
        url = "https://depmap.org/portal/api/gene/summary"
        r = requests.get(url, params={"gene": gene_name}, timeout=_TIMEOUT)
        if r.ok:
            data = r.json()
            return {
                "gene": gene_name,
                "mean_crispr_score": data.get("mean_crispr_ko"),
                "n_dependent_lines": data.get("n_dependent_cell_lines"),
                "top_lineages": data.get("top_lineages", [])[:5],
                "source": "DepMap API",
            }
    except Exception as e:
        logger.warning(f"DepMap API failed: {e}")

    return {"gene": gene_name, "error": "DepMap data unavailable"}


# ─────────────────────────────────────────────────────────────────────────────
# DisGeNET — gene-disease associations
# ─────────────────────────────────────────────────────────────────────────────

def query_disgenets(gene_name: str, top_k: int = 20) -> dict:
    """Get gene-disease associations from DisGeNET."""
    df = _try_parquet("disgenets_gene_disease.parquet")
    if df is not None:
        matches = df[df["gene_symbol"].str.upper() == gene_name.upper()]
        if not matches.empty:
            top = matches.nlargest(top_k, "score")[["disease_name", "score", "disease_id"]].to_dict("records")
            return {"gene": gene_name, "diseases": top, "source": "DisGeNET (local)"}

    # DisGeNET v7 REST API
    try:
        url = f"https://www.disgenet.org/api/gda/gene/{gene_name}"
        r = requests.get(url, headers={"accept": "application/json"},
                         params={"limit": top_k}, timeout=_TIMEOUT)
        if r.ok:
            data = r.json()
            diseases = [
                {
                    "disease_name": d.get("disease_name"),
                    "score": round(d.get("score", 0), 4),
                    "disease_id": d.get("diseaseId"),
                }
                for d in data[:top_k]
            ]
            return {"gene": gene_name, "diseases": diseases, "source": "DisGeNET API"}
    except Exception as e:
        logger.warning(f"DisGeNET API failed: {e}")

    # Fallback: Open Targets covers most DisGeNET content
    try:
        from .database_connectors import query_opentargets
        result = query_opentargets(gene_name, limit=top_k)
        if not result.get("error"):
            result["source"] = "Open Targets (DisGeNET fallback)"
            return result
    except Exception as e:
        logger.warning(f"Open Targets fallback failed: {e}")

    return {"gene": gene_name, "diseases": [], "error": "DisGeNET data unavailable"}


# ─────────────────────────────────────────────────────────────────────────────
# BindingDB — drug-protein binding affinities
# ─────────────────────────────────────────────────────────────────────────────

def query_bindingdb(target_gene: str, top_k: int = 20) -> dict:
    """Get drug-protein binding affinities from BindingDB."""
    df = _try_parquet("bindingdb_affinities.parquet")
    if df is not None:
        matches = df[df["target_gene"].str.upper().str.contains(target_gene.upper(), na=False)]
        if not matches.empty:
            top = matches.nsmallest(top_k, "ki_nm")[
                ["ligand_name", "smiles", "ki_nm", "ic50_nm"]
            ].to_dict("records")
            return {"target": target_gene, "compounds": top, "source": "BindingDB (local)"}

    # BindingDB REST API
    try:
        url = "https://bindingdb.org/axis2/services/BDBService/getLigandsByTarget"
        r = requests.get(url, params={"targetname": target_gene, "response": "json"}, timeout=_TIMEOUT)
        if r.ok:
            data = r.json()
            affinities = data.get("getLigandsByTargetResponse", {}).get("affinities", [])
            compounds = [
                {
                    "ligand_name": a.get("compound_name", ""),
                    "ki_nm": a.get("ki", ""),
                    "ic50_nm": a.get("ic50", ""),
                    "smiles": a.get("smile", ""),
                }
                for a in affinities[:top_k]
            ]
            return {"target": target_gene, "compounds": compounds, "source": "BindingDB API"}
    except Exception as e:
        logger.warning(f"BindingDB API failed: {e}")

    return {"target": target_gene, "compounds": [], "error": "BindingDB data unavailable"}


# ─────────────────────────────────────────────────────────────────────────────
# MSigDB — gene set membership
# ─────────────────────────────────────────────────────────────────────────────

def query_msigdb(gene_list: list, collection: str = "H") -> dict:
    """
    Check membership of genes in MSigDB gene sets.
    Uses gseapy (installed) to query MSigDB directly — no local data needed.
    """
    df = _try_parquet(f"msigdb_{collection.lower()}.parquet")
    if df is not None:
        input_set = set(g.upper() for g in gene_list)
        results = []
        for _, row in df.iterrows():
            gs_genes = set(str(row.get("genes", "")).upper().split(","))
            overlap = input_set & gs_genes
            if overlap:
                jaccard = len(overlap) / len(input_set | gs_genes)
                results.append({
                    "gene_set": row.get("gene_set", ""),
                    "matched_genes": sorted(overlap),
                    "n_matched": len(overlap),
                    "jaccard_similarity": round(jaccard, 4),
                })
        results.sort(key=lambda x: x["jaccard_similarity"], reverse=True)
        return {"collection": collection, "top_matches": results[:20], "source": "MSigDB (local)"}

    # gseapy queries MSigDB directly
    try:
        import gseapy as gp
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=_msigdb_name(collection),
            outdir=None,
            verbose=False,
        )
        top = [
            {
                "gene_set": row.get("Term", ""),
                "p_value": float(row.get("P-value", 1.0)),
                "adjusted_p": float(row.get("Adjusted P-value", 1.0)),
                "overlap": row.get("Overlap", ""),
            }
            for _, row in enr.results.head(20).iterrows()
        ]
        return {"collection": collection, "n_genes_input": len(gene_list), "top_matches": top, "source": "MSigDB via GSEApy"}
    except Exception as e:
        logger.warning(f"GSEApy MSigDB query failed: {e}")

    return {"gene_list": gene_list, "error": "MSigDB data unavailable"}


def _msigdb_name(collection: str) -> str:
    return {
        "H": "MSigDB_Hallmark_2020",
        "C2": "KEGG_2021_Human",
        "C5": "GO_Biological_Process_2023",
        "C6": "MSigDB_Oncogenic_Signatures",
        "C7": "MSigDB_Immunologic_Signatures",
    }.get(collection.upper(), "MSigDB_Hallmark_2020")


# ─────────────────────────────────────────────────────────────────────────────
# OMIM — genetic disorder associations
# ─────────────────────────────────────────────────────────────────────────────

def query_omim(gene_name: str) -> dict:
    """Get genetic disorder associations. Falls back to Open Targets if local data absent."""
    df = _try_parquet("omim_gene_disorders.parquet")
    if df is not None:
        matches = df[df["gene_symbol"].str.upper() == gene_name.upper()]
        if not matches.empty:
            disorders = matches[["disorder_name", "mim_number", "inheritance"]].to_dict("records")
            return {"gene": gene_name, "disorders": disorders, "source": "OMIM (local)"}

    try:
        from .database_connectors import query_opentargets
        result = query_opentargets(gene_name, limit=15)
        if not result.get("error"):
            result["note"] = "Sourced from Open Targets (local OMIM data not mounted)"
            return result
    except Exception as e:
        logger.warning(f"Open Targets OMIM fallback failed: {e}")

    return {"gene": gene_name, "disorders": [], "error": "OMIM data unavailable (mount BIOMNI_DATA_LAKE to enable)"}


# ─────────────────────────────────────────────────────────────────────────────
# Precision Medicine Knowledge Graph — local only
# ─────────────────────────────────────────────────────────────────────────────

def query_precision_medicine_kg(entity: str, entity_type: str = "gene",
                                 relation: str = None, top_k: int = 20) -> dict:
    """
    Query the precision medicine knowledge graph (17,000 diseases, 4M relationships).
    Requires local data lake — no public API available.
    """
    df = _try_parquet("precision_medicine_kg.parquet")
    if df is None:
        return {
            "entity": entity,
            "error": (
                f"Precision Medicine KG not found at {DATA_LAKE}/precision_medicine_kg.parquet. "
                "Mount the Biomni data volume and set BIOMNI_DATA_LAKE env var."
            ),
        }
    try:
        mask = df["source_name"].str.lower().str.contains(entity.lower(), na=False)
        if entity_type:
            mask = mask & (df["source_type"].str.lower() == entity_type.lower())
        if relation:
            mask = mask & (df["relation"].str.lower() == relation.lower())
        matches = df[mask].head(top_k)
        if matches.empty:
            return {"entity": entity, "neighbors": [], "error": "Not found in knowledge graph"}
        neighbors = matches[["target_name", "target_type", "relation", "weight"]].to_dict("records")
        return {
            "entity": entity,
            "entity_type": entity_type,
            "n_neighbors": len(neighbors),
            "neighbors": neighbors,
            "source": "Precision Medicine KG (local)",
        }
    except Exception as e:
        return {"entity": entity, "error": str(e)}
