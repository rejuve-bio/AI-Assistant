"""
Biomni local data lake access layer.

All functions read from the mounted data volume at BIOMNI_DATA_LAKE env var
(default /data/biomni). The files are parquet/CSV datasets bundled with Biomni.
"""

import logging
import os

logger = logging.getLogger(__name__)
DATA_LAKE = os.environ.get("BIOMNI_DATA_LAKE", "/data/biomni")


def _parquet(filename: str):
    try:
        import pandas as pd
        path = os.path.join(DATA_LAKE, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data lake file not found: {path}. "
                f"Mount the Biomni data volume at {DATA_LAKE}."
            )
        return pd.read_parquet(path)
    except ImportError:
        raise ImportError("pandas not installed. Run: pip install pandas pyarrow")


def query_depmap(gene_name: str, top_k: int = 20) -> dict:
    """
    Get cancer cell line dependency scores for a gene from DepMap.
    Higher score = more essential in more cancer lines.

    Args:
        gene_name: Gene symbol (e.g. 'KRAS', 'MYC')
        top_k: Number of most dependent cell lines to return

    Returns:
        dict with keys: gene, mean_dependency_score, most_dependent_lines,
                        cancer_types_most_dependent
    """
    try:
        df = _parquet("depmap_gene_dependency.parquet")
        if gene_name not in df.columns:
            return {"gene": gene_name, "error": f"{gene_name} not found in DepMap"}
        col = df[gene_name].dropna().sort_values()
        return {
            "gene": gene_name,
            "mean_dependency_score": round(float(col.mean()), 4),
            "n_cell_lines": len(col),
            "most_dependent_lines": col.head(top_k).to_dict(),
            "source": "DepMap",
        }
    except Exception as e:
        return {"gene": gene_name, "error": str(e)}


def query_disgenets(gene_name: str, top_k: int = 20) -> dict:
    """
    Get gene-disease associations from DisGeNET.

    Args:
        gene_name: Gene symbol (e.g. 'BRCA1', 'APOE')
        top_k: Number of top disease associations to return

    Returns:
        dict with keys: gene, diseases (list of {disease_name, score, pmids})
    """
    try:
        df = _parquet("disgenets_gene_disease.parquet")
        matches = df[df["gene_symbol"].str.upper() == gene_name.upper()]
        if matches.empty:
            return {"gene": gene_name, "diseases": [], "error": "Not found in DisGeNET"}
        top = matches.nlargest(top_k, "score")[["disease_name", "score", "disease_id"]].to_dict("records")
        return {"gene": gene_name, "diseases": top, "source": "DisGeNET"}
    except Exception as e:
        return {"gene": gene_name, "error": str(e)}


def query_bindingdb(target_gene: str, top_k: int = 20) -> dict:
    """
    Get drug-protein binding affinities from BindingDB for a target protein.

    Args:
        target_gene: Target gene/protein name (e.g. 'EGFR', 'ABL1')
        top_k: Number of top binding compounds to return

    Returns:
        dict with keys: target, compounds (list of {name, smiles, ki_nm, ic50_nm})
    """
    try:
        df = _parquet("bindingdb_affinities.parquet")
        matches = df[df["target_gene"].str.upper().str.contains(target_gene.upper(), na=False)]
        if matches.empty:
            return {"target": target_gene, "compounds": [], "error": "Not found in BindingDB"}
        top = matches.nsmallest(top_k, "ki_nm")[
            ["ligand_name", "smiles", "ki_nm", "ic50_nm"]
        ].to_dict("records")
        return {"target": target_gene, "compounds": top, "source": "BindingDB"}
    except Exception as e:
        return {"target": target_gene, "error": str(e)}


def query_msigdb(gene_list: list, collection: str = "H") -> dict:
    """
    Check membership of genes in MSigDB gene sets.

    Args:
        gene_list: List of gene symbols to check
        collection: MSigDB collection. Options:
                    'H' = Hallmarks, 'C2' = Curated, 'C5' = GO,
                    'C6' = Oncogenic, 'C7' = Immunologic, 'C8' = Cell type

    Returns:
        dict with keys: gene_sets (list of {name, matched_genes, jaccard_similarity})
    """
    try:
        df = _parquet(f"msigdb_{collection.lower()}.parquet")
        gene_set_col = "gene_set"
        genes_col = "genes"
        input_set = set(g.upper() for g in gene_list)
        results = []
        for _, row in df.iterrows():
            gs_genes = set(str(row[genes_col]).upper().split(","))
            overlap = input_set & gs_genes
            if overlap:
                jaccard = len(overlap) / len(input_set | gs_genes)
                results.append({
                    "gene_set": row[gene_set_col],
                    "matched_genes": sorted(overlap),
                    "n_matched": len(overlap),
                    "jaccard_similarity": round(jaccard, 4),
                })
        results.sort(key=lambda x: x["jaccard_similarity"], reverse=True)
        return {
            "collection": collection,
            "n_genes_input": len(gene_list),
            "n_gene_sets_matched": len(results),
            "top_matches": results[:20],
            "source": "MSigDB",
        }
    except Exception as e:
        return {"gene_list": gene_list, "error": str(e)}


def query_omim(gene_name: str) -> dict:
    """
    Get genetic disorder associations for a gene from OMIM.

    Args:
        gene_name: Gene symbol (e.g. 'BRCA1', 'CFTR', 'HTT')

    Returns:
        dict with keys: gene, disorders (list of {name, mim_number, inheritance, phenotype})
    """
    try:
        df = _parquet("omim_gene_disorders.parquet")
        matches = df[df["gene_symbol"].str.upper() == gene_name.upper()]
        if matches.empty:
            return {"gene": gene_name, "disorders": [], "error": "Not found in OMIM data lake"}
        disorders = matches[["disorder_name", "mim_number", "inheritance", "phenotype_key"]].to_dict("records")
        return {"gene": gene_name, "disorders": disorders, "source": "OMIM"}
    except Exception as e:
        return {"gene": gene_name, "error": str(e)}


def query_precision_medicine_kg(entity: str, entity_type: str = "gene",
                                  relation: str = None, top_k: int = 20) -> dict:
    """
    Query the precision medicine knowledge graph (17,000 diseases, 4M relationships).

    Args:
        entity: Entity name (gene, disease, drug)
        entity_type: Type of entity: 'gene', 'disease', 'drug'
        relation: Optional specific relation type to filter by
        top_k: Number of results to return

    Returns:
        dict with keys: entity, neighbors (list of {target, relation, weight})
    """
    try:
        df = _parquet("precision_medicine_kg.parquet")
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
            "source": "Precision Medicine KG",
        }
    except Exception as e:
        return {"entity": entity, "error": str(e)}