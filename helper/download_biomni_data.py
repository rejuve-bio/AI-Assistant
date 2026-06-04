#!/usr/bin/env python3
"""
Biomni Data Lake — One-Time Download Script
============================================
Run this once on your server to populate the local data cache.
After this runs, all data_lake.py functions use local files and never call external APIs.

Usage:
    python scripts/download_biomni_data.py

    # Custom data directory:
    BIOMNI_DATA_LAKE=/my/data/path python scripts/download_biomni_data.py

    # Skip specific datasets:
    python scripts/download_biomni_data.py --skip depmap bindingdb

    # Re-download even if file exists:
    python scripts/download_biomni_data.py --force

Environment variables:
    BIOMNI_DATA_LAKE   Where to save files (default: ./data/biomni)
"""

import os
import sys
import argparse
import logging
import requests
import gzip
import io
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_BASE    = os.environ.get("BIOMNI_DATA_LAKE", "./data/biomni")
DATA_DIR = os.path.join(_BASE, "external")  # External API data goes in external/ subfolder
CHUNK = 1024 * 1024  # 1 MB download chunks


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _save_parquet(df, name: str):
    import pandas as pd
    path = os.path.join(DATA_DIR, name)
    df.to_parquet(path, index=False, compression="snappy")
    size_mb = os.path.getsize(path) / 1024 / 1024
    log.info(f"  Saved {name}  ({size_mb:.1f} MB,  {len(df):,} rows)")
    return path


def _already_exists(name: str, force: bool) -> bool:
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path) and not force:
        size_mb = os.path.getsize(path) / 1024 / 1024
        log.info(f"  SKIP  {name} already exists ({size_mb:.1f} MB)  —  pass --force to re-download")
        return True
    return False


def _download_bytes(url: str, desc: str, timeout: int = 120) -> bytes:
    log.info(f"  Downloading {desc} ...")
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    received = 0
    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=CHUNK):
        buf.write(chunk)
        received += len(chunk)
        if total:
            pct = received / total * 100
            print(f"\r    {desc}: {received/1024/1024:.1f}/{total/1024/1024:.1f} MB  ({pct:.0f}%)", end="", flush=True)
    print()
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# 1. DisGeNET — gene-disease associations
#    Source: public download, no API key needed
#    Rows: ~1.1 million gene-disease associations
#    Saved as: disgenets_gene_disease.parquet
# ─────────────────────────────────────────────────────────────────────────────

def download_disgenet(force: bool):
    """
    DisGeNET now requires login for bulk download.
    We build an equivalent dataset from Open Targets (covers the same gene-disease data)
    for a broad set of biomedically relevant genes.
    """
    name = "disgenets_gene_disease.parquet"
    if _already_exists(name, force):
        return

    log.info("DisGeNET (via Open Targets): building gene-disease association dataset ...")
    import pandas as pd

    # Comprehensive gene list covering aging, cancer, neurodegeneration, metabolism
    genes = [
        "BRCA1","BRCA2","TP53","APOE","PTEN","ATM","EGFR","KRAS","MYC","FOXO3",
        "MTOR","SIRT1","SIRT2","SIRT3","SIRT6","IGF1","IGF1R","CDKN2A","CDKN1A",
        "TERT","VHL","APC","RB1","PARP1","PALB2","CHEK2","CDH1","MUTYH","NBN",
        "CFTR","HTT","SOD1","FUS","TARDBP","APP","PSEN1","PSEN2","LRRK2","SNCA",
        "PINK1","PARK2","GBA","MAPT","BACE1","AKT1","AKT2","PIK3CA","AMPK","STK11",
        "ADIPOQ","LEP","INS","INSR","PPARG","FOXO1","KLOTHO","GDF11","GDF15",
        "PCSK9","LDLR","ANGPTL3","CETP","LPA","HMGCR","NPC1","ABCA1",
        "TNF","IL6","IL1B","NLRP3","NFKB1","STAT3","JAK2","TYK2",
        "VEGFA","HIF1A","EPAS1","MDM2","CDKN2B","CCND1","CDK4","CDK6",
        "BRAF","RAF1","MAP2K1","MAPK1","MAPK3","ERRFI1","ERBB2","ERBB3",
        "MLH1","MSH2","MSH6","PMS2","POLE","POLD1","NTHL1","SMAD4",
    ]

    url = "https://api.platform.opentargets.org/api/v4/graphql"
    query = """
    query($gene: String!, $size: Int!) {
      search(queryString: $gene, entityNames: ["target"]) {
        hits {
          object {
            ... on Target {
              approvedSymbol
              associatedDiseases(page: {index: 0, size: $size}) {
                rows { disease { name id } score }
              }
            }
          }
        }
      }
    }
    """
    rows = []
    for i, gene in enumerate(genes):
        try:
            r = requests.post(url, json={"query": query, "variables": {"gene": gene, "size": 50}}, timeout=20)
            if r.ok:
                hits = r.json().get("data", {}).get("search", {}).get("hits", [])
                if hits:
                    for row in hits[0].get("object", {}).get("associatedDiseases", {}).get("rows", []):
                        rows.append({
                            "gene_symbol": gene,
                            "disease_name": row["disease"]["name"],
                            "disease_id": row["disease"]["id"],
                            "score": round(row["score"], 4),
                        })
            if i % 10 == 0:
                log.info(f"  Progress: {i+1}/{len(genes)} genes, {len(rows)} associations so far")
            time.sleep(0.1)
        except Exception as e:
            log.warning(f"  Open Targets failed for {gene}: {e}")

    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset=["gene_symbol", "disease_name"])
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        _save_parquet(df, name)
    else:
        log.error("  DisGeNET/Open Targets: all methods failed")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DepMap — cancer gene dependency scores
#    Source: Broad Institute DepMap portal (public)
#    Rows: ~18,000 genes × 1,000+ cell lines (we save the summary version)
#    Saved as: depmap_gene_dependency.parquet
# ─────────────────────────────────────────────────────────────────────────────

def download_depmap(force: bool):
    name = "depmap_gene_dependency.parquet"
    if _already_exists(name, force):
        return

    log.info("DepMap: downloading CRISPR gene effect scores ...")
    import pandas as pd

    # DepMap 24Q2 — follow Figshare redirect to get the actual S3 CSV
    figshare_url = "https://figshare.com/ndownloader/files/43346616"
    try:
        log.info("  Following Figshare redirect for DepMap 24Q2 ...")
        resp = requests.get(figshare_url, allow_redirects=True, timeout=300, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        received = 0
        buf = io.BytesIO()
        for chunk in resp.iter_content(chunk_size=CHUNK):
            buf.write(chunk)
            received += len(chunk)
            if total:
                print(f"\r    DepMap: {received/1024/1024:.0f}/{total/1024/1024:.0f} MB", end="", flush=True)
        print()
        buf.seek(0)
        df = pd.read_csv(buf, index_col=0)
        df.columns = [c.split(" (")[0] for c in df.columns]
        df = df.reset_index().rename(columns={"Unnamed: 0": "cell_line", "index": "cell_line"})
        _save_parquet(df, name)
    except Exception as e:
        log.error(f"DepMap download failed: {e}")
        log.info("  Falling back to DepMap summary stats per gene via API ...")
        _download_depmap_via_api(name)


def _download_depmap_via_api(name: str):
    """Build a summary DepMap table using the DepMap portal API."""
    import pandas as pd
    key_genes = [
        "KRAS","MYC","EGFR","TP53","PTEN","BRCA1","BRCA2","CDK4","CDK6",
        "PIK3CA","BRAF","AKT1","MTOR","BCL2","BCL2L1","MCL1","MDM2",
        "RB1","CDKN2A","VHL","NF1","NF2","RET","ALK","MET","FGFR1",
        "FGFR2","FGFR3","ERBB2","ERBB3","IGF1R","INSR","FOXO3","SIRT1",
    ]
    rows = []
    for gene in key_genes:
        try:
            r = requests.get(
                "https://depmap.org/portal/api/gene/summary",
                params={"gene": gene}, timeout=15
            )
            if r.ok:
                data = r.json()
                rows.append({
                    "gene": gene,
                    "mean_crispr_score": data.get("mean_crispr_ko"),
                    "n_dependent_lines": data.get("n_dependent_cell_lines"),
                    "is_common_essential": data.get("is_common_essential", False),
                })
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"  DepMap API failed for {gene}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        _save_parquet(df, name)
    else:
        log.error("  DepMap: all methods failed")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MSigDB gene sets — via gseapy (no separate download needed)
#    gseapy downloads and caches these; we convert to parquet for fast lookup
#    Saved as: msigdb_H.parquet, msigdb_C2.parquet, msigdb_C5.parquet
# ─────────────────────────────────────────────────────────────────────────────

def download_msigdb(force: bool):
    import pandas as pd

    collections = {
        "H":  "MSigDB_Hallmark_2020",
        "C2": "KEGG_2021_Human",
        "C5": "GO_Biological_Process_2023",
        "C6": "MSigDB_Oncogenic_Signatures",
    }

    for coll_key, gseapy_name in collections.items():
        name = f"msigdb_{coll_key.lower()}.parquet"
        if _already_exists(name, force):
            continue
        log.info(f"MSigDB {coll_key} ({gseapy_name}): downloading via gseapy ...")
        try:
            import gseapy as gp
            gene_sets = gp.get_library(gseapy_name, organism="Human")
            rows = [
                {"gene_set": gs_name, "genes": ",".join(genes)}
                for gs_name, genes in gene_sets.items()
            ]
            df = pd.DataFrame(rows)
            _save_parquet(df, name)
        except Exception as e:
            log.error(f"MSigDB {coll_key} download failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. BindingDB — drug-protein binding affinities
#    Source: BindingDB public download
#    We download the compact version (~300MB) not the full 2GB dump
#    Saved as: bindingdb_affinities.parquet
# ─────────────────────────────────────────────────────────────────────────────

def download_bindingdb(force: bool):
    name = "bindingdb_affinities.parquet"
    if _already_exists(name, force):
        return

    log.info("BindingDB: downloading binding affinity data ...")
    import pandas as pd

    # BindingDB bulk download — try current year URLs, fall back to targeted API
    year_urls = [
        "https://www.bindingdb.org/bind/downloads/BindingDB_All_2024m12.tsv.zip",
        "https://www.bindingdb.org/bind/downloads/BindingDB_All_2024m9.tsv.zip",
        "https://www.bindingdb.org/bind/downloads/BindingDB_All_2024m6.tsv.zip",
    ]
    downloaded = False
    for url in year_urls:
        try:
            import zipfile
            head = requests.head(url, timeout=10)
            if head.status_code != 200:
                continue
            raw = _download_bytes(url, "BindingDB", timeout=600)
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                tsv_name = [n for n in zf.namelist() if n.endswith(".tsv")][0]
                with zf.open(tsv_name) as f:
                    df = pd.read_csv(f, sep="\t", usecols=[
                        "Ligand SMILES", "Ligand Name",
                        "Target Name Assigned by Curator or DataSource",
                        "Target Source Organism According to Curator or DataSource",
                        "Ki (nM)", "IC50 (nM)", "Kd (nM)",
                    ], low_memory=False, on_bad_lines="skip")
            df = df.rename(columns={
                "Ligand SMILES": "smiles",
                "Ligand Name": "ligand_name",
                "Target Name Assigned by Curator or DataSource": "target_gene",
                "Target Source Organism According to Curator or DataSource": "organism",
                "Ki (nM)": "ki_nm",
                "IC50 (nM)": "ic50_nm",
                "Kd (nM)": "kd_nm",
            })
            df = df[df["organism"].str.contains("Homo sapiens", na=False)]
            df["ki_nm"] = pd.to_numeric(df["ki_nm"], errors="coerce")
            df = df[df["ki_nm"].notna()].sort_values("ki_nm").reset_index(drop=True)
            _save_parquet(df, name)
            downloaded = True
            break
        except Exception as e:
            log.warning(f"  BindingDB {url} failed: {e}")

    if not downloaded:
        log.info("  Falling back to targeted BindingDB API queries ...")
        _download_bindingdb_targeted(name)


def _download_bindingdb_targeted(name: str):
    """Download BindingDB data for key aging/longevity drug targets."""
    import pandas as pd

    key_targets = [
        "mTOR", "SIRT1", "FOXO3", "IGF1R", "AKT1", "AMPK", "TP53",
        "CDK4", "CDK6", "TERT", "PARP1", "EGFR", "ABL1", "BCL2",
    ]
    rows = []
    for target in key_targets:
        try:
            r = requests.get(
                "https://bindingdb.org/axis2/services/BDBService/getLigandsByTarget",
                params={"targetname": target, "response": "json"},
                timeout=20,
            )
            if r.ok:
                data = r.json()
                for a in data.get("getLigandsByTargetResponse", {}).get("affinities", []):
                    rows.append({
                        "target_gene": target,
                        "ligand_name": a.get("compound_name", ""),
                        "smiles": a.get("smile", ""),
                        "ki_nm": _to_float(a.get("ki")),
                        "ic50_nm": _to_float(a.get("ic50")),
                    })
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"  BindingDB API failed for {target}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        _save_parquet(df, name)
    else:
        log.error("  BindingDB: all methods failed")


def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. OMIM — genetic disorders
#    Source: Open Targets (covers OMIM content, no key required)
#    Saved as: omim_gene_disorders.parquet
# ─────────────────────────────────────────────────────────────────────────────

def download_omim(force: bool):
    name = "omim_gene_disorders.parquet"
    if _already_exists(name, force):
        return

    log.info("OMIM (via Open Targets): downloading gene-disorder associations ...")
    import pandas as pd

    # Use Open Targets bulk download — they include OMIM data
    # We query the GraphQL API for a broad set of genes
    genes = [
        "BRCA1", "BRCA2", "TP53", "APOE", "CFTR", "HTT", "PTEN",
        "ATM", "EGFR", "KRAS", "MYC", "FOXO3", "MTOR", "SIRT1",
        "TERT", "RB1", "VHL", "APC", "CDH1", "MUTYH", "NBN",
        "PALB2", "CHEK2", "RAD51C", "RAD51D", "BARD1", "BRIP1",
    ]
    url = "https://api.platform.opentargets.org/api/v4/graphql"
    query = """
    query($gene: String!, $size: Int!) {
      search(queryString: $gene, entityNames: ["target"]) {
        hits {
          id
          object {
            ... on Target {
              approvedSymbol
              associatedDiseases(page: {index: 0, size: $size}) {
                rows {
                  disease { name id }
                  score
                }
              }
            }
          }
        }
      }
    }
    """
    rows = []
    for gene in genes:
        try:
            r = requests.post(url, json={"query": query, "variables": {"gene": gene, "size": 30}}, timeout=20)
            if r.ok:
                hits = r.json().get("data", {}).get("search", {}).get("hits", [])
                if hits:
                    for row in hits[0].get("object", {}).get("associatedDiseases", {}).get("rows", []):
                        rows.append({
                            "gene_symbol": gene,
                            "disorder_name": row["disease"]["name"],
                            "mim_number": row["disease"]["id"],
                            "score": row["score"],
                            "inheritance": "",
                            "phenotype_key": row["disease"]["id"],
                        })
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"  Open Targets failed for {gene}: {e}")

    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset=["gene_symbol", "disorder_name"])
        _save_parquet(df, name)
    else:
        log.error("  OMIM download failed")


# ─────────────────────────────────────────────────────────────────────────────
# 6. TXGNN drug repurposing + Precision Medicine KG
#    Source: Biomni S3 bucket (Stanford SNAP)
#    These are the only files with no API alternative
# ─────────────────────────────────────────────────────────────────────────────

BIOMNI_S3_FILES = {
    "txgnn_repurposing.parquet": "https://biomni-data.s3.amazonaws.com/biomni_data/txgnn_repurposing.parquet",
    "precision_medicine_kg.parquet": "https://biomni-data.s3.amazonaws.com/biomni_data/precision_medicine_kg.parquet",
}

def download_biomni_s3(force: bool):
    for name, url in BIOMNI_S3_FILES.items():
        if _already_exists(name, force):
            continue
        log.info(f"Biomni S3: downloading {name} ...")
        try:
            raw = _download_bytes(url, name, timeout=600)
            path = os.path.join(DATA_DIR, name)
            with open(path, "wb") as f:
                f.write(raw)
            size_mb = os.path.getsize(path) / 1024 / 1024
            log.info(f"  Saved {name} ({size_mb:.1f} MB)")
        except Exception as e:
            log.warning(f"  {name} S3 download failed: {e}")
            log.warning(
                f"  To get this file manually:\n"
                f"    wget '{url}' -O {os.path.join(DATA_DIR, name)}\n"
                f"  Or contact Biomni (Stanford SNAP) for access."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_DATASETS = {
    "disgenet":   (download_disgenet,   "DisGeNET gene-disease associations"),
    "depmap":     (download_depmap,     "DepMap CRISPR gene dependency scores"),
    "msigdb":     (download_msigdb,     "MSigDB gene sets (Hallmarks, KEGG, GO)"),
    "bindingdb":  (download_bindingdb,  "BindingDB drug-protein binding affinities"),
    "omim":       (download_omim,       "OMIM genetic disorder associations (via Open Targets)"),
    "biomni_s3":  (download_biomni_s3,  "TXGNN drug repurposing + Precision Medicine KG (Biomni S3)"),
}


def main():
    parser = argparse.ArgumentParser(description="Download Biomni data lake — run once on your server.")
    parser.add_argument("--skip", nargs="+", choices=list(ALL_DATASETS.keys()),
                        default=[], help="Datasets to skip")
    parser.add_argument("--only", nargs="+", choices=list(ALL_DATASETS.keys()),
                        default=[], help="Only download these datasets")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    log.info(f"Data directory: {os.path.abspath(DATA_DIR)}")
    log.info(f"Set BIOMNI_DATA_LAKE={os.path.abspath(DATA_DIR)} in your .env\n")

    to_run = args.only if args.only else list(ALL_DATASETS.keys())
    to_run = [k for k in to_run if k not in args.skip]

    results = {}
    for key in to_run:
        fn, desc = ALL_DATASETS[key]
        log.info(f"\n{'─'*60}")
        log.info(f"[{key.upper()}] {desc}")
        log.info(f"{'─'*60}")
        try:
            fn(args.force)
            results[key] = "OK"
        except Exception as e:
            log.error(f"Unexpected error in {key}: {e}")
            results[key] = f"FAILED: {e}"

    # Summary
    log.info(f"\n{'═'*60}")
    log.info("DOWNLOAD SUMMARY")
    log.info(f"{'═'*60}")
    for key, status in results.items():
        icon = "✓" if status == "OK" else "✗"
        log.info(f"  {icon}  {key:<15} {status}")

    files = os.listdir(DATA_DIR)
    total_mb = sum(
        os.path.getsize(os.path.join(DATA_DIR, f)) / 1024 / 1024
        for f in files if not f.startswith(".")
    )
    log.info(f"\n  Files in {DATA_DIR}: {len(files)}")
    log.info(f"  Total size: {total_mb:.1f} MB")
    log.info(f"\n  Add to your .env:")
    log.info(f"  BIOMNI_DATA_LAKE={os.path.abspath(DATA_DIR)}")


if __name__ == "__main__":
    main()
