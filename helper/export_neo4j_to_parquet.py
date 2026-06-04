#!/usr/bin/env python3
"""
Neo4j → Parquet Export Script
==============================
Exports Rejuve Atomspace entities to parquet files so the AI assistant can
query them without Neo4j credentials. Files land in BIOMNI_DATA_LAKE/neo4j/.

The running app auto-detects file changes — no restart needed after export.
Optionally call POST /admin/reload-parquet to pre-warm the cache immediately.

── AUTOMATION ────────────────────────────────────────────────────────────────
Option A: Run after every Neo4j import (recommended)
  Add to your Neo4j import pipeline as a post-step:
    python /AI-Assistant/helper/export_neo4j_to_parquet.py --only genes pathways
    curl -X POST http://ai-assistant:5002/admin/reload-parquet -H "Authorization: Bearer $TOKEN"

Option B: Cron job (weekly / after each monthly import)
  Add to crontab on the server that has Neo4j access:
    0 3 * * 0  cd /AI-Assistant && NEO4J_URI=bolt://... NEO4J_USERNAME=... NEO4J_PASSWORD=... python helper/export_neo4j_to_parquet.py --only genes pathways

Option C: Watch for a trigger file written by the Neo4j import pipeline
  Your import pipeline writes: touch /data/biomni/.neo4j_updated
  A cron job checks: if file is newer than parquets → re-export → delete trigger file
──────────────────────────────────────────────────────────────────────────────

Usage:
    python helper/export_neo4j_to_parquet.py
    BIOMNI_DATA_LAKE=/data/biomni python helper/export_neo4j_to_parquet.py
    python helper/export_neo4j_to_parquet.py --only genes pathways
    python helper/export_neo4j_to_parquet.py --force
"""

import os
import sys
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_BASE    = os.environ.get("BIOMNI_DATA_LAKE", "./data/biomni")
DATA_DIR = os.path.join(_BASE, "neo4j")   # Neo4j exports go in the neo4j/ subfolder


def _connect():
    from neo4j import GraphDatabase
    uri  = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    pwd  = os.environ.get("NEO4J_PASSWORD")
    if not uri:
        log.error("NEO4J_URI not set. Add it to your .env")
        sys.exit(1)
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    driver.verify_connectivity()
    log.info(f"Connected to Neo4j at {uri}")
    return driver


def _run_query(driver, cypher: str, params: dict = None) -> list:
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(r) for r in result]


def _save(df, name: str):
    import pandas as pd
    path = os.path.join(DATA_DIR, name)
    df.to_parquet(path, index=False, compression="snappy")
    size_mb = os.path.getsize(path) / 1024 / 1024
    log.info(f"  Saved {name}  ({size_mb:.1f} MB,  {len(df):,} rows)")


def _skip(name: str, force: bool) -> bool:
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path) and not force:
        size_mb = os.path.getsize(path) / 1024 / 1024
        log.info(f"  SKIP  {name} already exists ({size_mb:.1f} MB) — pass --force to re-export")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────

def export_genes(driver, force: bool):
    name = "neo4j_genes.parquet"
    if _skip(name, force): return
    log.info("Exporting genes...")
    import pandas as pd
    rows = _run_query(driver, """
        MATCH (g:Gene)
        RETURN g.id AS gene_id,
               g.gene_name AS gene_name,
               g.gene_type AS gene_type,
               g.chr AS chromosome,
               g.start AS start,
               g.end AS end,
               g.strand AS strand
        LIMIT 100000
    """)
    _save(pd.DataFrame(rows), name)


def export_variants(driver, force: bool):
    name = "neo4j_variants.parquet"
    if _skip(name, force): return
    log.info("Exporting variants/SNPs (top 5M by CADD score)...")
    import pandas as pd
    rows = _run_query(driver, """
        MATCH (v:Variant)
        RETURN v.id AS variant_id,
               v.rsid AS rsid,
               v.chr AS chromosome,
               v.pos AS position,
               v.ref AS ref,
               v.alt AS alt,
               v.cadd_score AS cadd_score
        ORDER BY v.cadd_score DESC
        LIMIT 5000000
    """)
    _save(pd.DataFrame(rows), name)


def export_pathways(driver, force: bool):
    name = "neo4j_pathways.parquet"
    if _skip(name, force): return
    log.info("Exporting pathway → gene memberships...")
    import pandas as pd
    rows = _run_query(driver, """
        MATCH (p:Pathway)-[:HAS_GENE|INCLUDES]->(g:Gene)
        RETURN p.id AS pathway_id,
               p.name AS pathway_name,
               p.source AS pathway_source,
               g.gene_name AS gene_name,
               g.id AS gene_id
    """)
    _save(pd.DataFrame(rows), name)


def export_eqtls(driver, force: bool):
    name = "neo4j_eqtls.parquet"
    if _skip(name, force): return
    log.info("Exporting eQTL associations (this may take a while — 63M rows)...")
    import pandas as pd

    # Export in batches to avoid memory issues
    batch_size = 500_000
    all_rows = []
    skip = 0
    while True:
        rows = _run_query(driver, """
            MATCH (v:Variant)-[e:eqtl_association]->(g:Gene)
            RETURN v.rsid AS rsid,
                   g.gene_name AS gene_name,
                   e.tissue AS tissue,
                   e.slope AS slope,
                   e.pval AS pval
            SKIP $skip LIMIT $limit
        """, {"skip": skip, "limit": batch_size})
        if not rows:
            break
        all_rows.extend(rows)
        skip += batch_size
        log.info(f"  {len(all_rows):,} eQTLs exported so far...")
        if len(rows) < batch_size:
            break

    _save(pd.DataFrame(all_rows), name)


def export_coexpression(driver, force: bool):
    name = "neo4j_coexpression.parquet"
    if _skip(name, force): return
    log.info("Exporting coexpression links (top 10M by score)...")
    import pandas as pd
    rows = _run_query(driver, """
        MATCH (g1:Gene)-[c:coexpressed_with]->(g2:Gene)
        RETURN g1.gene_name AS gene_a,
               g2.gene_name AS gene_b,
               c.score AS score,
               c.tissue AS tissue
        ORDER BY c.score DESC
        LIMIT 10000000
    """)
    _save(pd.DataFrame(rows), name)


def export_tissues(driver, force: bool):
    name = "neo4j_tissues.parquet"
    if _skip(name, force): return
    log.info("Exporting tissue expression data...")
    import pandas as pd
    rows = _run_query(driver, """
        MATCH (g:Gene)-[e:expressed_in]->(t)
        RETURN g.gene_name AS gene_name,
               t.name AS tissue,
               e.median_tpm AS median_tpm,
               e.tissue_site AS tissue_site
        LIMIT 5000000
    """)
    _save(pd.DataFrame(rows), name)


# ─────────────────────────────────────────────────────────────────────────────

ALL_EXPORTS = {
    "genes":        (export_genes,       "Gene nodes — 78K genes with coordinates"),
    "variants":     (export_variants,    "Variant/SNP nodes — top 5M by CADD score"),
    "pathways":     (export_pathways,    "Pathway → gene memberships (REACTOME, GO)"),
    "eqtls":        (export_eqtls,       "eQTL associations — 63M variant-gene-tissue links"),
    "coexpression": (export_coexpression,"Coexpression links — top 10M pairs"),
    "tissues":      (export_tissues,     "Tissue expression — gene TPM per tissue"),
}


def main():
    parser = argparse.ArgumentParser(description="Export Neo4j Atomspace entities to parquet files.")
    parser.add_argument("--skip",  nargs="+", choices=list(ALL_EXPORTS.keys()), default=[])
    parser.add_argument("--only",  nargs="+", choices=list(ALL_EXPORTS.keys()), default=[])
    parser.add_argument("--force", action="store_true", help="Re-export even if file exists")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    log.info(f"Output directory: {os.path.abspath(DATA_DIR)}")

    driver = _connect()
    to_run = args.only if args.only else list(ALL_EXPORTS.keys())
    to_run = [k for k in to_run if k not in args.skip]

    results = {}
    for key in to_run:
        fn, desc = ALL_EXPORTS[key]
        log.info(f"\n{'─'*60}\n[{key.upper()}] {desc}\n{'─'*60}")
        t0 = time.time()
        try:
            fn(driver, args.force)
            results[key] = f"OK ({time.time()-t0:.0f}s)"
        except Exception as e:
            log.error(f"Export failed: {e}", exc_info=True)
            results[key] = f"FAILED: {e}"

    driver.close()

    log.info(f"\n{'═'*60}\nEXPORT SUMMARY\n{'═'*60}")
    for key, status in results.items():
        icon = "✓" if status.startswith("OK") else "✗"
        log.info(f"  {icon}  {key:<15} {status}")

    files = [f for f in os.listdir(DATA_DIR) if f.startswith("neo4j_")]
    total_mb = sum(os.path.getsize(os.path.join(DATA_DIR, f)) / 1024 / 1024 for f in files)
    log.info(f"\n  Neo4j exports: {len(files)} files, {total_mb:.0f} MB total")
    log.info(f"  Set CODING_DATA_MODE=neo4j in .env to use these in the code executor")


if __name__ == "__main__":
    main()
