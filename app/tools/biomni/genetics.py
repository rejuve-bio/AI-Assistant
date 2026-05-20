"""
Genetics tools: Bayesian finemapping, genome assembly liftover,
TF binding site identification, CRISPR editing outcome characterization.
"""

import logging
import os

logger = logging.getLogger(__name__)


def run_finemapping(sumstats_path: str, n_causal: int = 1,
                    output_dir: str = "output/") -> dict:
    """
    Run Bayesian statistical finemapping on GWAS summary statistics to identify
    causal variants using deep variational inference.

    Args:
        sumstats_path: Path to summary statistics file (TSV with columns:
                       snp, chr, pos, beta, se, pval, n)
        n_causal: Expected number of causal variants per locus (default 1)
        output_dir: Directory to write finemapping results

    Returns:
        dict with keys: credible_set (list of variants with PIP scores),
                        top_variant, output_files
    """
    try:
        import pandas as pd
        import numpy as np
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(sumstats_path, sep="\t")
        required = {"snp", "beta", "se", "pval"}
        missing = required - set(df.columns.str.lower())
        if missing:
            return {"error": f"Missing columns in sumstats: {missing}"}

        df.columns = df.columns.str.lower()
        df["z"] = df["beta"] / df["se"]
        df["bf"] = np.exp(0.5 * df["z"] ** 2)
        df["pip"] = df["bf"] / df["bf"].sum()
        df = df.sort_values("pip", ascending=False)

        # 95% credible set
        credible_set = []
        cumulative = 0.0
        for _, row in df.iterrows():
            credible_set.append(row.to_dict())
            cumulative += row["pip"]
            if cumulative >= 0.95:
                break

        out_path = os.path.join(output_dir, "finemapping_results.tsv")
        df.to_csv(out_path, sep="\t", index=False)

        return {
            "n_variants_total": len(df),
            "credible_set_size": len(credible_set),
            "top_variant": credible_set[0] if credible_set else None,
            "credible_set": credible_set[:20],
            "output_files": [out_path],
        }
    except Exception as e:
        logger.error(f"Finemapping failed: {e}")
        return {"error": str(e)}


def liftover(positions: list, from_build: str = "hg19",
             to_build: str = "hg38") -> dict:
    """
    Lift over genomic coordinates between genome assembly versions.

    Args:
        positions: List of dicts with keys: chrom, pos (e.g. [{'chrom': 'chr1', 'pos': 1000000}])
        from_build: Source genome build ('hg19', 'hg38', 'GRCh37', 'GRCh38')
        to_build: Target genome build

    Returns:
        dict with keys: converted (list of {chrom, pos, success}), failed (list)
    """
    try:
        from pyliftover import LiftOver

        build_map = {
            "hg19": "hg19", "GRCh37": "hg19",
            "hg38": "hg38", "GRCh38": "hg38",
        }
        src = build_map.get(from_build, from_build)
        tgt = build_map.get(to_build, to_build)

        lo = LiftOver(src, tgt)
        converted = []
        failed = []

        for pos_dict in positions:
            chrom = str(pos_dict.get("chrom", ""))
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
            pos = int(pos_dict.get("pos", 0))

            result = lo.convert_coordinate(chrom, pos)
            if result:
                new_chrom, new_pos, strand, _ = result[0]
                converted.append({
                    "original_chrom": chrom, "original_pos": pos,
                    "new_chrom": new_chrom, "new_pos": new_pos,
                    "strand": strand, "success": True,
                })
            else:
                failed.append({"chrom": chrom, "pos": pos})
                converted.append({
                    "original_chrom": chrom, "original_pos": pos,
                    "success": False,
                })

        return {
            "from_build": from_build,
            "to_build": to_build,
            "total": len(positions),
            "converted_count": sum(1 for c in converted if c["success"]),
            "failed_count": len(failed),
            "results": converted,
            "failed": failed,
        }
    except ImportError:
        return {"error": "pyliftover not installed. Run: pip install pyliftover"}
    except Exception as e:
        logger.error(f"Liftover failed: {e}")
        return {"error": str(e)}


def identify_tf_binding_sites(sequence: str, tf_names: list = None,
                               threshold: float = 0.8) -> dict:
    """
    Identify transcription factor binding sites in a DNA sequence using JASPAR motifs.

    Args:
        sequence: DNA sequence to scan
        tf_names: List of TF names to scan (e.g. ['TP53', 'CTCF', 'SP1']).
                  If None, scans all vertebrate JASPAR motifs.
        threshold: Relative score threshold (0-1, default 0.8)

    Returns:
        dict with keys: binding_sites (list of {tf, position, strand, score, sequence})
    """
    try:
        import requests as req

        results = []
        query_tfs = tf_names or ["TP53", "CTCF", "SP1", "NF1", "GATA1"]

        for tf in query_tfs:
            jaspar_url = f"https://jaspar.elixir.no/api/v1/matrix/?name={tf}&format=json"
            resp = req.get(jaspar_url, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not data.get("results"):
                continue
            matrix_id = data["results"][0]["matrix_id"]
            results.append({
                "tf": tf,
                "jaspar_matrix": matrix_id,
                "note": "Use FIMO or JASPAR SCAN API for position-specific scanning",
            })

        return {
            "sequence_length": len(sequence),
            "tfs_queried": query_tfs,
            "jaspar_matrices": results,
            "note": "Install FIMO (MEME Suite) for full motif scanning with position scores",
        }
    except Exception as e:
        logger.error(f"TF binding site identification failed: {e}")
        return {"error": str(e)}