"""
Molecular biology tools: CRISPR sgRNA design, primer design, plasmid annotation,
restriction digest simulation, sequence alignment.
"""

import logging
import os

logger = logging.getLogger(__name__)


def design_sgrna(target_sequence: str, pam: str = "NGG",
                 guide_length: int = 20, top_k: int = 10) -> dict:
    """
    Design CRISPR sgRNA guides for a target DNA sequence.

    Args:
        target_sequence: DNA sequence to target (must include PAM context)
        pam: PAM sequence (default 'NGG' for SpCas9, 'NNGRRT' for SaCas9)
        guide_length: Length of guide RNA (default 20)
        top_k: Number of top guides to return

    Returns:
        dict with keys: guides (list of {sequence, position, gc_content, off_target_score})
    """
    try:
        from Bio.Seq import Seq
        import re

        sequence = target_sequence.upper().replace(" ", "")
        pam_pattern = pam.replace("N", "[ACGT]").replace("R", "[AG]").replace("Y", "[CT]")

        guides = []
        for i in range(len(sequence) - guide_length - len(pam) + 1):
            guide = sequence[i:i + guide_length]
            pam_seq = sequence[i + guide_length:i + guide_length + len(pam)]
            if re.match(pam_pattern, pam_seq):
                gc = (guide.count("G") + guide.count("C")) / len(guide)
                # Simple off-target heuristic: penalise homopolymers and low GC
                off_target_score = round(1.0 - abs(gc - 0.5) * 0.5, 2)
                guides.append({
                    "sequence": guide,
                    "pam": pam_seq,
                    "position": i,
                    "gc_content": round(gc, 2),
                    "off_target_score": off_target_score,
                    "strand": "forward",
                })

        # Also scan reverse complement
        rc = str(Seq(sequence).reverse_complement())
        for i in range(len(rc) - guide_length - len(pam) + 1):
            guide = rc[i:i + guide_length]
            pam_seq = rc[i + guide_length:i + guide_length + len(pam)]
            if re.match(pam_pattern, pam_seq):
                gc = (guide.count("G") + guide.count("C")) / len(guide)
                off_target_score = round(1.0 - abs(gc - 0.5) * 0.5, 2)
                guides.append({
                    "sequence": guide,
                    "pam": pam_seq,
                    "position": len(sequence) - i - guide_length,
                    "gc_content": round(gc, 2),
                    "off_target_score": off_target_score,
                    "strand": "reverse",
                })

        guides.sort(key=lambda x: x["off_target_score"], reverse=True)
        return {
            "pam": pam,
            "guide_length": guide_length,
            "total_guides_found": len(guides),
            "top_guides": guides[:top_k],
        }
    except ImportError:
        return {"error": "biopython not installed. Run: pip install biopython"}
    except Exception as e:
        logger.error(f"sgRNA design failed: {e}")
        return {"error": str(e)}


def design_primers(template_sequence: str, product_size_range: tuple = (100, 500),
                   tm_range: tuple = (55, 65), output_dir: str = "output/") -> dict:
    """
    Design PCR primers for a template sequence using Primer3.

    Args:
        template_sequence: DNA sequence to design primers for
        product_size_range: Tuple (min, max) for PCR product size in bp
        tm_range: Tuple (min, max) for primer melting temperature in °C
        output_dir: Directory to write primer results

    Returns:
        dict with keys: primer_pairs (list of {left, right, product_size, tm_left, tm_right})
    """
    try:
        import primer3
        os.makedirs(output_dir, exist_ok=True)

        result = primer3.design_primers(
            {
                "SEQUENCE_ID": "template",
                "SEQUENCE_TEMPLATE": template_sequence.upper(),
            },
            {
                "PRIMER_OPT_SIZE": 20,
                "PRIMER_MIN_SIZE": 18,
                "PRIMER_MAX_SIZE": 25,
                "PRIMER_OPT_TM": 60.0,
                "PRIMER_MIN_TM": tm_range[0],
                "PRIMER_MAX_TM": tm_range[1],
                "PRIMER_MIN_GC": 40.0,
                "PRIMER_MAX_GC": 65.0,
                "PRIMER_PRODUCT_SIZE_RANGE": [list(product_size_range)],
                "PRIMER_NUM_RETURN": 5,
            },
        )

        pairs = []
        for i in range(result.get("PRIMER_PAIR_NUM_RETURNED", 0)):
            pairs.append({
                "pair_id": i,
                "left_sequence": result.get(f"PRIMER_LEFT_{i}_SEQUENCE", ""),
                "right_sequence": result.get(f"PRIMER_RIGHT_{i}_SEQUENCE", ""),
                "product_size": result.get(f"PRIMER_PAIR_{i}_PRODUCT_SIZE", 0),
                "tm_left": round(result.get(f"PRIMER_LEFT_{i}_TM", 0), 1),
                "tm_right": round(result.get(f"PRIMER_RIGHT_{i}_TM", 0), 1),
                "penalty": round(result.get(f"PRIMER_PAIR_{i}_PENALTY", 0), 3),
            })

        return {"template_length": len(template_sequence), "primer_pairs": pairs}
    except ImportError:
        return {"error": "primer3-py not installed. Run: pip install primer3-py"}
    except Exception as e:
        logger.error(f"Primer design failed: {e}")
        return {"error": str(e)}


def simulate_restriction_digest(sequence: str, enzymes: list = None) -> dict:
    """
    Simulate restriction enzyme digest of a DNA sequence.

    Args:
        sequence: DNA sequence to digest
        enzymes: List of restriction enzyme names (e.g. ['EcoRI', 'BamHI']).
                 Defaults to common cloning enzymes.

    Returns:
        dict with keys: fragments (list of {start, end, size}), cut_sites per enzyme
    """
    try:
        from Bio.Restriction import RestrictionBatch, Analysis
        from Bio.Seq import Seq

        if enzymes is None:
            enzymes = ["EcoRI", "BamHI", "HindIII", "NcoI", "XhoI", "NdeI"]

        rb = RestrictionBatch(enzymes)
        seq = Seq(sequence.upper())
        ana = Analysis(rb, seq)
        results = ana.full()

        cut_sites = {str(enz): sites for enz, sites in results.items() if sites}
        all_cuts = sorted(set(pos for sites in cut_sites.values() for pos in sites))
        all_cuts = [0] + all_cuts + [len(sequence)]

        fragments = [
            {"start": all_cuts[i], "end": all_cuts[i + 1], "size": all_cuts[i + 1] - all_cuts[i]}
            for i in range(len(all_cuts) - 1)
        ]

        return {
            "sequence_length": len(sequence),
            "cut_sites": cut_sites,
            "n_fragments": len(fragments),
            "fragments": sorted(fragments, key=lambda x: x["size"], reverse=True),
        }
    except ImportError:
        return {"error": "biopython not installed. Run: pip install biopython"}
    except Exception as e:
        logger.error(f"Restriction digest simulation failed: {e}")
        return {"error": str(e)}