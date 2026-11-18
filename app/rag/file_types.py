"""File-type allowlist and helpers for raw uploads.

This is deliberately dependency-free so it can be imported from tests
without pulling in the rest of the application.
"""
import os

MAX_FILE_SIZE_MB = 50

# Accepted file extensions, grouped by category.
# This dict is the single source of truth: ALLOWED_FILE_EXTENSIONS and
# the /v1/files/supported-types endpoint are both derived from it.
FILE_TYPE_CATEGORIES = {
    "gwas_summary_stats": [".tsv", ".txt", ".csv", ".gz", ".bgz",
                           ".tsv.gz", ".txt.gz", ".csv.gz"],
    "sequence_data":      [".fastq", ".fq", ".fastq.gz", ".fq.gz"],
    "reference_sequences": [".fasta", ".fa", ".fasta.gz", ".fa.gz"],
    "genomics":           [".vcf", ".vcf.gz", ".bed", ".bam", ".sam"],
    "gene_annotations":   [".gff", ".gff3", ".gtf"],
    "documents":          [".pdf"],
}

ALLOWED_FILE_EXTENSIONS = {ext for exts in FILE_TYPE_CATEGORIES.values() for ext in exts}

# Sort longest-first so .tsv.gz matches before .gz
_SORTED_EXTENSIONS = sorted(ALLOWED_FILE_EXTENSIONS, key=len, reverse=True)


def match_allowed_extension(lower_filename: str):
    """Return the matching allowed extension for a lowered filename, or None."""
    for ext in _SORTED_EXTENSIONS:
        if lower_filename.endswith(ext):
            return ext
    return None


def format_file_description(file_info: dict, file_id: str) -> str:
    """Build a one-line file description from a MongoDB file record.
    Shared by workflow resource summaries and the explain_resource tool."""
    fname = file_info.get("filename", file_id)
    fsize = (file_info.get("metadata") or {}).get("file_size_mb")
    size_str = f" ({fsize} MB)" if fsize is not None else ""
    return f"{fname}{size_str}"

