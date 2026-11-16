"""Tests for file extension validation logic.

Imports app/rag/file_types.py directly by path to avoid triggering
app/__init__.py and its heavy dependency chain.
"""
import importlib.util
import os
import sys

import pytest

# Load file_types.py by file path, bypassing app/__init__.py
_FILE_TYPES_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "app", "rag", "file_types.py"
)
_spec = importlib.util.spec_from_file_location(
    "file_types", os.path.abspath(_FILE_TYPES_PATH)
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ALLOWED_FILE_EXTENSIONS = _mod.ALLOWED_FILE_EXTENSIONS
FILE_TYPE_CATEGORIES = _mod.FILE_TYPE_CATEGORIES
match_allowed_extension = _mod.match_allowed_extension


class TestMatchAllowedExtension:
    """match_allowed_extension should pick the longest matching suffix."""

    @pytest.mark.parametrize("filename,expected", [
        # Simple extensions
        ("data.tsv", ".tsv"),
        ("data.csv", ".csv"),
        ("data.txt", ".txt"),
        ("data.pdf", ".pdf"),
        ("data.vcf", ".vcf"),
        ("data.bed", ".bed"),
        ("data.bam", ".bam"),
        ("data.sam", ".sam"),
        ("data.gff", ".gff"),
        ("data.gff3", ".gff3"),
        ("data.gtf", ".gtf"),
        ("data.bgz", ".bgz"),
        # Compound extensions -- the core bug this test guards against
        ("sumstats.tsv.gz", ".tsv.gz"),
        ("sumstats.txt.gz", ".txt.gz"),
        ("sumstats.csv.gz", ".csv.gz"),
        ("reads.fastq.gz", ".fastq.gz"),
        ("reads.fq.gz", ".fq.gz"),
        ("ref.fasta.gz", ".fasta.gz"),
        ("ref.fa.gz", ".fa.gz"),
        ("variants.vcf.gz", ".vcf.gz"),
        # Names with extra dots (e.g. version numbers)
        ("data.v2.tsv.gz", ".tsv.gz"),
        ("project.final.csv.gz", ".csv.gz"),
        ("sample_1.2.3.vcf.gz", ".vcf.gz"),
        # Bare .gz (plain compressed, no inner extension we recognise)
        ("archive.gz", ".gz"),
        # Case insensitivity (filenames are lowered before calling)
        ("DATA.TSV", ".tsv"),
    ])
    def test_accepted(self, filename, expected):
        result = match_allowed_extension(filename.lower())
        assert result == expected, f"{filename} should match {expected}, got {result}"

    @pytest.mark.parametrize("filename", [
        "image.png",
        "document.docx",
        "data.xlsx",
        "script.py",
        "archive.zip",
        "genome.bai",      # BAM index, not BAM
        "noext",
    ])
    def test_rejected(self, filename):
        result = match_allowed_extension(filename.lower())
        assert result is None, f"{filename} should be rejected but matched {result}"


class TestFileCategoryConsistency:
    """FILE_TYPE_CATEGORIES should be consistent with ALLOWED_FILE_EXTENSIONS."""

    def test_all_category_extensions_in_allowlist(self):
        for cat, exts in FILE_TYPE_CATEGORIES.items():
            for ext in exts:
                assert ext in ALLOWED_FILE_EXTENSIONS, (
                    f"{ext} in category '{cat}' is not in ALLOWED_FILE_EXTENSIONS"
                )

    def test_all_allowed_in_some_category(self):
        categorised = {ext for exts in FILE_TYPE_CATEGORIES.values() for ext in exts}
        assert categorised == ALLOWED_FILE_EXTENSIONS, (
            f"Mismatch: in allowlist but not categorised: "
            f"{ALLOWED_FILE_EXTENSIONS - categorised}"
        )

    def test_compound_extensions_present(self):
        """The main GWAS compressed formats must be explicitly listed."""
        required = {".tsv.gz", ".txt.gz", ".csv.gz", ".vcf.gz",
                    ".fastq.gz", ".fq.gz", ".fasta.gz", ".fa.gz"}
        missing = required - ALLOWED_FILE_EXTENSIONS
        assert not missing, f"Missing compound extensions: {missing}"
