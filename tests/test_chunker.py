"""
Unit tests for Track 1: Heading-aware Markdown chunker.

Tests `ContentProcessor.chunk_by_headings()` — no LLM, no Qdrant, no network.

ContentProcessor is imported inside each fixture so it resolves AFTER
conftest.py has stubbed sentence_transformers and friends into sys.modules.
"""
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cp():
    from app.rag.utils import content_processor
    # Lower the minimum size for unit tests so small strings aren't skipped
    content_processor._HEADING_CHUNK_MIN = 10
    return content_processor.ContentProcessor()


# ---------------------------------------------------------------------------
# chunk_by_headings — happy path
# ---------------------------------------------------------------------------

class TestChunkByHeadingsHappyPath:

    def test_single_h1_returns_one_chunk(self, cp):
        md = "# Introduction\nThis is the introduction body."
        chunks = cp.chunk_by_headings(md)
        assert len(chunks) == 1
        assert chunks[0]["section"] == "Introduction"
        assert chunks[0]["breadcrumb"] == "Introduction"
        assert "introduction body" in chunks[0]["text"]

    def test_two_h1_sections_return_two_chunks(self, cp):
        md = "# Background\nSome background.\n\n# Methods\nSome methods."
        chunks = cp.chunk_by_headings(md)
        assert len(chunks) == 2
        sections = [c["section"] for c in chunks]
        assert "Background" in sections
        assert "Methods" in sections

    def test_nested_headings_build_correct_breadcrumb(self, cp):
        md = (
            "# Introduction\n"
            "Intro text.\n\n"
            "## Background\n"
            "Background text.\n\n"
            "### Sub-background\n"
            "Sub text here and more text to exceed minimum length.\n"
        )
        chunks = cp.chunk_by_headings(md)
        crumbs = [c["breadcrumb"] for c in chunks]
        assert any("Introduction > Background" in b for b in crumbs)
        assert any("Introduction > Background > Sub-background" in b for b in crumbs)

    def test_heading_level_reset_clears_deeper_ancestors(self, cp):
        """Going back to H1 after H2 must drop H2 from the breadcrumb."""
        md = (
            "# First\nFirst body with enough text to pass minimum.\n\n"
            "## Sub of First\nSub body with enough text to pass minimum.\n\n"
            "# Second\nSecond body with enough text to pass minimum.\n"
        )
        chunks = cp.chunk_by_headings(md)
        second_chunk = next(c for c in chunks if c["section"] == "Second")
        assert second_chunk["breadcrumb"] == "Second"
        assert "First" not in second_chunk["breadcrumb"]

    def test_chunk_text_does_not_contain_heading_marker(self, cp):
        md = "## Methods\nWe used CRISPR to edit the genome and here is more detail."
        chunks = cp.chunk_by_headings(md)
        assert len(chunks) == 1
        assert "##" not in chunks[0]["text"]

    def test_section_key_equals_innermost_heading(self, cp):
        md = (
            "# H1\nBody one with enough text.\n\n"
            "## H2\nNested body with enough text to pass minimum check."
        )
        chunks = cp.chunk_by_headings(md)
        sections = {c["section"] for c in chunks}
        assert sections == {"H1", "H2"}


# ---------------------------------------------------------------------------
# chunk_by_headings — edge cases
# ---------------------------------------------------------------------------

class TestChunkByHeadingsEdgeCases:

    def test_empty_string_returns_empty_list(self, cp):
        assert cp.chunk_by_headings("") == []

    def test_whitespace_only_returns_empty_list(self, cp):
        assert cp.chunk_by_headings("   \n\n  ") == []

    def test_no_headings_falls_back_to_recursive_splitter(self, cp):
        """Plain text with no headings should produce at least one chunk via fallback."""
        text = "Aging is a complex biological process. " * 30
        chunks = cp.chunk_by_headings(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert "text" in c
            assert c["breadcrumb"] == ""
            assert c["section"] == ""

    def test_toc_block_is_stripped(self, cp):
        md = (
            "Table of Contents\n"
            "1. Introduction\n2. Methods\n\n"
            "# Introduction\nReal content here with enough words to pass minimum.\n"
        )
        chunks = cp.chunk_by_headings(md)
        full_text = " ".join(c["text"] for c in chunks)
        assert "Table of Contents" not in full_text
        assert "Real content here" in full_text

    def test_tiny_section_below_minimum_is_skipped(self, cp):
        """A body with fewer chars than _HEADING_CHUNK_MIN should be dropped."""
        md = "# Tiny\nHi.\n\n# Normal\n" + "Normal content. " * 10
        chunks = cp.chunk_by_headings(md)
        sections = [c["section"] for c in chunks]
        assert "Tiny" not in sections
        assert "Normal" in sections

    def test_oversized_section_is_split_into_sub_chunks(self, cp):
        """Section body > _HEADING_CHUNK_MAX should be split further."""
        md = "# Long Section\n" + ("aging longevity mTOR rapamycin " * 60)
        chunks = cp.chunk_by_headings(md)
        long_chunks = [c for c in chunks if c["section"] == "Long Section"]
        assert len(long_chunks) >= 2
        for c in long_chunks:
            assert c["breadcrumb"] == "Long Section"

    def test_all_chunks_have_required_keys(self, cp):
        md = "# A\nContent A with enough text.\n\n# B\nContent B with enough text."
        chunks = cp.chunk_by_headings(md)
        for chunk in chunks:
            assert "text" in chunk
            assert "breadcrumb" in chunk
            assert "section" in chunk

    def test_chunk_text_is_never_empty(self, cp):
        md = "# Section\n" + "Word " * 20
        chunks = cp.chunk_by_headings(md)
        for c in chunks:
            assert c["text"].strip(), "Found chunk with empty text"


