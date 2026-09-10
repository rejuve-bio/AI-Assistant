"""
Tests for literature contradiction & consensus reasoning.

Covers:
- Stance classification prompt against known-established and
  known-controversial claims.
- Consensus label computation from stances.
- Contradiction warning generation.

All tests mock the LLM so no live API calls are made.
"""

import pytest
from unittest.mock import MagicMock

from app.rag.literature_consensus import LiteratureConsensusAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Return a mock LLM whose .generate() we can control per test."""
    return MagicMock()


@pytest.fixture
def analyzer(mock_llm):
    return LiteratureConsensusAnalyzer(mock_llm)


# Reusable fake papers
FAKE_PAPERS = [
    {"pmid": "111", "title": "Paper A", "abstract": "Supports the claim.", "url": "https://pubmed.ncbi.nlm.nih.gov/111/"},
    {"pmid": "222", "title": "Paper B", "abstract": "Also supports.", "url": "https://pubmed.ncbi.nlm.nih.gov/222/"},
    {"pmid": "333", "title": "Paper C", "abstract": "Contradicts the claim.", "url": "https://pubmed.ncbi.nlm.nih.gov/333/"},
    {"pmid": "444", "title": "Paper D", "abstract": "Inconclusive results.", "url": "https://pubmed.ncbi.nlm.nih.gov/444/"},
]


# ---------------------------------------------------------------------------
# Milestone 1 — Stance classification against known claims
# ---------------------------------------------------------------------------

class TestStanceClassification:
    """Verify that the analyzer correctly parses LLM stance output."""

    def test_known_established_claim_all_support(self, analyzer, mock_llm):
        """Known-established claim: all papers support → STRONGLY_SUPPORTED."""
        mock_llm.generate.return_value = {
            "paper_stances": [
                {"pmid": "111", "stance": "SUPPORT", "rationale": "Confirms BRCA1 link."},
                {"pmid": "222", "stance": "SUPPORT", "rationale": "Large cohort study agrees."},
                {"pmid": "333", "stance": "SUPPORT", "rationale": "Meta-analysis supports."},
            ],
            "consensus": {
                "label": "STRONGLY_SUPPORTED",
                "support_count": 3,
                "oppose_count": 0,
                "inconclusive_count": 0,
                "summary": "All papers confirm the association.",
            },
        }

        result = analyzer.analyze_consensus(
            "BRCA1 mutations increase the risk of breast cancer.",
            FAKE_PAPERS[:3],
        )

        assert result["consensus_label"] == "STRONGLY_SUPPORTED"
        assert result["support_count"] == 3
        assert result["oppose_count"] == 0
        assert result["has_contradiction"] is False
        assert result["warning_text"] is None

    def test_known_controversial_claim_contested(self, analyzer, mock_llm):
        """Known-controversial claim: papers disagree → CONTESTED + contradiction."""
        mock_llm.generate.return_value = {
            "paper_stances": [
                {"pmid": "111", "stance": "SUPPORT", "rationale": "Supports telomere theory."},
                {"pmid": "222", "stance": "SUPPORT", "rationale": "Agrees with claim."},
                {"pmid": "333", "stance": "OPPOSE", "rationale": "No causal relationship found."},
                {"pmid": "444", "stance": "OPPOSE", "rationale": "Cross-species analysis contradicts."},
            ],
            "consensus": {
                "label": "CONTESTED",
                "support_count": 2,
                "oppose_count": 2,
                "inconclusive_count": 0,
                "summary": "Evidence is divided.",
            },
        }

        result = analyzer.analyze_consensus(
            "Telomere length directly determines organismal lifespan.",
            FAKE_PAPERS,
        )

        assert result["consensus_label"] == "CONTESTED"
        assert result["has_contradiction"] is True
        assert result["warning_text"] is not None
        assert "Contradiction detected" in result["warning_text"]

    def test_inconclusive_papers(self, analyzer, mock_llm):
        """All papers are inconclusive → INCONCLUSIVE, no contradiction."""
        mock_llm.generate.return_value = {
            "paper_stances": [
                {"pmid": "111", "stance": "INCONCLUSIVE", "rationale": "Mixed results."},
                {"pmid": "222", "stance": "INCONCLUSIVE", "rationale": "Insufficient data."},
            ],
            "consensus": {
                "label": "INCONCLUSIVE",
                "support_count": 0,
                "oppose_count": 0,
                "inconclusive_count": 2,
                "summary": "Evidence is insufficient.",
            },
        }

        result = analyzer.analyze_consensus(
            "Metformin extends lifespan in healthy humans.",
            FAKE_PAPERS[:2],
        )

        assert result["consensus_label"] == "INCONCLUSIVE"
        assert result["has_contradiction"] is False
        assert result["warning_text"] is None


# ---------------------------------------------------------------------------
# Milestone 2 — Consensus label computation
# ---------------------------------------------------------------------------

class TestConsensusLabels:
    """Verify consensus label determination and count derivation."""

    def test_counts_derived_from_stances_when_llm_omits_them(self, analyzer, mock_llm):
        """If the LLM returns all-zero counts, derive them from the stances list."""
        mock_llm.generate.return_value = {
            "paper_stances": [
                {"pmid": "111", "stance": "SUPPORT", "rationale": "Supports."},
                {"pmid": "222", "stance": "OPPOSE", "rationale": "Opposes."},
                {"pmid": "333", "stance": "INCONCLUSIVE", "rationale": "Unclear."},
            ],
            "consensus": {
                "label": "CONTESTED",
                "support_count": 0,  # LLM forgot to fill these
                "oppose_count": 0,
                "inconclusive_count": 0,
                "summary": "Mixed.",
            },
        }

        result = analyzer.analyze_consensus("Some claim.", FAKE_PAPERS[:3])

        assert result["support_count"] == 1
        assert result["oppose_count"] == 1
        assert result["inconclusive_count"] == 1
        assert result["has_contradiction"] is True

    def test_supported_label(self, analyzer, mock_llm):
        """Majority support with 1 oppose → SUPPORTED."""
        mock_llm.generate.return_value = {
            "paper_stances": [
                {"pmid": "111", "stance": "SUPPORT", "rationale": ""},
                {"pmid": "222", "stance": "SUPPORT", "rationale": ""},
                {"pmid": "333", "stance": "OPPOSE", "rationale": ""},
            ],
            "consensus": {
                "label": "SUPPORTED",
                "support_count": 2,
                "oppose_count": 1,
                "inconclusive_count": 0,
                "summary": "Mostly supported.",
            },
        }

        result = analyzer.analyze_consensus("Claim.", FAKE_PAPERS[:3])

        assert result["consensus_label"] == "SUPPORTED"
        # Still has contradiction since oppose_count >= 1
        assert result["has_contradiction"] is True


# ---------------------------------------------------------------------------
# Milestone 3 — Contradiction warning text
# ---------------------------------------------------------------------------

class TestContradictionWarning:
    """Verify that contradiction warnings are properly generated."""

    def test_warning_includes_paper_titles(self, analyzer, mock_llm):
        """Warning text should reference actual paper titles, not just PMIDs."""
        mock_llm.generate.return_value = {
            "paper_stances": [
                {"pmid": "111", "stance": "SUPPORT", "rationale": "Supports."},
                {"pmid": "333", "stance": "OPPOSE", "rationale": "Contradicts."},
            ],
            "consensus": {
                "label": "CONTESTED",
                "support_count": 1,
                "oppose_count": 1,
                "inconclusive_count": 0,
                "summary": "Divided.",
            },
        }

        result = analyzer.analyze_consensus("Claim X.", FAKE_PAPERS)

        assert result["has_contradiction"] is True
        assert "Paper A" in result["warning_text"]
        assert "Paper C" in result["warning_text"]
        assert "supporting" in result["warning_text"].lower()
        assert "opposing" in result["warning_text"].lower()

    def test_no_warning_when_no_contradiction(self, analyzer, mock_llm):
        """No contradiction → no warning text."""
        mock_llm.generate.return_value = {
            "paper_stances": [
                {"pmid": "111", "stance": "SUPPORT", "rationale": ""},
            ],
            "consensus": {
                "label": "SUPPORTED",
                "support_count": 1,
                "oppose_count": 0,
                "inconclusive_count": 0,
                "summary": "Supported.",
            },
        }

        result = analyzer.analyze_consensus("Claim Y.", FAKE_PAPERS[:1])

        assert result["has_contradiction"] is False
        assert result["warning_text"] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Verify graceful handling of empty inputs and LLM failures."""

    def test_empty_papers_returns_inconclusive(self, analyzer, mock_llm):
        result = analyzer.analyze_consensus("Any claim.", [])
        assert result["consensus_label"] == "INCONCLUSIVE"
        assert result["has_contradiction"] is False
        mock_llm.generate.assert_not_called()

    def test_empty_claim_returns_inconclusive(self, analyzer, mock_llm):
        result = analyzer.analyze_consensus("", FAKE_PAPERS)
        assert result["consensus_label"] == "INCONCLUSIVE"
        mock_llm.generate.assert_not_called()

    def test_llm_returns_garbage_string(self, analyzer, mock_llm):
        """If LLM returns unparseable text, fall back gracefully."""
        mock_llm.generate.return_value = "I don't understand the question."
        result = analyzer.analyze_consensus("Claim.", FAKE_PAPERS)
        assert result["consensus_label"] == "INCONCLUSIVE"
        assert result["has_contradiction"] is False

    def test_llm_raises_exception(self, analyzer, mock_llm):
        """If LLM throws, return empty result instead of crashing."""
        mock_llm.generate.side_effect = RuntimeError("API down")
        result = analyzer.analyze_consensus("Claim.", FAKE_PAPERS)
        assert result["consensus_label"] == "INCONCLUSIVE"
        assert result["has_contradiction"] is False
