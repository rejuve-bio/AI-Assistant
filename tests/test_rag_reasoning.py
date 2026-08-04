"""
Unit tests for RAG reasoning enhancements:
  - _reflect_and_revise: GOOD, REVISE, and malformed verdict payloads
  - _decompose_query: heuristic gate and LLM decomposition
  - _confidence_label: qualitative score mapping
"""
import pytest
import json
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Create a mock LLM that we can control per-test."""
    llm = MagicMock()
    llm.generate = MagicMock(return_value="")
    return llm


@pytest.fixture
def rag_instance(mock_llm):
    """Instantiate RAG with a mocked LLM and Qdrant client."""
    with patch("app.rag.rag.ContentProcessor"), \
         patch("app.rag.rag.ContentAnalyzer"):
        from app.rag.rag import RAG
        rag = RAG(llm=mock_llm, qdrant_client=MagicMock())
    return rag


# ===========================================================================
# Tests for _reflect_and_revise
# ===========================================================================

class TestReflectAndRevise:
    """Tests for the RAG reflection critic step."""

    SAMPLE_QUERY = "What is Rejuve Bio?"
    SAMPLE_CHUNKS = [{"text": "Rejuve Bio is a longevity research company."}]
    SAMPLE_ANSWER = "Rejuve Bio is a longevity research company."

    def test_good_verdict_returns_original_answer(self, rag_instance, mock_llm):
        """GOOD verdict → original answer returned unchanged with high confidence."""
        mock_llm.generate.return_value = json.dumps({
            "verdict": "GOOD",
            "confidence": 0.92,
        })

        answer, confidence = rag_instance._reflect_and_revise(
            self.SAMPLE_QUERY, self.SAMPLE_CHUNKS, self.SAMPLE_ANSWER
        )

        assert answer == self.SAMPLE_ANSWER
        assert confidence == pytest.approx(0.92, abs=0.01)

    def test_revise_verdict_triggers_revision(self, rag_instance, mock_llm):
        """REVISE verdict → LLM called again with feedback, revised answer returned."""
        mock_llm.generate.side_effect = [
            # First call: reflection verdict
            json.dumps({
                "verdict": "REVISE",
                "confidence": 0.4,
                "feedback": "Answer misses the company's focus on AI-driven drug discovery.",
            }),
            # Second call: revised answer
            "Rejuve Bio is a longevity company using AI-driven drug discovery.",
        ]

        answer, confidence = rag_instance._reflect_and_revise(
            self.SAMPLE_QUERY, self.SAMPLE_CHUNKS, self.SAMPLE_ANSWER
        )

        assert answer != self.SAMPLE_ANSWER
        assert "AI-driven" in answer
        # Revised confidence = original (0.4) + 0.15 boost = 0.55
        assert confidence == pytest.approx(0.55, abs=0.01)
        # Verify the LLM was called twice (reflection + revision)
        assert mock_llm.generate.call_count == 2

    def test_malformed_verdict_returns_original_with_default_confidence(self, rag_instance, mock_llm):
        """Malformed LLM output → graceful fallback to original answer."""
        mock_llm.generate.return_value = "I think the answer looks fine overall."

        answer, confidence = rag_instance._reflect_and_revise(
            self.SAMPLE_QUERY, self.SAMPLE_CHUNKS, self.SAMPLE_ANSWER
        )

        # Should fall back to original answer
        assert answer == self.SAMPLE_ANSWER
        # Default confidence for unexpected format
        assert confidence == pytest.approx(0.5, abs=0.01)

    def test_malformed_json_with_good_prefix_is_accepted(self, rag_instance, mock_llm):
        """Plain text starting with 'GOOD' triggers text-fallback approval."""
        mock_llm.generate.return_value = "GOOD - the answer is well grounded."

        answer, confidence = rag_instance._reflect_and_revise(
            self.SAMPLE_QUERY, self.SAMPLE_CHUNKS, self.SAMPLE_ANSWER
        )

        assert answer == self.SAMPLE_ANSWER
        assert confidence == pytest.approx(0.75, abs=0.01)

    def test_invalid_confidence_value_defaults_safely(self, rag_instance, mock_llm):
        """Non-numeric confidence value → defaults to 0.5, does not crash."""
        mock_llm.generate.return_value = json.dumps({
            "verdict": "GOOD",
            "confidence": "not-a-number",
        })

        answer, confidence = rag_instance._reflect_and_revise(
            self.SAMPLE_QUERY, self.SAMPLE_CHUNKS, self.SAMPLE_ANSWER
        )

        assert answer == self.SAMPLE_ANSWER
        assert confidence == pytest.approx(0.5, abs=0.01)

    def test_confidence_clamped_to_valid_range(self, rag_instance, mock_llm):
        """Out-of-range confidence (e.g., 1.5 or -0.3) is clamped to [0.0, 1.0]."""
        mock_llm.generate.return_value = json.dumps({
            "verdict": "GOOD",
            "confidence": 1.5,
        })

        _, confidence = rag_instance._reflect_and_revise(
            self.SAMPLE_QUERY, self.SAMPLE_CHUNKS, self.SAMPLE_ANSWER
        )
        assert confidence == 1.0

    def test_llm_exception_returns_original(self, rag_instance, mock_llm):
        """If the reflection LLM call throws, fall back gracefully."""
        mock_llm.generate.side_effect = Exception("API timeout")

        answer, confidence = rag_instance._reflect_and_revise(
            self.SAMPLE_QUERY, self.SAMPLE_CHUNKS, self.SAMPLE_ANSWER
        )

        assert answer == self.SAMPLE_ANSWER
        assert confidence == 0.5


# ===========================================================================
# Tests for _decompose_query
# ===========================================================================

class TestDecomposeQuery:
    """Tests for query decomposition with heuristic gating."""

    def test_simple_query_skips_llm(self, rag_instance, mock_llm):
        """Single-topic query without split keywords → no LLM call."""
        result = rag_instance._decompose_query("What is BRCA1?")

        assert result == ["What is BRCA1?"]
        mock_llm.generate.assert_not_called()

    def test_query_with_and_triggers_llm(self, rag_instance, mock_llm):
        """'and' keyword triggers LLM decomposition."""
        mock_llm.generate.return_value = json.dumps({
            "sub_queries": [
                "What does Rejuve Bio do?",
                "What are Methuselah flies?",
            ]
        })

        result = rag_instance._decompose_query(
            "What does Rejuve Bio do and what are Methuselah flies?"
        )

        assert len(result) == 2
        assert "Rejuve Bio" in result[0]
        assert "Methuselah" in result[1]
        mock_llm.generate.assert_called_once()

    def test_query_with_compare_triggers_llm(self, rag_instance, mock_llm):
        """'compare' keyword triggers LLM decomposition."""
        mock_llm.generate.return_value = json.dumps({
            "sub_queries": [
                "What is the role of BRCA1 in breast cancer?",
                "What is the role of TP53 in breast cancer?",
            ]
        })

        result = rag_instance._decompose_query(
            "Compare BRCA1 and TP53 in breast cancer"
        )

        assert len(result) == 2
        mock_llm.generate.assert_called_once()

    def test_llm_returns_single_item_no_split(self, rag_instance, mock_llm):
        """LLM returns a single sub-query → same as original, no split."""
        mock_llm.generate.return_value = json.dumps({
            "sub_queries": ["What is BRCA1 and its function?"]
        })

        result = rag_instance._decompose_query("What is BRCA1 and its function?")

        assert len(result) == 1

    def test_malformed_llm_output_falls_back(self, rag_instance, mock_llm):
        """Garbage LLM output → fall back to original query."""
        mock_llm.generate.return_value = "I cannot decompose this query."

        result = rag_instance._decompose_query(
            "Tell me about BRCA1 and TP53"
        )

        assert result == ["Tell me about BRCA1 and TP53"]


# ===========================================================================
# Tests for _confidence_label
# ===========================================================================

class TestConfidenceLabel:
    """Tests for confidence score → qualitative label mapping."""

    def test_high_confidence(self):
        from app.main import _confidence_label
        assert _confidence_label(0.9) == "high"
        assert _confidence_label(0.7) == "high"

    def test_medium_confidence(self):
        from app.main import _confidence_label
        assert _confidence_label(0.6) == "medium"
        assert _confidence_label(0.5) == "medium"

    def test_low_confidence(self):
        from app.main import _confidence_label
        assert _confidence_label(0.4) == "low"
        assert _confidence_label(0.0) == "low"

    def test_boundary_values(self):
        from app.main import _confidence_label
        assert _confidence_label(0.7) == "high"    # exact boundary
        assert _confidence_label(0.69) == "medium"  # just below
        assert _confidence_label(0.5) == "medium"   # exact boundary
        assert _confidence_label(0.49) == "low"     # just below
