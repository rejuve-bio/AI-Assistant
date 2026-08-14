"""
Unit tests for:
  - Track 2: app/rag/utils/reranker.py  (CrossEncoder re-ranking)
  - Track 3: app/ingestion/pubmed_ingestion.py  (PubMedIngester + dedup)
  - Qdrant._upsert_content_data: dict vs str chunk handling

All app modules are imported inside fixtures so they resolve after conftest.py
has stubbed heavy dependencies into sys.modules.
"""
import pytest
from unittest.mock import MagicMock, patch


# ===========================================================================
# Track 2 – Reranker
# ===========================================================================

class TestReranker:

    def _make_candidates(self, n):
        return [{"text": f"Candidate {i}", "score": i * 0.1} for i in range(n)]

    @patch("app.rag.utils.reranker._get_model")
    def test_returns_top_k(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = list(range(10, 0, -1))
        mock_get_model.return_value = mock_model

        from app.rag.utils.reranker import rerank
        results = rerank("longevity aging", self._make_candidates(10), top_k=3)
        assert len(results) == 3

    @patch("app.rag.utils.reranker._get_model")
    def test_results_sorted_highest_score_first(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        mock_get_model.return_value = mock_model

        from app.rag.utils.reranker import rerank
        candidates = [
            {"text": "Low relevance"},
            {"text": "High relevance"},
            {"text": "Medium relevance"},
        ]
        results = rerank("aging", candidates, top_k=3)
        assert results[0]["text"] == "High relevance"
        assert results[1]["text"] == "Medium relevance"
        assert results[2]["text"] == "Low relevance"

    @patch("app.rag.utils.reranker._get_model")
    def test_rerank_score_injected_into_payload(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8, 0.3]
        mock_get_model.return_value = mock_model

        from app.rag.utils.reranker import rerank
        results = rerank("query", [{"text": "A"}, {"text": "B"}], top_k=2)
        for r in results:
            assert "_rerank_score" in r
            assert isinstance(r["_rerank_score"], float)

    @patch("app.rag.utils.reranker._get_model")
    def test_fewer_candidates_than_top_k_returns_all(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.7, 0.4]
        mock_get_model.return_value = mock_model

        from app.rag.utils.reranker import rerank
        results = rerank("query", [{"text": "A"}, {"text": "B"}], top_k=10)
        assert len(results) == 2

    def test_empty_candidates_returns_empty_list(self):
        from app.rag.utils.reranker import rerank
        assert rerank("query", [], top_k=5) == []

    @patch("app.rag.utils.reranker._get_model")
    def test_original_payload_fields_preserved(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9]
        mock_get_model.return_value = mock_model

        from app.rag.utils.reranker import rerank
        candidates = [{"text": "Body text", "pmid": "12345678", "url": "http://x.com"}]
        results = rerank("query", candidates, top_k=1)
        assert results[0]["pmid"] == "12345678"
        assert results[0]["url"] == "http://x.com"


# ===========================================================================
# Track 3 – PubMedIngester
# ===========================================================================

class TestPubMedIngester:

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def mock_qdrant(self):
        q = MagicMock()
        q.ensure_collection_exists.return_value = None
        q.upsert_data.return_value = "Content Data Successfully Uploaded"
        return q

    @pytest.fixture
    def mock_mongo(self):
        db = MagicMock()
        col = MagicMock()
        col.find.return_value = []
        col.update_one.return_value = None
        db.__getitem__.return_value = col
        return db

    @pytest.fixture
    def ingester(self, mock_qdrant, mock_mongo):
        from app.ingestion.pubmed_ingestion import PubMedIngester
        return PubMedIngester(qdrant_client=mock_qdrant, mongo_db=mock_mongo)

    # ------------------------------------------------------------------
    # NCBI response helpers
    # ------------------------------------------------------------------

    def _search_response(self, pmids):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"esearchresult": {"idlist": pmids}}
        return resp

    def _fetch_response_xml(self, pmid, title, abstract):
        xml = f"""<?xml version="1.0"?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>{pmid}</PMID>
              <Article>
                <ArticleTitle>{title}</ArticleTitle>
                <Abstract><AbstractText>{abstract}</AbstractText></Abstract>
              </Article>
              <AuthorList>
                <Author><LastName>Smith</LastName><Initials>J</Initials></Author>
              </AuthorList>
            </MedlineCitation>
            <PubmedData>
              <History>
                <PubMedPubDate PubStatus="pubmed"><Year>2024</Year></PubMedPubDate>
              </History>
            </PubmedData>
          </PubmedArticle>
        </PubmedArticleSet>"""
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.content = xml.encode()
        return resp

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @patch("app.ingestion.pubmed_ingestion.requests.get")
    def test_ingest_new_paper_is_stored(self, mock_get, ingester, mock_qdrant):
        mock_get.side_effect = [
            self._search_response(["11111111"]),
            self._fetch_response_xml(
                "11111111",
                "mTOR signaling and aging",
                "mTOR is a key regulator of aging."
            ),
        ]
        stats = ingester.ingest(topics=["mTOR aging"], lookback_days=7)
        assert stats["new_papers_ingested"] == 1
        assert stats["errors"] == 0
        mock_qdrant.upsert_data.assert_called_once()

    @patch("app.ingestion.pubmed_ingestion.requests.get")
    def test_already_indexed_pmid_is_skipped(self, mock_get, mock_qdrant, mock_mongo):
        """PMIDs already in MongoDB pubmed_index must be skipped (idempotency)."""
        col = MagicMock()
        col.find.return_value = [{"pmid": "22222222"}]
        mock_mongo.__getitem__.return_value = col

        from app.ingestion.pubmed_ingestion import PubMedIngester
        ingester = PubMedIngester(qdrant_client=mock_qdrant, mongo_db=mock_mongo)
        mock_get.return_value = self._search_response(["22222222"])

        stats = ingester.ingest(topics=["rapamycin aging"], lookback_days=7)
        assert stats["new_papers_ingested"] == 0
        assert stats["skipped_duplicates"] == 1
        mock_qdrant.upsert_data.assert_not_called()

    @patch("app.ingestion.pubmed_ingestion.requests.get")
    def test_paper_without_abstract_is_skipped(self, mock_get, ingester, mock_qdrant):
        xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>33333333</PMID>
              <Article><ArticleTitle>No abstract paper</ArticleTitle></Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>"""
        fetch_resp = MagicMock()
        fetch_resp.raise_for_status.return_value = None
        fetch_resp.content = xml.encode()

        mock_get.side_effect = [self._search_response(["33333333"]), fetch_resp]
        stats = ingester.ingest(topics=["senescence"], lookback_days=7)
        assert stats["new_papers_ingested"] == 0
        mock_qdrant.upsert_data.assert_not_called()

    @patch("app.ingestion.pubmed_ingestion.requests.get")
    def test_ncbi_search_error_is_handled_gracefully(self, mock_get, ingester):
        mock_get.side_effect = Exception("Connection refused")
        stats = ingester.ingest(topics=["FOXO3 aging"], lookback_days=7)
        # Should complete without crashing; no papers ingested
        assert stats["new_papers_ingested"] == 0

    def test_ingest_with_no_topics_returns_error(self, ingester):
        with patch("app.ingestion.pubmed_ingestion.load_topics", return_value=[]):
            stats = ingester.ingest(topics=None)
        assert "error" in stats

    @patch("app.ingestion.pubmed_ingestion.requests.get")
    def test_second_run_does_not_duplicate(self, mock_get, mock_qdrant, mock_mongo):
        """Running ingest twice on the same PMID should only upsert once."""
        col = MagicMock()
        # Run 1: nothing indexed; Run 2: PMID now in index
        col.find.side_effect = [[], [{"pmid": "44444444"}]]
        mock_mongo.__getitem__.return_value = col

        from app.ingestion.pubmed_ingestion import PubMedIngester
        ingester = PubMedIngester(qdrant_client=mock_qdrant, mongo_db=mock_mongo)

        xml_resp = self._fetch_response_xml(
            "44444444", "Sirtuins and lifespan", "Sirtuins extend lifespan via NAD+."
        )
        mock_get.side_effect = [
            self._search_response(["44444444"]),  # run 1 search
            xml_resp,                              # run 1 fetch
            self._search_response(["44444444"]),  # run 2 search
        ]

        stats1 = ingester.ingest(topics=["sirtuins aging"], lookback_days=7)
        stats2 = ingester.ingest(topics=["sirtuins aging"], lookback_days=7)

        assert stats1["new_papers_ingested"] == 1
        assert stats2["new_papers_ingested"] == 0
        assert stats2["skipped_duplicates"] == 1
        assert mock_qdrant.upsert_data.call_count == 1

    @patch("app.ingestion.pubmed_ingestion.requests.get")
    def test_stats_contain_expected_keys(self, mock_get, ingester):
        mock_get.side_effect = [self._search_response([])]
        stats = ingester.ingest(topics=["NAD+ aging"], lookback_days=7)
        for key in ("topics_processed", "new_papers_ingested",
                    "skipped_duplicates", "errors",
                    "started_at", "finished_at"):
            assert key in stats, f"Missing key: {key}"


# ===========================================================================
# Track 2 – Qdrant._upsert_content_data with structured dict chunks
# ===========================================================================

class TestQdrantStructuredChunks:

    @pytest.fixture
    def qdrant_instance(self):
        from app.storage.qdrant import Qdrant
        mock_embed = MagicMock(return_value=[[0.1] * 384, [0.2] * 384])
        q = Qdrant(embedding_model=mock_embed, vector_size=384)
        q.ensure_collection_exists = MagicMock()
        q.client = MagicMock()
        return q

    def test_dict_chunks_embed_text_field_only(self, qdrant_instance):
        """Embedding model must receive only the 'text' strings, not full dicts."""
        chunks = [
            {"text": "Chunk A body", "breadcrumb": "Intro", "section": "Intro"},
            {"text": "Chunk B body", "breadcrumb": "Intro > Methods", "section": "Methods"},
        ]
        qdrant_instance._upsert_content_data("test_col", chunks, {"source": "test"})
        call_args = qdrant_instance.embedding_model.call_args[0][0]
        assert call_args == ["Chunk A body", "Chunk B body"]

    def test_dict_chunks_breadcrumb_in_qdrant_payload(self, qdrant_instance):
        """breadcrumb and section from dict chunks must land in Qdrant payload."""
        qdrant_instance.embedding_model.return_value = [[0.1] * 384]
        chunks = [{"text": "Hello aging", "breadcrumb": "Results", "section": "Results"}]
        qdrant_instance._upsert_content_data("test_col", chunks, {"source": "pdf"})

        points = qdrant_instance.client.upsert.call_args.kwargs["points"]
        payload = points[0].payload
        assert payload["breadcrumb"] == "Results"
        assert payload["section"] == "Results"
        assert payload["text"] == "Hello aging"

    def test_plain_str_chunks_still_work(self, qdrant_instance):
        """Plain string chunks (web content) must still be handled correctly."""
        qdrant_instance.embedding_model.return_value = [[0.1] * 384]
        chunks = ["Plain text chunk for web content"]
        qdrant_instance._upsert_content_data("test_col", chunks, {"source": "web"})

        points = qdrant_instance.client.upsert.call_args.kwargs["points"]
        assert points[0].payload["text"] == "Plain text chunk for web content"
        assert "breadcrumb" not in points[0].payload

    def test_mixed_str_and_dict_chunks_raise_no_error(self, qdrant_instance):
        """If somehow mixed types are passed, it must not crash."""
        qdrant_instance.embedding_model.return_value = [[0.1] * 384, [0.2] * 384]
        chunks = [
            "Plain string chunk",
            {"text": "Dict chunk", "breadcrumb": "B", "section": "S"},
        ]
        # Should not raise
        qdrant_instance._upsert_content_data("test_col", chunks, {"source": "mixed"})
        assert qdrant_instance.client.upsert.called
