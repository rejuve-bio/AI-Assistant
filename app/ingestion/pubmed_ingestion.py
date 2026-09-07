"""
PubMed background ingestion service.

Fetches recent abstracts from PubMed for a configurable list of longevity /
aging topics, embeds them, and upserts them into a dedicated Qdrant collection
(``pubmed_abstracts`` by default).

Design principles:
- **Idempotent**: each abstract is keyed by its PMID; re-running never
  duplicates data.
- **Incremental**: a MongoDB ``paper_index`` collection tracks ingested papers
  so only genuinely new papers are embedded (saves embedding API calls).
- **Non-blocking**: intended to be called from an APScheduler background job,
  not from a request path.
- **Pluggable**: inherits from :class:`BaseIngester` so the same dedup /
  tracking layer is shared with future source adapters (bioRxiv, etc.).
"""
from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any

import requests

from app.ingestion.base_ingester import BaseIngester, Paper, content_hash

logger = logging.getLogger(__name__)

PUBMED_COLLECTION = os.getenv("PUBMED_COLLECTION", "pubmed_abstracts")
NCBI_API_KEY      = os.getenv("ncbi_api_key", "")
NCBI_BASE         = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# With an API key NCBI allows 10 req/s; without it: 3 req/s.
_NCBI_DELAY       = 0.12 if NCBI_API_KEY else 0.35

# How many abstracts to fetch per topic per run.
ABSTRACTS_PER_TOPIC = int(os.getenv("PUBMED_ABSTRACTS_PER_TOPIC", "20"))
# How many days back to search for new papers.
LOOKBACK_DAYS       = int(os.getenv("PUBMED_LOOKBACK_DAYS", "7"))
# Path to the topic list file (one topic per line).
TOPICS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "pubmed_topics.txt"
)


# ---------------------------------------------------------------------------
# Topic loading
# ---------------------------------------------------------------------------

def load_topics(topics_file: str = TOPICS_FILE) -> list[str]:
    """Load topics from a plain-text file (one per line, # comments ignored)."""
    topics_file = os.path.abspath(topics_file)
    if not os.path.exists(topics_file):
        logger.warning(f"Topics file not found: {topics_file}")
        return []
    topics = []
    with open(topics_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                topics.append(line)
    logger.info(f"Loaded {len(topics)} PubMed topics from {topics_file}")
    return topics


# ---------------------------------------------------------------------------
# NCBI helpers
# ---------------------------------------------------------------------------

def _ncbi_params(**kwargs) -> dict:
    """Base params shared by all NCBI E-utility requests."""
    p = {
        "tool": "rejuve-ai-assistant-ingestion",
        "email": "assistant@rejuve.bio",
        **kwargs,
    }
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p


def _fetch_pmids(topic: str, max_results: int, min_date: str) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    try:
        resp = requests.get(
            f"{NCBI_BASE}/esearch.fcgi",
            params=_ncbi_params(
                db="pubmed",
                term=topic,
                retmax=max_results,
                retmode="json",
                sort="relevance",
                mindate=min_date,
                datetype="pdat",
            ),
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception as exc:
        logger.error(f"esearch failed for topic '{topic}': {exc}")
        return []


def _parse_articles(content: bytes, topic: str) -> list[Paper]:
    """Parse PubMed XML response into a list of ``Paper`` objects."""
    try:
        root = ET.fromstring(content)
    except Exception as exc:
        logger.error(f"XML parsing failed: {exc}")
        return []

    def _text(el) -> str:
        return "".join(el.itertext()).strip() if el is not None else ""

    papers: list[Paper] = []
    for article in root.findall(".//PubmedArticle"):
        pmid  = article.findtext(".//PMID", "")
        title = _text(article.find(".//ArticleTitle"))
        if not title or len(title) < 5:
            continue
        abstract_parts = [_text(t) for t in article.findall(".//AbstractText")]
        abstract = " ".join(p for p in abstract_parts if p)
        if not abstract:
            continue  # Skip papers without abstracts — nothing to embed.
        year = (
            article.findtext(".//PubDate/Year")
            or article.findtext(".//PubDate/MedlineDate", "")[:4]
        )
        authors = [
            f"{a.findtext('LastName', '')} {a.findtext('Initials', '')}".strip()
            for a in article.findall(".//Author")[:3]
        ]
        # Try to extract a DOI for cross-source deduplication.
        doi = ""
        for aid in article.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").strip()
                break

        papers.append(Paper(
            source_id=pmid,
            source="pubmed",
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            doi=doi,
            topic=topic,
        ))
    return papers


# ---------------------------------------------------------------------------
# Main ingester
# ---------------------------------------------------------------------------

class PubMedIngester(BaseIngester):
    """
    Orchestrates fetch → dedup → embed → upsert for PubMed abstracts.

    Inherits shared deduplication and tracking logic from
    :class:`~app.ingestion.base_ingester.BaseIngester`.
    """

    # ------ BaseIngester interface ------

    def fetch_papers(
        self,
        topic: str,
        min_date: str,
        max_results: int,
    ) -> list[Paper]:
        """Fetch papers from PubMed for a single topic."""
        time.sleep(_NCBI_DELAY)
        pmids = _fetch_pmids(topic, max_results, min_date)
        if not pmids:
            return []

        # Fetch full article XML.
        time.sleep(_NCBI_DELAY)
        try:
            resp = requests.get(
                f"{NCBI_BASE}/efetch.fcgi",
                params=_ncbi_params(
                    db="pubmed",
                    id=",".join(pmids),
                    retmode="xml",
                ),
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.error(f"efetch failed: {exc}")
            return []

        return _parse_articles(resp.content, topic)

    # ------ Ingestion orchestrator ------

    def ingest(
        self,
        topics: list[str] | None = None,
        lookback_days: int = LOOKBACK_DAYS,
    ) -> dict[str, Any]:
        """
        Run one ingestion cycle.

        Returns a summary dict::

            {
                "topics_processed": int,
                "new_papers_ingested": int,
                "skipped_duplicates": int,
                "errors": int,
                "started_at": str,
                "finished_at": str,
            }
        """
        started_at = datetime.utcnow()
        topics = topics or load_topics()
        if not topics:
            logger.warning("PubMedIngester.ingest: no topics configured, aborting.")
            return {"error": "no topics configured"}

        min_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y/%m/%d")
        indexed_ids = self.get_indexed_ids()

        stats = {
            "topics_processed": 0,
            "new_papers_ingested": 0,
            "skipped_duplicates": 0,
            "errors": 0,
            "started_at": started_at.isoformat(),
        }

        # Ensure the dedicated Qdrant collection exists.
        self.qdrant.ensure_collection_exists(PUBMED_COLLECTION)

        for topic in topics:
            logger.info(f"[ingestion] Fetching papers for: '{topic}' (since {min_date})")
            papers = self.fetch_papers(topic, min_date, ABSTRACTS_PER_TOPIC)

            new_papers = [p for p in papers if p.source_id not in indexed_ids]
            logger.info(
                f"[ingestion] '{topic}': {len(papers)} hits, "
                f"{len(new_papers)} new, {len(papers) - len(new_papers)} already indexed"
            )
            stats["skipped_duplicates"] += len(papers) - len(new_papers)

            if not new_papers:
                stats["topics_processed"] += 1
                continue

            for paper in new_papers:
                try:
                    chash = content_hash(paper.source_id, paper.abstract)

                    # Build the text to embed: title + abstract for richer context.
                    embed_text = f"{paper.title}. {paper.abstract}"

                    # Metadata stored in the Qdrant payload alongside the text.
                    metadata = {
                        "source":       paper.source,
                        "pmid":         paper.source_id,
                        "title":        paper.title,
                        "authors":      ", ".join(paper.authors),
                        "year":         paper.year,
                        "url":          paper.url,
                        "doi":          paper.doi,
                        "topic":        topic,
                        "content_hash": chash,
                        "ingested_at":  datetime.utcnow().isoformat(),
                    }

                    # Use the existing Qdrant upsert path (handles embedding internally).
                    self.qdrant.upsert_data(
                        collection_name=PUBMED_COLLECTION,
                        data=None,
                        is_content=True,
                        chunks=[embed_text],   # single chunk per abstract
                        metadata=metadata,
                    )

                    self.mark_ingested(paper, chash, topic)
                    indexed_ids.add(paper.source_id)  # update local set for this run
                    stats["new_papers_ingested"] += 1

                except Exception as exc:
                    logger.error(f"[ingestion] Failed to ingest PMID {paper.source_id}: {exc}")
                    stats["errors"] += 1

            stats["topics_processed"] += 1

        stats["finished_at"] = datetime.utcnow().isoformat()
        logger.info(
            f"[ingestion] Cycle complete: {stats['new_papers_ingested']} new papers, "
            f"{stats['skipped_duplicates']} skipped, {stats['errors']} errors."
        )
        return stats
