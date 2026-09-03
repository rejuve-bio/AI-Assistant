"""
Semantic Scholar background ingestion service.

Fetches recent abstracts from `Semantic Scholar <https://www.semanticscholar.org/>`_ 
for the same configurable list of longevity / aging topics.

Semantic Scholar adds value through its AI-curated citation graph.
Cross-source deduplication is handled automatically by :class:`BaseIngester`.
If a paper was already ingested from PubMed or Europe PMC (matched by DOI),
it will be skipped here, keeping the database free of duplicates while
catching uniquely indexed papers.

API docs: https://api.semanticscholar.org/api-docs/graph
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

import requests

from app.ingestion.base_ingester import BaseIngester, Paper, content_hash
from app.ingestion.pubmed_ingestion import load_topics, PUBMED_COLLECTION

logger = logging.getLogger(__name__)

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Semantic Scholar rate limit for unauthenticated users is 100 requests per 5 minutes.
_S2_DELAY = 1.0

# How many results to fetch per topic per run.
S2_RESULTS_PER_TOPIC = int(os.getenv("S2_RESULTS_PER_TOPIC", "25"))
# How many days back to search for new papers (converted to year).
S2_LOOKBACK_DAYS = int(os.getenv("S2_LOOKBACK_DAYS", "7"))
# Qdrant collection to upsert into (shared with PubMed by default).
S2_COLLECTION = os.getenv("S2_COLLECTION", PUBMED_COLLECTION)


class SemanticScholarIngester(BaseIngester):
    """
    Fetches abstracts from Semantic Scholar and upserts them into Qdrant.

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
        """Fetch papers from Semantic Scholar for a single topic."""
        # Semantic Scholar's year filter supports ranges like "2023-" or "2023-2024"
        # min_date is passed as YYYY/MM/DD by BaseIngester logic, we extract the year.
        try:
            min_year = datetime.strptime(min_date, "%Y/%m/%d").year
        except ValueError:
            min_year = datetime.utcnow().year

        year_range = f"{min_year}-"

        try:
            time.sleep(_S2_DELAY)
            headers = {}
            s2_api_key = os.getenv("S2_API_KEY")
            if s2_api_key:
                headers["x-api-key"] = s2_api_key

            resp = requests.get(
                S2_BASE_URL,
                headers=headers,
                params={
                    "query": topic,
                    "year": year_range,
                    "limit": max_results,
                    "fields": "paperId,url,title,abstract,authors,year,externalIds",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error(f"[semanticscholar] Search failed for topic '{topic}': {exc}")
            return []

        results = data.get("data", [])
        papers: list[Paper] = []

        for item in results:
            title = (item.get("title") or "").strip()
            abstract = (item.get("abstract") or "").strip()
            
            # Semantic Scholar sometimes returns papers without abstracts, we skip those
            if not title or len(title) < 5 or not abstract:
                continue

            # source_id will be S2:{paperId}
            s2_id = item.get("paperId", "")
            if not s2_id:
                continue
                
            source_id = f"S2:{s2_id}"

            # Extract DOI if available (often stored in externalIds)
            external_ids = item.get("externalIds", {})
            doi = (external_ids.get("DOI") or "").strip()

            # Extract authors
            author_list = item.get("authors", [])
            authors = [
                a.get("name", "").strip() for a in author_list[:3] if a.get("name")
            ]

            year = str(item.get("year") or "")
            url = item.get("url") or (f"https://doi.org/{doi}" if doi else "")

            papers.append(Paper(
                source_id=source_id,
                source="semanticscholar",
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                url=url,
                doi=doi,
                topic=topic,
            ))

        return papers

    # ------ Ingestion orchestrator ------

    def ingest(
        self,
        topics: list[str] | None = None,
        lookback_days: int = S2_LOOKBACK_DAYS,
    ) -> dict[str, Any]:
        """
        Run one Semantic Scholar ingestion cycle.

        Returns a summary dict with the same shape as
        :meth:`~app.ingestion.pubmed_ingestion.PubMedIngester.ingest`.
        """
        started_at = datetime.utcnow()
        topics = topics or load_topics()
        if not topics:
            logger.warning("SemanticScholarIngester.ingest: no topics configured, aborting.")
            return {"error": "no topics configured"}

        min_date = (
            datetime.utcnow() - timedelta(days=lookback_days)
        ).strftime("%Y/%m/%d")
        indexed_ids = self.get_indexed_ids()

        stats = {
            "topics_processed": 0,
            "new_papers_ingested": 0,
            "skipped_duplicates": 0,
            "errors": 0,
            "started_at": started_at.isoformat(),
        }

        self.qdrant.ensure_collection_exists(S2_COLLECTION)

        for topic in topics:
            logger.info(f"[semanticscholar] Fetching papers for: '{topic}' (since {min_date})")
            papers = self.fetch_papers(topic, min_date, S2_RESULTS_PER_TOPIC)

            new_papers: list[Paper] = []
            for p in papers:
                # First check local ID cache, then cross-source DOI dedup.
                if p.source_id in indexed_ids:
                    stats["skipped_duplicates"] += 1
                elif self.is_duplicate(p):
                    stats["skipped_duplicates"] += 1
                    indexed_ids.add(p.source_id)
                else:
                    new_papers.append(p)

            logger.info(
                f"[semanticscholar] '{topic}': {len(papers)} hits, "
                f"{len(new_papers)} new, {len(papers) - len(new_papers)} already indexed"
            )

            for paper in new_papers:
                try:
                    chash = content_hash(paper.source_id, paper.abstract)
                    embed_text = f"{paper.title}. {paper.abstract}"

                    metadata = {
                        "source":       paper.source,
                        "source_id":    paper.source_id,
                        "title":        paper.title,
                        "authors":      ", ".join(paper.authors),
                        "year":         paper.year,
                        "url":          paper.url,
                        "doi":          paper.doi,
                        "topic":        topic,
                        "content_hash": chash,
                        "ingested_at":  datetime.utcnow().isoformat(),
                    }

                    self.qdrant.upsert_data(
                        collection_name=S2_COLLECTION,
                        data=None,
                        is_content=True,
                        chunks=[embed_text],
                        metadata=metadata,
                    )

                    self.mark_ingested(paper, chash, topic)
                    indexed_ids.add(paper.source_id)
                    stats["new_papers_ingested"] += 1

                except Exception as exc:
                    logger.error(
                        f"[semanticscholar] Failed to ingest {paper.source_id}: {exc}"
                    )
                    stats["errors"] += 1

            stats["topics_processed"] += 1

        stats["finished_at"] = datetime.utcnow().isoformat()
        logger.info(
            f"[semanticscholar] Cycle complete: {stats['new_papers_ingested']} new papers, "
            f"{stats['skipped_duplicates']} skipped, {stats['errors']} errors."
        )
        return stats
