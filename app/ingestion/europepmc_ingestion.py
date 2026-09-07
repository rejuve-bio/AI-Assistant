"""
Europe PMC background ingestion service.

Fetches recent abstracts from `Europe PMC <https://europepmc.org>`_ for the
same configurable list of longevity / aging topics used by the PubMed ingester.

Europe PMC aggregates content from PubMed, PubMed Central, preprint servers,
and other life-science sources — so it catches papers that the PubMed-only
ingester misses.

Cross-source deduplication is handled automatically by :class:`BaseIngester`:
if a paper was already ingested from PubMed (matched by DOI), it will be
skipped here.

API docs: https://europepmc.org/RestfulWebService
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

EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# Europe PMC is generous with rate limits but let's be polite.
_EPMC_DELAY = 0.25

# How many results to fetch per topic per run.
EPMC_RESULTS_PER_TOPIC = int(os.getenv("EPMC_RESULTS_PER_TOPIC", "25"))
# How many days back to search for new papers.
EPMC_LOOKBACK_DAYS = int(os.getenv("EPMC_LOOKBACK_DAYS", "7"))
# Qdrant collection to upsert into (shared with PubMed by default).
EPMC_COLLECTION = os.getenv("EPMC_COLLECTION", PUBMED_COLLECTION)


class EuropePMCIngester(BaseIngester):
    """
    Fetches abstracts from Europe PMC and upserts them into Qdrant.

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
        """Fetch papers from Europe PMC for a single topic."""
        # Europe PMC date filter uses FIRST_PDATE:[YYYY-MM-DD TO YYYY-MM-DD]
        today = datetime.utcnow().strftime("%Y-%m-%d")
        # Convert min_date from YYYY/MM/DD to YYYY-MM-DD
        min_date_fmt = min_date.replace("/", "-")
        date_filter = f"FIRST_PDATE:[{min_date_fmt} TO {today}]"

        query = f"({topic}) AND {date_filter}"

        try:
            time.sleep(_EPMC_DELAY)
            resp = requests.get(
                EPMC_BASE,
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": max_results,
                    "resultType": "core",  # includes abstracts
                    "sort": "relevance",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error(f"[europepmc] Search failed for topic '{topic}': {exc}")
            return []

        results = data.get("resultList", {}).get("result", [])
        papers: list[Paper] = []

        for item in results:
            title = (item.get("title") or "").strip()
            abstract = (item.get("abstractText") or "").strip()
            if not title or len(title) < 5 or not abstract:
                continue

            # Build a stable source_id from Europe PMC's own identifiers.
            epmc_id = item.get("id", "")
            source_name = (item.get("source") or "MED").upper()
            source_id = f"{source_name}:{epmc_id}"

            doi = (item.get("doi") or "").strip()

            # Extract authors.
            author_list = item.get("authorList", {}).get("author", [])
            authors = [
                f"{a.get('lastName', '')} {a.get('initials', '')}".strip()
                for a in author_list[:3]
            ]

            year = item.get("pubYear", "")

            # Build URL — prefer DOI link, fall back to Europe PMC page.
            if doi:
                url = f"https://doi.org/{doi}"
            else:
                url = f"https://europepmc.org/article/{source_name}/{epmc_id}"

            papers.append(Paper(
                source_id=source_id,
                source="europepmc",
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
        lookback_days: int = EPMC_LOOKBACK_DAYS,
    ) -> dict[str, Any]:
        """
        Run one Europe PMC ingestion cycle.

        Returns a summary dict with the same shape as
        :meth:`~app.ingestion.pubmed_ingestion.PubMedIngester.ingest`.
        """
        started_at = datetime.utcnow()
        topics = topics or load_topics()
        if not topics:
            logger.warning("EuropePMCIngester.ingest: no topics configured, aborting.")
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

        self.qdrant.ensure_collection_exists(EPMC_COLLECTION)

        for topic in topics:
            logger.info(f"[europepmc] Fetching papers for: '{topic}' (since {min_date})")
            papers = self.fetch_papers(topic, min_date, EPMC_RESULTS_PER_TOPIC)

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
                f"[europepmc] '{topic}': {len(papers)} hits, "
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
                        collection_name=EPMC_COLLECTION,
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
                        f"[europepmc] Failed to ingest {paper.source_id}: {exc}"
                    )
                    stats["errors"] += 1

            stats["topics_processed"] += 1

        stats["finished_at"] = datetime.utcnow().isoformat()
        logger.info(
            f"[europepmc] Cycle complete: {stats['new_papers_ingested']} new papers, "
            f"{stats['skipped_duplicates']} skipped, {stats['errors']} errors."
        )
        return stats
