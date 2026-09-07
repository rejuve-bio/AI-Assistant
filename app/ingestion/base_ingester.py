"""
Source-agnostic ingestion base layer.

Provides the ``Paper`` data-class and ``BaseIngester`` abstract base that all
source-specific ingesters (PubMed, bioRxiv, Semantic Scholar, …) must inherit.

The shared ``paper_index`` MongoDB collection tracks every paper that has been
processed, keyed by ``source_id`` *and* ``doi`` so that the same paper arriving
from two different sources is caught as a duplicate.
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical paper representation
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """Source-agnostic paper representation used across all ingesters."""

    source_id: str          # Primary ID from the source (PMID, bioRxiv ID, …)
    source: str             # "pubmed", "biorxiv", "semantic_scholar", …
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    year: str = ""
    url: str = ""
    doi: str = ""           # For cross-source deduplication
    topic: str = ""         # The search topic that surfaced this paper


# ---------------------------------------------------------------------------
# Deduplication / tracking helpers
# ---------------------------------------------------------------------------

PAPER_INDEX_COLLECTION = "paper_index"


def content_hash(source_id: str, abstract: str) -> str:
    """Deterministic hash used to detect content changes for a given paper."""
    return hashlib.md5(f"{source_id}:{abstract}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Abstract base ingester
# ---------------------------------------------------------------------------

class BaseIngester(ABC):
    """
    All source ingesters inherit from this class.

    Subclasses only need to implement :meth:`fetch_papers`.  The shared
    ``paper_index`` logic (dedup, tracking) is handled here.
    """

    def __init__(self, qdrant_client, mongo_db):
        self.qdrant = qdrant_client
        self.mongo = mongo_db

    # ------ abstract method: subclasses implement this ------

    @abstractmethod
    def fetch_papers(
        self,
        topic: str,
        min_date: str,
        max_results: int,
    ) -> list[Paper]:
        """Fetch papers from the source API.  Each subclass implements this."""
        ...

    # ------ shared deduplication ------

    def get_indexed_ids(self) -> set[str]:
        """Return the set of ``source_id`` values already in paper_index."""
        try:
            col = self.mongo[PAPER_INDEX_COLLECTION]
            return {doc["source_id"] for doc in col.find({}, {"source_id": 1})}
        except Exception as exc:
            logger.warning(f"Could not read {PAPER_INDEX_COLLECTION}: {exc}")
            return set()

    def is_duplicate(self, paper: Paper) -> bool:
        """Check paper_index by source_id *or* DOI (cross-source dedup)."""
        query: dict[str, Any] = {"$or": [
            {"source_id": paper.source_id, "source": paper.source},
        ]}
        if paper.doi:
            query["$or"].append({"doi": paper.doi})
        return self.mongo[PAPER_INDEX_COLLECTION].find_one(query) is not None

    def mark_ingested(
        self,
        paper: Paper,
        chash: str,
        topic: str,
    ) -> None:
        """Record a successfully ingested paper in ``paper_index``."""
        try:
            col = self.mongo[PAPER_INDEX_COLLECTION]
            col.update_one(
                {"source_id": paper.source_id, "source": paper.source},
                {"$set": {
                    "source_id": paper.source_id,
                    "source": paper.source,
                    "doi": paper.doi,
                    "title": paper.title,
                    "content_hash": chash,
                    "topic": topic,
                    "ingested_at": datetime.utcnow(),
                }},
                upsert=True,
            )
        except Exception as exc:
            logger.warning(
                f"Could not update {PAPER_INDEX_COLLECTION} for "
                f"{paper.source}:{paper.source_id}: {exc}"
            )
