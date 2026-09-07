"""
Fine-tuning QA-pair generator.

Takes :class:`~app.ingestion.base_ingester.Paper` objects (from any source
ingester) and uses the project's LLM to synthesise question–answer training
pairs.  Results are stored in the MongoDB ``ft_dataset`` collection and can
later be exported to ``.jsonl`` via :mod:`scripts.export_ft_dataset`.

Design principles:
- **Source-agnostic**: works with any ``Paper``, whether it came from PubMed,
  bioRxiv, or a future adapter.
- **Idempotent**: papers already processed are tracked via ``source_id`` in
  ``ft_dataset``; re-running never duplicates QA pairs.
- **Auto-approved**: generated pairs are immediately available for export
  (no manual review gate).
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any

from app.ingestion.base_ingester import Paper

logger = logging.getLogger(__name__)

# MongoDB collection that holds the generated QA dataset.
FT_DATASET_COLLECTION = "ft_dataset"

# Number of QA pairs to request per abstract.
QA_PAIRS_PER_PAPER = int(os.getenv("FT_QA_PAIRS_PER_PAPER", "3"))

# System prompt that instructs the LLM how to generate QA pairs.
_SYSTEM_PROMPT = (
    "You are a scientific data-preparation assistant for a longevity-research "
    "AI.  Your job is to generate high-quality question–answer training pairs "
    "from published research abstracts."
)

# Per-paper user prompt template.
_USER_PROMPT_TEMPLATE = """\
Given the research abstract below, generate exactly {n} question–answer pairs \
suitable for fine-tuning a longevity-research assistant.

Rules:
- Questions should be the kind a researcher or curious user would naturally ask.
- Answers MUST be grounded ONLY in the abstract text — do NOT hallucinate.
- Include specific data points, gene names, or compounds when available.
- Vary question types: factual, mechanistic, comparative.

Title: {title}
Abstract: {abstract}

Respond with ONLY a JSON array (no markdown fences, no explanation):
[{{"question": "...", "answer": "..."}}]
"""


class QAGenerator:
    """
    Generates QA fine-tuning pairs from research papers using an LLM.

    Args:
        llm:       An instance of :class:`~app.llm_handle.llm_models.LLMInterface`.
        mongo_db:  A PyMongo ``Database`` instance for storing generated QA pairs.
    """

    def __init__(self, llm, mongo_db):
        self.llm = llm
        self.mongo = mongo_db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_from_papers(
        self,
        papers: list[Paper],
    ) -> dict[str, Any]:
        """
        Generate QA pairs for a batch of papers.

        Already-processed papers (by ``source_id``) are skipped automatically.

        Returns a summary dict::

            {
                "processed": int,
                "skipped_existing": int,
                "total_qa_pairs": int,
                "errors": int,
            }
        """
        processed_ids = self._get_processed_ids()

        stats = {
            "processed": 0,
            "skipped_existing": 0,
            "total_qa_pairs": 0,
            "errors": 0,
        }

        for paper in papers:
            if paper.source_id in processed_ids:
                stats["skipped_existing"] += 1
                continue

            try:
                qa_pairs = self._generate_qa(paper)
                if not qa_pairs:
                    logger.warning(
                        f"[qa_generator] No QA pairs produced for "
                        f"{paper.source}:{paper.source_id}"
                    )
                    stats["errors"] += 1
                    continue

                self._store_qa(paper, qa_pairs)
                processed_ids.add(paper.source_id)
                stats["processed"] += 1
                stats["total_qa_pairs"] += len(qa_pairs)
                logger.info(
                    f"[qa_generator] Generated {len(qa_pairs)} QA pairs for "
                    f"{paper.source}:{paper.source_id}"
                )

            except Exception as exc:
                logger.error(
                    f"[qa_generator] Failed for {paper.source}:{paper.source_id}: {exc}"
                )
                stats["errors"] += 1

        logger.info(
            f"[qa_generator] Batch complete: {stats['processed']} papers, "
            f"{stats['total_qa_pairs']} QA pairs, {stats['errors']} errors."
        )
        return stats

    def generate_from_index(
        self,
        source: str | None = None,
        year: str | None = None,
        topic: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Generate QA pairs for papers already in ``paper_index`` that have NOT
        yet been processed into ``ft_dataset``.

        Supports optional filtering by source, year, and topic.
        """
        query: dict[str, Any] = {}
        if source:
            query["source"] = source
        if topic:
            query["topic"] = {"$regex": topic, "$options": "i"}

        try:
            index_docs = list(
                self.mongo["paper_index"].find(query).limit(limit)
            )
        except Exception as exc:
            logger.error(f"[qa_generator] Could not read paper_index: {exc}")
            return {"error": str(exc)}

        # Convert index docs back to Paper objects and optionally filter by year.
        papers: list[Paper] = []
        for doc in index_docs:
            if year and doc.get("year", "") != year:
                continue
            papers.append(Paper(
                source_id=doc.get("source_id", doc.get("pmid", "")),
                source=doc.get("source", "pubmed"),
                title=doc.get("title", ""),
                abstract="",  # We'll need abstract from Qdrant or refetch
                doi=doc.get("doi", ""),
                topic=doc.get("topic", ""),
            ))

        # Filter out papers we can't generate QA for (no title).
        papers = [p for p in papers if p.title]

        logger.info(
            f"[qa_generator] Found {len(papers)} candidate papers from paper_index"
        )
        return self.generate_from_papers(papers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_qa(self, paper: Paper) -> list[dict[str, str]]:
        """Call the LLM to produce QA pairs from a single paper."""
        if not paper.abstract:
            return []

        prompt = _USER_PROMPT_TEMPLATE.format(
            n=QA_PAIRS_PER_PAPER,
            title=paper.title,
            abstract=paper.abstract,
        )

        result = self.llm.generate(prompt, system_prompt=_SYSTEM_PROMPT)

        # The LLM may return a parsed list (if JSON was clean) or a raw string.
        qa_pairs = self._parse_qa_response(result)
        return qa_pairs

    @staticmethod
    def _parse_qa_response(result) -> list[dict[str, str]]:
        """
        Normalise the LLM response into a list of ``{"question": …, "answer": …}``
        dicts.  Handles both pre-parsed JSON (list/dict) and raw strings.
        """
        # Already parsed by the LLM wrapper.
        if isinstance(result, list):
            return [
                r for r in result
                if isinstance(r, dict) and "question" in r and "answer" in r
            ]

        # Raw string — try to extract JSON array.
        if isinstance(result, str):
            # Strip markdown code fences if present.
            cleaned = re.sub(r"```json\s*", "", result)
            cleaned = re.sub(r"```\s*$", "", cleaned).strip()
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return [
                        r for r in parsed
                        if isinstance(r, dict) and "question" in r and "answer" in r
                    ]
            except json.JSONDecodeError:
                logger.warning("[qa_generator] Could not parse LLM response as JSON")

        return []

    def _get_processed_ids(self) -> set[str]:
        """Return source_ids already present in ft_dataset."""
        try:
            col = self.mongo[FT_DATASET_COLLECTION]
            return {doc["source_id"] for doc in col.find({}, {"source_id": 1})}
        except Exception:
            return set()

    def _store_qa(self, paper: Paper, qa_pairs: list[dict[str, str]]) -> None:
        """Persist generated QA pairs to MongoDB."""
        doc = {
            "source_id": paper.source_id,
            "source": paper.source,
            "doi": paper.doi,
            "topic": paper.topic,
            "year": paper.year,
            "title": paper.title,
            "qa_pairs": qa_pairs,
            "model_used": getattr(self.llm, "model_provider", "unknown"),
            "generated_at": datetime.utcnow(),
            "approved": True,  # Auto-approved per project decision.
        }
        self.mongo[FT_DATASET_COLLECTION].update_one(
            {"source_id": paper.source_id, "source": paper.source},
            {"$set": doc},
            upsert=True,
        )
