"""
Two-stage retrieval: Cross-Encoder re-ranker.

Wraps ``sentence_transformers.CrossEncoder`` (already a project dependency)
to re-score a wide candidate set returned by Qdrant and return only the
top-k highest-confidence chunks to the LLM.

The model is lazy-loaded on first use so it does not block Flask startup.
All gunicorn workers share the same loaded weights via ``--preload``.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Public model ID.  ~568 MB, runs fully on CPU.
_RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
)

_model = None  # Singleton — loaded once, shared across workers.


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            logger.info(f"Loading re-ranker model: {_RERANKER_MODEL}")
            _model = CrossEncoder(_RERANKER_MODEL, max_length=512)
            logger.info("Re-ranker model loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load re-ranker model: {exc}")
            raise
    return _model


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    text_key: str = "text",
) -> list[dict[str, Any]]:
    """
    Score each candidate against *query* using the cross-encoder and return
    the top-*k* highest-scoring dicts (highest first).

    Args:
        query:      The original user query string.
        candidates: List of Qdrant payload dicts.  Each must contain the key
                    specified by *text_key* (default ``"text"``).
        top_k:      Maximum number of results to return.
        text_key:   The key inside each candidate dict that holds the text.

    Returns:
        Sorted list of up to *top_k* candidate dicts, highest score first.
        The score is injected as ``_rerank_score`` for transparency/logging.
    """
    if not candidates:
        return []

    model = _get_model()

    pairs = [(query, c.get(text_key, "")) for c in candidates]
    scores = model.predict(pairs)

    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, candidate in scored[:top_k]:
        enriched = dict(candidate)
        enriched["_rerank_score"] = float(score)
        results.append(enriched)

    logger.debug(
        f"rerank: {len(candidates)} candidates → top {len(results)} "
        f"(scores: {[round(r['_rerank_score'], 3) for r in results]})"
    )
    return results
