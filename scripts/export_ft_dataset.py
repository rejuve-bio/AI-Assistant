#!/usr/bin/env python3
"""
Export the MongoDB ``ft_dataset`` collection to a ``.jsonl`` file suitable for
fine-tuning LLMs.

Supports two output formats:

- **openai**: ``{"messages": [{"role": "system", ...}, {"role": "user", ...},
  {"role": "assistant", ...}]}``
- **gemini**: ``{"contents": [{"role": "user", ...}, {"role": "model", ...}]}``

Usage::

    # Export all QA pairs in OpenAI format
    python scripts/export_ft_dataset.py --format openai --output data/ft_train.jsonl

    # Export only 2026 papers from PubMed
    python scripts/export_ft_dataset.py --format openai --year 2026 --source pubmed

    # Export in Gemini format
    python scripts/export_ft_dataset.py --format gemini --output data/ft_gemini.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = (
    "You are a knowledgeable longevity and aging research assistant. "
    "Answer questions accurately based on the latest scientific literature."
)


def _connect_mongo():
    """Connect to MongoDB using environment variables."""
    mongo_url = os.getenv("MONGO_URL")
    db_name = os.getenv("MONGO_DATABASE", "ai_assistant")
    if not mongo_url:
        logger.error("MONGO_URL is not set in .env")
        sys.exit(1)
    client = MongoClient(mongo_url)
    return client[db_name]


def _build_query(args) -> dict:
    """Build a MongoDB filter from CLI arguments."""
    query: dict = {}
    if args.source:
        query["source"] = args.source
    if args.year:
        query["year"] = args.year
    if args.topic:
        query["topic"] = {"$regex": args.topic, "$options": "i"}
    if args.approved_only:
        query["approved"] = True
    return query


def _format_openai(qa: dict, title: str) -> dict:
    """Format a single QA pair in OpenAI fine-tuning JSONL format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ]
    }


def _format_gemini(qa: dict, title: str) -> dict:
    """Format a single QA pair in Gemini fine-tuning JSONL format."""
    return {
        "contents": [
            {"role": "user", "parts": [{"text": qa["question"]}]},
            {"role": "model", "parts": [{"text": qa["answer"]}]},
        ]
    }


FORMATTERS = {
    "openai": _format_openai,
    "gemini": _format_gemini,
}


def main():
    parser = argparse.ArgumentParser(
        description="Export ft_dataset from MongoDB to .jsonl for LLM fine-tuning."
    )
    parser.add_argument(
        "--format",
        choices=list(FORMATTERS.keys()),
        default="gemini",
        help="Output format (default: gemini)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/ft_train.jsonl",
        help="Output file path (default: data/ft_train.jsonl)",
    )
    parser.add_argument("--source", help="Filter by source (e.g. pubmed, biorxiv)")
    parser.add_argument("--year", help="Filter by publication year (e.g. 2026)")
    parser.add_argument("--topic", help="Filter by topic (regex, case-insensitive)")
    parser.add_argument(
        "--approved-only",
        action="store_true",
        default=False,
        help="Export only approved QA pairs",
    )
    args = parser.parse_args()

    db = _connect_mongo()
    query = _build_query(args)
    formatter = FORMATTERS[args.format]

    docs = list(db["ft_dataset"].find(query))
    if not docs:
        logger.warning("No documents matched the query. Nothing to export.")
        sys.exit(0)

    # Ensure output directory exists.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_pairs = 0
    with open(args.output, "w", encoding="utf-8") as fh:
        for doc in docs:
            title = doc.get("title", "")
            for qa in doc.get("qa_pairs", []):
                if "question" not in qa or "answer" not in qa:
                    continue
                line = formatter(qa, title)
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                total_pairs += 1

    logger.info(
        f"Exported {total_pairs} QA pairs from {len(docs)} papers "
        f"to {args.output} (format: {args.format})"
    )


if __name__ == "__main__":
    main()
