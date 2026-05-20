"""
Embedding-based function selector for Biomni tools.

Before code generation, the CodeExecutor calls get_relevant_functions(query)
which returns the signatures + usage examples of the top-k most relevant
Biomni functions to inject into the code generation prompt.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict

import numpy as np

logger = logging.getLogger(__name__)

SCHEMAS_DIR = Path(__file__).parent / "schemas"
DEFAULT_TOP_K = 15


class BiomniFunctionRetriever:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self._schemas: List[Dict] = []
        self._embeddings: np.ndarray | None = None
        self._load_schemas()

    def _load_schemas(self):
        if not SCHEMAS_DIR.exists():
            logger.warning(f"Biomni schemas directory not found: {SCHEMAS_DIR}")
            return
        for path in sorted(SCHEMAS_DIR.glob("*.json")):
            try:
                with open(path) as f:
                    self._schemas.append(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load schema {path.name}: {e}")
        logger.info(f"Loaded {len(self._schemas)} Biomni function schemas")

    def _build_embeddings(self):
        if not self._schemas or self.embedding_model is None:
            return
        texts = [f"{s['name']}: {s['description']}" for s in self._schemas]
        try:
            self._embeddings = np.array(self.embedding_model.encode(texts))
        except Exception as e:
            logger.warning(f"Failed to build Biomni embeddings: {e}")

    def get_relevant_functions(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Returns a formatted string of the top-k most relevant Biomni function
        signatures for injection into the code generation prompt.
        """
        if not self._schemas:
            return ""

        selected = self._rank_by_embedding(query, top_k)

        lines = ["Available Biomni functions (pre-selected for this task):"]
        for schema in selected:
            lines.append(f"\n# {schema['name']} — {schema['description']}")
            lines.append(f"# Usage: {schema.get('usage', '')}")
            params = schema.get("parameters", {})
            if params:
                param_str = ", ".join(
                    f"{k}: {v}" for k, v in params.items()
                )
                lines.append(f"# Parameters: {param_str}")
            returns = schema.get("returns", "")
            if returns:
                lines.append(f"# Returns: {returns}")

        lines.append(
            "\nImport pattern: from app.tools.biomni.<module> import <function>"
        )
        return "\n".join(lines)

    def _rank_by_embedding(self, query: str, top_k: int) -> List[Dict]:
        if self.embedding_model is None or self._embeddings is None:
            self._build_embeddings()

        if self._embeddings is None:
            return self._rank_by_keyword(query, top_k)

        try:
            q_emb = np.array(self.embedding_model.encode([query])[0])
            scores = self._embeddings @ q_emb / (
                np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9
            )
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [self._schemas[i] for i in top_idx]
        except Exception as e:
            logger.warning(f"Embedding ranking failed: {e}, falling back to keyword")
            return self._rank_by_keyword(query, top_k)

    def _rank_by_keyword(self, query: str, top_k: int) -> List[Dict]:
        query_lower = query.lower()
        scored = []
        for schema in self._schemas:
            text = f"{schema['name']} {schema['description']} {' '.join(schema.get('tags', []))}".lower()
            score = sum(1 for word in query_lower.split() if word in text)
            scored.append((score, schema))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]