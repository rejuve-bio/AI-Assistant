"""
Parquet-based annotation lookup — replaces Neo4j similarity search.

Loads neo4j_property_values.parquet from BIOMNI_DATA_LAKE once at startup,
caches it in memory, and does fast string similarity search using rapidfuzz
(or difflib as fallback). No database credentials required.
"""

import os
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

_BASE = os.environ.get("BIOMNI_DATA_LAKE", "/data/biomni")
NEO4J_DIR = os.path.join(_BASE, "neo4j")          # Rejuve Atomspace exports
PROPERTY_VALUES_FILE = "neo4j_property_values.parquet"


class ParquetAnnotationLookup:
    """
    Drop-in replacement for Neo4jConnection.get_similar_property_values_batch().
    Reads from parquet files — no Neo4j URI, username, or password needed.
    """

    def __init__(self, neo4j_dir: str = None):
        self._neo4j_dir = neo4j_dir or NEO4J_DIR
        self._df = None
        self._value_index = {}
        self._file_mtime = None   # track file modification time for auto-reload

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        """Load property values parquet. Auto-reloads if the file changed on disk."""
        path = os.path.join(self._neo4j_dir, PROPERTY_VALUES_FILE)
        if not os.path.exists(path):
            logger.warning(
                f"Property values parquet not found at {path}. "
                f"Run helper/export_neo4j_to_parquet.py to generate it. "
                f"Annotation validation will skip similarity checks."
            )
            return False

        # Check if file changed since last load (auto-reload when Neo4j exports run)
        try:
            current_mtime = os.path.getmtime(path)
        except OSError:
            return False

        if self._df is not None and self._file_mtime == current_mtime:
            return True  # up to date, use cached version

        try:
            import pandas as pd
            self._df = pd.read_parquet(path)
            self._file_mtime = current_mtime
            self._value_index = {}
            for (nt, pk), group in self._df.groupby(["node_type", "property_key"]):
                self._value_index[(nt, pk)] = group["value"].dropna().tolist()
            logger.info(
                f"{'Reloaded' if self._file_mtime else 'Loaded'} property values: "
                f"{len(self._df)} rows, {len(self._value_index)} lookup keys"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load property values parquet: {e}")
            return False

    def reload(self):
        """Force reload on next request (call after re-exporting parquet files)."""
        self._file_mtime = None
        logger.info("ParquetAnnotationLookup: marked for reload on next request")

    # ── Similarity search ─────────────────────────────────────────────────────

    def get_similar_property_values_batch(
        self,
        label: str,
        property_key: str,
        search_values: List[str],
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find similar property values for a batch of search strings.
        Returns: {search_value: [(matched_value, similarity_score), ...]}
        """
        if not self._load():
            # Parquet not available — return empty (validation will skip)
            return {sv: [] for sv in search_values}

        candidates = self._value_index.get((label, property_key), [])
        if not candidates:
            logger.debug(f"No values found for ({label}, {property_key}) in parquet")
            return {sv: [] for sv in search_values}

        results = {}
        for sv in search_values:
            matches = _find_similar(sv, candidates, top_k=top_k, threshold=threshold)
            results[sv] = matches

        return results

    def get_similar_property_values(
        self,
        label: str,
        property_key: str,
        search_value: str,
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """Single-value version of the batch method."""
        batch = self.get_similar_property_values_batch(
            label, property_key, [search_value], top_k=top_k, threshold=threshold
        )
        return batch.get(search_value, [])

    def get_all_values(self, label: str, property_key: str) -> List[str]:
        """Return all known values for a (node_type, property_key) pair."""
        if not self._load():
            return []
        return self._value_index.get((label, property_key), [])

    def is_available(self) -> bool:
        """True if the parquet file loaded successfully."""
        return self._load()


# ── String similarity helpers ─────────────────────────────────────────────────

def _find_similar(
    query: str,
    candidates: List[str],
    top_k: int = 10,
    threshold: float = 0.3,
) -> List[Tuple[str, float]]:
    """
    Find the most similar strings to query in candidates.
    Uses rapidfuzz if installed (faster), falls back to difflib.
    """
    q_lower = query.lower()

    # Try rapidfuzz first (much faster for large lists)
    try:
        from rapidfuzz import fuzz, process as rfprocess
        matches = rfprocess.extract(
            q_lower,
            [c.lower() for c in candidates],
            scorer=fuzz.WRatio,
            limit=top_k,
            score_cutoff=threshold * 100,
        )
        # Map back to original case
        lower_to_orig = {c.lower(): c for c in candidates}
        return [
            (lower_to_orig.get(m[0], m[0]), round(m[1] / 100, 3))
            for m in matches
        ]
    except ImportError:
        pass

    # Fallback: difflib (stdlib, no extra install needed)
    import difflib
    scored = []
    for c in candidates:
        ratio = difflib.SequenceMatcher(None, q_lower, c.lower()).ratio()
        if ratio >= threshold:
            scored.append((c, round(ratio, 3)))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]
