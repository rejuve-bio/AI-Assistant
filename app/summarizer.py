"""
Summarization pipeline for annotation graphs.

SummaryPipeline handles annotation output formatting:
  - Annotation graphs: chunk node/edge descriptions → batched LLM summary
  - Annotation by ID: fetch from KG service → LLM summary (Redis-cached)
"""

import json
import logging
import os
import requests
import tiktoken
from collections import defaultdict
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from app.prompts.summarizer_prompts import (
    SUMMARY_PROMPT,
    SUMMARY_PROMPT_BASED_ON_USER_QUERY,
    SUMMARY_PROMPT_CHUNKING,
    SUMMARY_PROMPT_CHUNKING_USER_QUERY,
)
from app.storage.redis import redis_manager

load_dotenv()
logger = logging.getLogger(__name__)

_KG_SERVICE_URL = os.getenv("ANNOTATION_SERVICE_URL", "")


class SummaryPipeline:
    """
    Central summarization pipeline for graphs, annotations, and hypotheses.

    Usage:
        pipeline = SummaryPipeline(llm)

        # Annotation graph (inline data)
        result = pipeline.summarize(graph=graph_dict, query="What genes are involved?")

        # Annotation by graph ID (fetches from KG service, Redis-cached)
        result = pipeline.summarize(graph_id="abc123", token=token, query="...")
    """

    def __init__(self, llm) -> None:
        self.llm = llm
        self.max_tokens = self._resolve_token_limit()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _resolve_token_limit(self) -> int:
        override = os.getenv("SUMMARIZER_TOKEN_LIMIT")
        if override:
            return int(override)
        if self.llm.__class__.__name__ == "OpenAIModel":
            return 100_000
        return 2_000

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def summarize(
        self,
        graph: Optional[dict] = None,
        query: Optional[str] = None,
        graph_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main entry point for annotation/graph summarization."""
        if graph_id:
            return self._summarize_by_id(graph_id, token, query)
        if graph:
            return self._summarize_graph_data(graph, query)
        return {"text": "No graph data provided."}

    def _summarize_by_id(
        self, graph_id: str, token: str, query: Optional[str]
    ) -> Dict[str, Any]:
        """Fetch annotation from KG service and summarize (Redis-cached)."""
        cached = redis_manager.get_graph_by_id(graph_id)
        if cached and cached.get("graph_summary"):
            logger.info("Cache hit for graph_id=%s", graph_id)
            return {"summary": cached["graph_summary"], "text": None}

        logger.info("Fetching annotation for graph_id=%s", graph_id)
        try:
            resp = requests.get(
                f"{_KG_SERVICE_URL}/annotation/{graph_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error("HTTP error fetching annotation: %s", e)
            return {"summary": None, "text": f"Failed to contact graph service: {e}"}

        data = resp.json()
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                data = {}
        if not isinstance(data, dict):
            data = {}

        raw_summary = data.get("summary")
        if not raw_summary:
            return {"summary": None, "text": "Graph is too big — no summaries provided."}

        enhanced = {
            "summary": raw_summary,
            "node_count": data.get("node_count", 0),
            "edge_count": data.get("edge_count", 0),
            "node_count_by_label": data.get("node_count_by_label", {}),
            "edge_count_by_label": data.get("edge_count_by_label", {}),
        }
        redis_manager.create_graph(graph_id=graph_id, graph_summary=json.dumps(enhanced))

        text = None
        if query:
            text = self.llm.generate(
                f"Based on this graph:\n"
                f"Summary: {raw_summary}\n"
                f"Nodes: {enhanced['node_count']} ({enhanced['node_count_by_label']})\n"
                f"Edges: {enhanced['edge_count']} ({enhanced['edge_count_by_label']})\n\n"
                f"Question: {query}\nAnswer:"
            )
        return {"summary": enhanced, "text": text}

    def _summarize_graph_data(self, graph: dict, query: Optional[str]) -> Dict[str, Any]:
        """Chunk raw graph data and run through the LLM in batches."""
        descriptions = self._build_descriptions(graph)
        if not descriptions:
            return {"text": "Empty graph."}

        batches = self._batch_by_tokens(descriptions)
        prev_summary: List = []
        result = None

        for batch in batches:
            if prev_summary:
                prompt = (
                    SUMMARY_PROMPT_CHUNKING_USER_QUERY.format(
                        description=batch, user_query=query, prev_summery=prev_summary
                    )
                    if query
                    else SUMMARY_PROMPT_CHUNKING.format(
                        description=batch, prev_summery=prev_summary
                    )
                )
            else:
                prompt = (
                    SUMMARY_PROMPT_BASED_ON_USER_QUERY.format(
                        description=batch, user_query=query
                    )
                    if query
                    else SUMMARY_PROMPT.format(description=batch)
                )
            result = self.llm.generate(prompt)
            prev_summary = [result]

        return {"text": result}

    #  Graph description helpers                                           
    def _build_descriptions(self, graph: dict, max_nodes: int = 100) -> List[str]:
        if not graph or "nodes" not in graph or not graph["nodes"]:
            return []

        node_ids = {
            graph["nodes"][i]["data"]["id"]
            for i in range(min(max_nodes, len(graph["nodes"])))
        }
        nodes = {
            n["data"]["id"]: n["data"]
            for n in graph["nodes"]
            if n["data"]["id"] in node_ids
        }
        edges = [
            {
                "source": e["data"]["source"],
                "target": e["data"]["target"],
                "label": e["data"]["label"],
            }
            for e in graph.get("edges", [])
            if e["data"]["source"] in node_ids and e["data"]["target"] in node_ids
        ]
        return self._group_descriptions(edges, nodes)

    def _node_description(self, node: dict) -> str:
        parts = []
        for key, value in node.items():
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list) and parsed:
                        parts.append(f"{key.capitalize()}: {', '.join(parsed[:3])}")
                        continue
                except json.JSONDecodeError:
                    pass
            parts.append(f"{key.capitalize()}: {value}")
        return " | ".join(parts)

    def _group_descriptions(self, edges: list, nodes: dict) -> List[str]:
        grouped: Dict[str, list] = defaultdict(list)
        for edge in edges:
            grouped[edge["source"].split(" ")[-1]].append(edge)

        descriptions = []
        for source_id, related in grouped.items():
            source_desc = self._node_description(nodes.get(source_id, {}))
            target_lines = [
                f"{e['label']} -> Target ({e['target']}): "
                f"{self._node_description(nodes.get(e['target'].split(' ')[-1], {}))}"
                for e in related
            ]
            descriptions.append(
                f"Source ({source_id}): {source_desc}\n" + "\n".join(target_lines)
            )
        return descriptions

    def _batch_by_tokens(self, descriptions: List[str]) -> List[List[str]]:
        batches: List[List[str]] = []
        current: List[str] = []
        accumulated = 0

        for desc in descriptions:
            count = len(self.tokenizer.encode(desc))
            if accumulated + count > self.max_tokens and current:
                batches.append(current)
                current, accumulated = [desc], count
            else:
                current.append(desc)
                accumulated += count

        if current:
            batches.append(current)
        return batches
