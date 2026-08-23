"""Query classification and execution-policy decisions, isolated from Flask."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage

from .contracts import AgentName, AgentState
from .registry import AgentRegistry

logger = logging.getLogger(__name__)

ContentSummaryProvider = Callable[[str, list[str] | None], list[dict[str, Any]]]


class QueryPlanner:
    """Converts an LLM classification into a validated, executable agent plan."""

    _TYPE_TO_AGENT = {
        "annotation_biological": AgentName.ANNOTATION.value,
        "annotation_general": AgentName.ANNOTATION.value,
        "hypothesis_generation": AgentName.HYPOTHESIS.value,
        "galaxy": AgentName.GALAXY.value,
        "rag": AgentName.RAG.value,
        "biogpt": AgentName.BIOGPT.value,
    }

    def __init__(
        self,
        llm: Any,
        classifier_prompt: str,
        content_summaries: ContentSummaryProvider,
        registry: AgentRegistry,
    ) -> None:
        self._llm = llm
        self._classifier_prompt = classifier_prompt
        self._content_summaries = content_summaries
        self._registry = registry

    def classify(self, state: AgentState) -> dict[str, Any]:
        resource = state.get("resource")
        graph_id = state.get("graph_id")
        if resource == "hypothesis":
            agent = AgentName.CONTENT_RETRIEVAL.value if graph_id else AgentName.HYPOTHESIS.value
            return self._classification_update(["hypothesis_generation"], [agent])

        prompt = self._classifier_prompt.format(
            query=state["user_query"],
            content_summaries=self._content_summaries(state["user_id"], state.get("content_ids")),
        )
        response = self._llm.generate(prompt)
        query_types = self._normalise_query_types(response)
        if not query_types:
            logger.warning("Classifier output was unusable; using RAG fallback: %r", response)
            query_types = ["rag"]
        return self._classification_update(query_types, self._build_agent_plan(query_types, state))

    def _classification_update(self, query_types: list[str], agents: list[str]) -> dict[str, Any]:
        unknown = set(agents).difference(self._registry.names)
        if unknown:
            raise ValueError(f"Plan contains unregistered agents: {sorted(unknown)}")
        return {
            "query_types": query_types,
            "agents_to_run": agents,
            "agents_completed": [],
            "messages": [HumanMessage(content=f"Query classified as: {', '.join(query_types)}")],
        }

    def _normalise_query_types(self, response: Any) -> list[str]:
        parsed_types = None
        if isinstance(response, dict):
            parsed_types = response.get("query_types")
        elif isinstance(response, str):
            parsed_types = self._parse_json_types(response)

        candidates = parsed_types if isinstance(parsed_types, list) else str(response).replace("and", ",").replace("\n", ",").split(",")
        result: list[str] = []
        for value in candidates[:5]:
            if not isinstance(value, str):
                continue
            normalized = value.strip().lower()
            query_type = self._canonical_type(normalized)
            if query_type and query_type not in result:
                result.append(query_type)
        return result

    @staticmethod
    def _parse_json_types(response: str) -> Any:
        candidate = response.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(candidate).get("query_types")
        except (json.JSONDecodeError, AttributeError):
            match = re.search(r"\{[^{}]+\}", candidate)
            if not match:
                return None
            try:
                return json.loads(match.group()).get("query_types")
            except (json.JSONDecodeError, AttributeError):
                return None

    @staticmethod
    def _canonical_type(value: str) -> str | None:
        aliases = {
            "annotation biological": "annotation_biological",
            "annotation general": "annotation_general",
            "hypothesis": "hypothesis_generation",
        }
        if value in aliases:
            return aliases[value]
        for query_type in ("annotation_biological", "annotation_general", "galaxy", "rag", "hypothesis", "biogpt", "literature"):
            if query_type in value:
                return aliases.get(query_type, query_type)
        return None

    def _build_agent_plan(self, query_types: list[str], state: AgentState) -> list[str]:
        plan: list[str] = []
        if state.get("content_ids") or state.get("urls") or state.get("graph_id"):
            plan.append(AgentName.CONTENT_RETRIEVAL.value)
        for query_type in query_types:
            if query_type == "literature":
                for agent in (AgentName.RAG, AgentName.PUBMED, AgentName.CLINICAL_TRIALS):
                    if agent.value not in plan:
                        plan.append(agent.value)
                continue
            agent = self._TYPE_TO_AGENT.get(query_type)
            if agent and agent not in plan:
                plan.append(agent)
        return plan or [AgentName.RAG.value]


class ExecutionPolicy:
    """Encodes ordering and safe parallelism independently of agent implementations."""

    def next_step(self, state: AgentState) -> str | list[str]:
        if state.get("stop_pipeline"):
            return "aggregator"
        remaining = [
            agent for agent in state.get("agents_to_run", [])
            if agent not in state.get("agents_completed", [])
        ]
        if not remaining:
            return "aggregator"
        if (
            AgentName.CONTENT_RETRIEVAL.value in remaining
            and AgentName.ANNOTATION.value in remaining
            and state.get("graph_id")
        ):
            batch = [AgentName.CONTENT_RETRIEVAL.value]
            if AgentName.BIOGPT.value in remaining:
                batch.append(AgentName.BIOGPT.value)
            return batch
        if AgentName.ANNOTATION.value in remaining and AgentName.BIOGPT.value in remaining:
            return [AgentName.ANNOTATION.value, AgentName.BIOGPT.value]
        return remaining[0]
