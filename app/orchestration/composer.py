"""Response aggregation and finalization — the last two LangGraph nodes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .contracts import AgentState

logger = logging.getLogger(__name__)

ANNOTATION_DB = "annotation database"
KNOWLEDGE_BASE = "knowledge base"
GALAXY_PLATFORM = "Galaxy platform"

# Confidence score labels for user-facing output
_CONFIDENCE_THRESHOLDS = {
    "high": 0.7,    # >= 0.7
    "medium": 0.5,  # >= 0.5
    "low": 0.0,     # < 0.5
}


def _confidence_label(score: float) -> str:
    """Map a numeric confidence score (0.0–1.0) to a qualitative label."""
    if score >= _CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    elif score >= _CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


StatusEmitter = Callable[..., None]


class ResponseComposer:
    """Owns the aggregator and finalizer LangGraph nodes.

    Constructed once at startup and injected into ``AssistantWorkflow``.
    """

    def __init__(
        self,
        advanced_llm: Any,
        aggregator_prompt: str,
        hypothesis_aggregator_prompt: str,
        emit_status: StatusEmitter,
    ) -> None:
        self._llm = advanced_llm
        self._aggregator_prompt = aggregator_prompt
        self._hypothesis_aggregator_prompt = hypothesis_aggregator_prompt
        self._emit_status = emit_status

    # ------------------------------------------------------------------
    # LangGraph node: aggregate
    # ------------------------------------------------------------------

    def aggregate(self, state: AgentState) -> dict[str, Any]:
        """Aggregate responses from all agents with source attribution.

        Ensures that text content is combined coherently and structured JSON data
        (json_format) is always included when available.
        """
        # If an agent already set a final response (e.g. hypothesis failure with stop_pipeline),
        # return it directly without re-aggregating.
        if state.get("stop_pipeline") and state.get("response", {}).get("text"):
            logger.info("stop_pipeline with pre-built response — skipping aggregation")
            return {"response": state["response"]}

        # Hypothesis fast-path — bypass LLM aggregation entirely
        hyp_resp = state.get("hypothesis_response") or {}
        if hyp_resp:
            hyp_succeeded = (
                isinstance(hyp_resp.get("resource"), dict)
                and hyp_resp["resource"].get("type") == "hypothesis"
            )
            if hyp_succeeded:
                hyp_text = hyp_resp.get("text", "")
                sources_footer = self._build_sources_footer(state)
                final_text = hyp_text.rstrip()
                if sources_footer:
                    final_text += "\n\n" + sources_footer
                return {
                    "response": {"text": final_text, "json_format": None, "organism": None},
                    "resource": hyp_resp.get("resource"),
                }
            else:
                hyp_text = hyp_resp.get("text") or (
                    "The hypothesis service is not returning any results at the moment. "
                    "There is nothing I can help with directly, but I can search for "
                    "similar clinical trials and published research — please try asking "
                    "about the topic directly."
                )
                return {
                    "response": {"text": hyp_text, "json_format": None, "organism": None},
                }

        user_query = state.get("user_query", "")
        logger.info("Aggregating responses from multiple agents with source attribution")

        agent_outputs: list[dict[str, Any]] = []
        json_format = None
        organism = None

        # Annotation agent
        annotation_resp = state.get("annotation_response")
        if annotation_resp:
            json_format, organism, needs_confirm = self._aggregate_annotation_response(
                annotation_resp, agent_outputs,
            )
            if needs_confirm:
                return {
                    "response": {
                        "text": annotation_resp.get("text", ""),
                        "json_format": None,
                    }
                }

        resource_to_save = self._aggregate_content_responses(state, agent_outputs)

        # Handle JSON-only case
        if json_format and not agent_outputs:
            nodes = json_format.get("nodes", [])
            logger.info(
                "[note-check] node statuses: %s",
                {n.get("node_id"): n.get("status") for n in nodes},
            )
            failed = [n for n in nodes if n.get("status") is False]
            logger.info("[note-check] failed nodes: %s", [n.get("node_id") for n in failed])
            return {
                "response": {
                    "text": self._build_annotation_text(json_format),
                    "json_format": json_format,
                    "organism": organism,
                }
            }

        if not agent_outputs:
            return {
                "response": {
                    "text": "I couldn't find any relevant information to answer your query.",
                    "json_format": None,
                }
            }

        # LLM aggregation
        try:
            sources_info: list[str] = []
            for output in agent_outputs:
                confidence = output.get("confidence")
                confidence_tag = f" (confidence={confidence:.2f})" if confidence is not None else ""
                logger.info(
                    "=== [%s] source=%s%s ===\n%s",
                    output["agent"],
                    output.get("source", "unknown"),
                    confidence_tag,
                    str(output.get("content", ""))[:300],
                )

                content = output.get("content", "")
                if isinstance(content, dict):
                    content = str(content)
                content = content.strip() if isinstance(content, str) else ""
                if content:
                    source_label = output.get("source", "unknown")
                    if confidence is not None:
                        source_label += f" [confidence: {_confidence_label(confidence)}]"
                    sources_info.append(f"From {source_label}: {content}")

            combined_text = "\n\n".join(sources_info)

            json_note = ""
            if json_format:
                nodes = json_format.get("nodes", [])
                predicates = json_format.get("predicates", [])
                if nodes or predicates:
                    node_ids = [n.get("id", "") for n in nodes if n.get("id")]
                    json_note = (
                        f"\n\nNote: An annotation query was successfully built for "
                        f"{', '.join(node_ids) if node_ids else 'the requested entities'} "
                        f"and will be displayed on the graph."
                    )
                else:
                    json_note = "\n\nNote: Structured annotation data is also available for this query."

            query_types = state.get("query_types", [])
            if "hypothesis_generation" in query_types and state.get("graph_id"):
                prompt = self._hypothesis_aggregator_prompt.format(
                    user_query=user_query, combined_text=combined_text,
                )
            else:
                prompt = self._aggregator_prompt.format(
                    user_query=user_query, combined_text=combined_text, json_note=json_note,
                )

            aggregated_text = self._llm.generate(prompt)
            logger.info("Successfully aggregated response: %s...", aggregated_text[:100])

            sources_footer = self._build_sources_footer(state)
            if sources_footer:
                aggregated_text = aggregated_text.rstrip() + "\n\n" + sources_footer

            confidence_scores = {
                output["agent"]: _confidence_label(output["confidence"])
                for output in agent_outputs
                if "confidence" in output
            }

            return {
                "response": {
                    "text": aggregated_text,
                    "json_format": json_format,
                    "organism": organism,
                    "confidence_scores": confidence_scores,
                },
                "resource": resource_to_save,
            }

        except Exception as e:
            logger.error("Error in aggregation: %s", str(e), exc_info=True)
            fallback_parts = []
            confidence_scores = {}
            for output in agent_outputs:
                if "confidence" in output:
                    confidence_scores[output["agent"]] = _confidence_label(output["confidence"])
                content = output.get("content", "")
                if isinstance(content, dict):
                    content = str(content)
                if content:
                    content_str = content.strip() if isinstance(content, str) else str(content)
                    fallback_parts.append(
                        f"**From {output.get('source', 'unknown')}:**\n{content_str}"
                    )

            fallback_text = "\n\n".join(fallback_parts) if fallback_parts else "Annotation data retrieved."

            return {
                "response": {
                    "text": fallback_text,
                    "json_format": json_format,
                    "organism": organism,
                    "confidence_scores": confidence_scores,
                },
                "resource": resource_to_save,
            }

    # ------------------------------------------------------------------
    # LangGraph node: finalize
    # ------------------------------------------------------------------

    def finalize(self, state: AgentState) -> dict[str, Any]:
        """Finalize and return the response."""
        response = state.get("response", {})
        user_id = state.get("user_id")

        logger.info("Finalizing response for user: %s", user_id)
        logger.info("here is the response : %s", response)

        if not isinstance(response, dict):
            response = {"text": str(response), "json_format": None}
        response.setdefault("text", "")
        # Include the resource (hypothesis graph) if available
        if state.get("resource"):
            response["resource"] = state.get("resource")

        self._emit_status(user=user_id, message=response, status="completed")

        return {"response": response}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_annotation_section(failed: list) -> str:
        missing_parts = []
        for n in failed:
            not_validated = n.get("not_validated")
            if not_validated:
                items = not_validated if isinstance(not_validated, list) else [not_validated]
                for item in items:
                    missing_parts.append(f'"{item}"')
            else:
                props = n.get("properties", {})
                name = next(iter(props.values()), n.get("type", "unknown"))
                missing_parts.append(f'"{name}"')
        verb = "was" if len(missing_parts) == 1 else "were"
        joined = ", ".join(missing_parts)
        return f" Note: {joined} {verb} not found in the database."

    def _build_annotation_text(self, json_format: dict) -> str:
        """Build human-readable text from annotation validation results for truly-failed nodes."""
        nodes = json_format.get("nodes", [])
        # Ignore nodes pending confirmation — those are handled by _build_confirmation_text
        failed = [n for n in nodes if n.get("status") is False and not n.get("needs_confirmation")]
        text = "The annotation structure was created successfully (see structured data)."
        if failed:
            text += self._format_annotation_section(failed)
        return text

    @staticmethod
    def _build_sources_footer(state: dict) -> str:
        """Build a markdown Sources section with clickable links from PubMed and ClinicalTrials."""
        sections = []

        pubmed_resp = state.get("pubmed_response")
        if pubmed_resp:
            papers = pubmed_resp.get("items", [])
            links = [
                f"- [{p.get('title', p.get('pmid', 'Article'))}]({p['url']})"
                for p in papers if p.get("url")
            ]
            if links:
                sections.append("**PubMed Sources:**\n" + "\n".join(links))

        clinical_resp = state.get("clinical_trials_response")
        if clinical_resp:
            trials = clinical_resp.get("items", [])
            links = [
                f"- [{t.get('title', t.get('nct_id', 'Trial'))} ({t.get('nct_id', '')})]({t['url']})"
                for t in trials if t.get("url")
            ]
            if links:
                sections.append("**ClinicalTrials.gov Sources:**\n" + "\n".join(links))

        return "\n\n".join(sections)

    def _aggregate_annotation_response(
        self, annotation_resp: dict, agent_outputs: list,
    ) -> tuple:
        if annotation_resp.get("needs_confirmation"):
            return None, None, True
        text_content = annotation_resp.get("text") or annotation_resp.get("summary") or ""
        json_format = annotation_resp.get("json_format")
        organism = annotation_resp.get("organism") if json_format else None
        if not text_content and json_format:
            text_content = self._build_annotation_text(json_format)
        if text_content:
            agent_outputs.append({
                "agent": "annotation_agent",
                "source": annotation_resp.get("source", ANNOTATION_DB),
                "content": text_content,
            })
        return json_format, organism, False

    @staticmethod
    def _aggregate_content_responses(state: dict, agent_outputs: list) -> Any:
        rag_resp = state.get("rag_response")
        if rag_resp:
            text_content = rag_resp.get("text", "")
            confidence = rag_resp.get("confidence")
            if text_content:
                entry: dict[str, Any] = {
                    "agent": "rag_agent",
                    "source": rag_resp.get("source", KNOWLEDGE_BASE),
                    "content": text_content,
                }
                if confidence is not None:
                    entry["confidence"] = confidence
                agent_outputs.append(entry)

        galaxy_resp = state.get("galaxy_response")
        if galaxy_resp:
            text_content = galaxy_resp.get("text", "")
            if text_content:
                agent_outputs.append({
                    "agent": "galaxy_agent",
                    "source": galaxy_resp.get("source", GALAXY_PLATFORM),
                    "content": text_content,
                })

        biogpt_resp = state.get("biogpt_response")
        if biogpt_resp:
            text_content = biogpt_resp.get("text", "")
            if text_content:
                agent_outputs.append({
                    "agent": "biogpt_agent",
                    "source": biogpt_resp.get("source", "biogpt"),
                    "content": text_content,
                })

        resource_to_save = state.get("resource")

        content_resp = state.get("content_retrieval_response")
        if content_resp:
            content_parts = content_resp.get("text", [])
            if isinstance(content_parts, list):
                for part in content_parts:
                    if isinstance(part, dict) and part.get("content"):
                        agent_outputs.append({
                            "agent": "content_retrieval_agent",
                            "source": part.get("source", "external content"),
                            "content": part["content"],
                        })
            elif isinstance(content_parts, str) and content_parts:
                sources = content_resp.get("sources", ["external content"])
                agent_outputs.append({
                    "agent": "content_retrieval_agent",
                    "source": ", ".join(sources),
                    "content": content_parts,
                })

        pubmed_resp = state.get("pubmed_response")
        if pubmed_resp:
            text_content = pubmed_resp.get("text", "")
            if text_content:
                agent_outputs.append({
                    "agent": "pubmed_agent",
                    "source": pubmed_resp.get("source", "PubMed"),
                    "content": text_content,
                })

        clinical_trials_resp = state.get("clinical_trials_response")
        if clinical_trials_resp:
            text_content = clinical_trials_resp.get("text", "")
            if text_content:
                agent_outputs.append({
                    "agent": "clinical_trials_agent",
                    "source": clinical_trials_resp.get("source", "ClinicalTrials.gov"),
                    "content": text_content,
                })

        return resource_to_save
