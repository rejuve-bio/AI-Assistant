"""Combining the agents' outputs into the single answer the caller receives."""
import logging
from typing import Any, Dict

from app.agents.state import AgentState, ANNOTATION_DB, GALAXY_PLATFORM, KNOWLEDGE_BASE
from app.prompts.classifier_prompt import aggregator_prompt, hypothesis_aggregator_prompt
from app.socket_manager import emit_to_user

logger = logging.getLogger(__name__)


class AggregationMixin:
    def _build_annotation_text(self, json_format: dict) -> str:
        """Human-readable text for annotation results, noting anything that failed.

        Nodes pending confirmation are ignored here — _build_confirmation_text
        handles those.
        """
        failed = [
            n for n in json_format.get("nodes", [])
            if n.get("status") is False and not n.get("needs_confirmation")
        ]
        text = "The annotation structure was created successfully (see structured data)."
        if not failed:
            return text

        missing = []
        for node in failed:
            not_validated = node.get("not_validated")
            if not_validated:
                items = not_validated if isinstance(not_validated, list) else [not_validated]
                missing.extend(f'"{item}"' for item in items)
            else:
                props = node.get("properties", {})
                missing.append(f'"{next(iter(props.values()), node.get("type", "unknown"))}"')

        verb = "was" if len(missing) == 1 else "were"
        return f"{text} Note: {', '.join(missing)} {verb} not found in the database."


    def _build_sources_footer(self, state: dict) -> str:
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


    def _aggregate_annotation_response(self, annotation_resp: dict, agent_outputs: list) -> tuple:
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
                "content": text_content
            })
        return json_format, organism, False


    _CONTENT_AGENTS = (
        ("rag_response", "rag_agent", KNOWLEDGE_BASE),
        ("galaxy_response", "galaxy_agent", GALAXY_PLATFORM),
        ("biogpt_response", "biogpt_agent", "biogpt"),
        ("content_retrieval_response", "content_retrieval_agent", "external content"),
        ("pubmed_response", "pubmed_agent", "PubMed"),
        ("clinical_trials_response", "clinical_trials_agent", "ClinicalTrials.gov"),
    )

    def _aggregate_content_responses(self, state: dict, agent_outputs: list) -> Any:
        """Collect each content agent's prose into `agent_outputs`.

        Returns the resource to persist, which rides along on the state rather
        than on any single agent's response.
        """
        for state_key, agent, default_source in self._CONTENT_AGENTS:
            resp = state.get(state_key)
            if not resp:
                continue

            content = resp.get("text", "")
            # content_retrieval returns a list of {source, content} parts; the
            # rest return one string.
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("content"):
                        agent_outputs.append({
                            "agent": agent,
                            "source": part.get("source", default_source),
                            "content": part["content"],
                        })
            elif content:
                agent_outputs.append({
                    "agent": agent,
                    "source": resp.get("source")
                              or ", ".join(resp.get("sources", [default_source])),
                    "content": content,
                })

        return state.get("resource")


    def _aggregate_responses(self, state: AgentState) -> Dict[str, Any]:
        """Combine every agent's output into one answer with source attribution.

        Several cases never reach the LLM: a halted pipeline, a hypothesis
        result, an annotation-only result, or nothing at all. Each is handled by
        its own helper below, so this reads as the order they're checked in.
        """
        for early in (self._stop_pipeline_response(state),
                      self._hypothesis_fast_path(state)):
            if early:
                return early

        user_query = state.get("user_query", "")
        logger.info("Aggregating responses from multiple agents with source attribution")

        agent_outputs = []
        json_format = None
        organism = None

        annotation_resp = state.get("annotation_response")
        if annotation_resp:
            json_format, organism, needs_confirm = self._aggregate_annotation_response(
                annotation_resp, agent_outputs
            )
            if needs_confirm:
                return {"response": {"text": annotation_resp.get("text", ""), "json_format": None}}

        resource_to_save = self._aggregate_content_responses(state, agent_outputs)

        if json_format and not agent_outputs:
            # An annotation was built but no agent produced prose to merge.
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

        return self._llm_aggregate(
            state, user_query, agent_outputs, json_format, organism, resource_to_save
        )

    def _stop_pipeline_response(self, state: AgentState):
        """An agent halted the run (e.g. the referenced graph wasn't found).

        Its message is already user-facing and carries a status/reason, so it is
        returned verbatim rather than being paraphrased by LLM aggregation.
        """
        if not state.get("stop_pipeline"):
            return None
        resp = state.get("content_retrieval_response") or state.get("response") or {}
        text = resp.get("text", "")
        if not text:
            return None
        logger.info("stop_pipeline — returning structured fallback response")
        return {
            "response": {
                "text": text,
                "json_format": resp.get("json_format"),
                "status": resp.get("status", "needs_input"),
                "reason": resp.get("reason"),
            }
        }

    def _hypothesis_fast_path(self, state: AgentState):
        """Hypothesis results bypass LLM aggregation entirely."""
        resp = state.get("hypothesis_response") or {}
        if not resp:
            return None

        resource = resp.get("resource")
        succeeded = isinstance(resource, dict) and resource.get("type") == "hypothesis"
        if succeeded:
            text = resp.get("text", "").rstrip()
            footer = self._build_sources_footer(state)
            if footer:
                text += "\n\n" + footer
            return {
                "response": {"text": text, "json_format": None, "organism": None},
                "resource": resource,
            }

        return {
            "response": {
                "text": resp.get("text") or (
                    "The hypothesis service is not returning any results at the moment. "
                    "There is nothing I can help with directly, but I can search for similar "
                    "clinical trials and published research — please try asking about the "
                    "topic directly."
                ),
                "json_format": None,
                "organism": None,
                "status": resp.get("status", "needs_input"),
                "reason": resp.get("reason"),
            },
        }

    @staticmethod
    def _annotation_note(json_format):
        """Tell the aggregating model how much of the annotation actually resolved.

        Without this it happily reports a partial annotation as a success.
        """
        if not json_format:
            return ""
        nodes = json_format.get("nodes", [])
        predicates = json_format.get("predicates", [])
        if not (nodes or predicates):
            return "\n\nNote: Structured annotation data is also available for this query."

        failed = [n for n in nodes if n.get("status") is False]
        if failed:
            missing = ", ".join(
                f"'{(n.get('properties') or {}).get('gene_name') or n.get('id') or n.get('node_id')}'"
                for n in failed
            )
            return (
                f"\n\nNote: Only a partial annotation structure was built — these entities were "
                f"NOT found in the database and remain unresolved: {missing}. Do NOT describe the "
                f"annotation as successful or complete; state plainly that they were not found."
            )
        node_ids = [n.get("id", "") for n in nodes if n.get("id")]
        return (
            f"\n\nNote: An annotation query was successfully built for "
            f"{', '.join(node_ids) if node_ids else 'the requested entities'} "
            f"and will be displayed on the graph."
        )

    @staticmethod
    def _readable_content(output):
        content = output.get("content", "")
        if isinstance(content, dict):
            content = str(content)
        return content.strip() if isinstance(content, str) else ""

    def _llm_aggregate(self, state, user_query, agent_outputs, json_format,
                       organism, resource_to_save):
        """Ask the model to merge the agents' outputs; fall back to concatenating
        them verbatim if that fails — a joined answer beats no answer."""
        try:
            sources_info = []
            for output in agent_outputs:
                logger.info("=== [%s] source=%s ===\n%s",
                            output["agent"],
                            output.get("source", "unknown"),
                            str(output.get("content", ""))[:300])
                content = self._readable_content(output)
                if content:
                    sources_info.append(f"From {output.get('source', 'unknown')}: {content}")

            combined_text = "\n\n".join(sources_info)
            query_types = state.get("query_types", [])

            if "hypothesis_generation" in query_types and state.get("graph_id"):
                prompt = hypothesis_aggregator_prompt.format(
                    user_query=user_query, combined_text=combined_text
                )
            else:
                prompt = aggregator_prompt.format(
                    user_query=user_query,
                    combined_text=combined_text,
                    json_note=self._annotation_note(json_format),
                )

            aggregated_text = self.advanced_llm.generate(prompt)
            logger.info(f"Successfully aggregated response: {aggregated_text[:100]}...")

            footer = self._build_sources_footer(state)
            if footer:
                aggregated_text = aggregated_text.rstrip() + "\n\n" + footer

            return {
                "response": {
                    "text": aggregated_text,
                    "json_format": json_format,
                    "organism": organism,
                },
                "resource": resource_to_save,
            }

        except Exception as e:
            logger.error(f"Error in aggregation: {str(e)}", exc_info=True)
            parts = [
                f"**From {o.get('source', 'unknown')}:**\n{self._readable_content(o)}"
                for o in agent_outputs if self._readable_content(o)
            ]
            return {
                "response": {
                    "text": "\n\n".join(parts) if parts else "Annotation data retrieved.",
                    "json_format": json_format,
                    "organism": organism,
                },
                "resource": resource_to_save,
            }



    def _finalize_response(self, state: AgentState) -> Dict[str, Any]:
        """Finalize and return the response"""
        response = state.get("response", {})
        user_id = state.get("user_id")
        
        logger.info(f"Finalizing response for user: {user_id}")
        logger.info(f"here is the response : {response}")
        
        # Ensure response has correct structure
        if not isinstance(response, dict):
            response = {"text": str(response), "json_format": None}
        response.setdefault("text", "")
         # Include the resource (hypothesis graph) if available
        if state.get("resource"):
            response["resource"] = state.get("resource")

        emit_to_user(user=user_id, message=response, status="completed")

        return {"response": response}

