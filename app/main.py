import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from .annotation_graph.annotated_graph import Graph
from .biogpt_agent.biogpt import BioGPTAgentOpenVINO
from .Galaxy_integration.galaxy import GalaxyHandler
from .hypothesis_generation.hypothesis import HypothesisGeneration
from .prompts.classifier_prompt import aggregator_prompt, hypothesis_aggregator_prompt
from .prompts.conversation_handler import conversation_prompt
from .rag.rag import RAG
from .socket_manager import emit_to_user
from .summarizer import SummaryPipeline
from .workflow import ANALYZING_MSG, GALAXY_PLATFORM, KNOWLEDGE_BASE, AgentPipeline

load_dotenv()
logger = logging.getLogger(__name__)


class AiAssistance:

    def __init__(
        self,
        advanced_llm,
        basic_llm,
        schema_handler,
        fly_schema_handler=None,
        qdrant_client=None,
        embedding_model=None,
        mongo_db_manager=None,
    ) -> None:
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.annotation_graph = Graph(advanced_llm, schema_handler, fly_schema_handler=fly_schema_handler)
        self.graph_summarizer = SummaryPipeline(self.advanced_llm)
        self.rag = RAG(llm=advanced_llm, qdrant_client=qdrant_client)
        self.store = mongo_db_manager
        self.hypothesis_generation = HypothesisGeneration(advanced_llm)
        self.galaxy_handler = GalaxyHandler(basic_llm, qdrant_client, embedding_model)
        self.embedding_model = embedding_model
        self.biogpt = BioGPTAgentOpenVINO(llm=advanced_llm)
        self.biogpt._load_if_needed()

        logger.info("AiAssistance initialized with advanced_llm: %s", type(self.advanced_llm).__name__)
        logger.info("Galaxy handler initialized: %s", type(self.galaxy_handler).__name__)

        self.pipeline = AgentPipeline(
            annotation_graph=self.annotation_graph,
            rag=self.rag,
            hypothesis_generation=self.hypothesis_generation,
            galaxy_handler=self.galaxy_handler,
            biogpt=self.biogpt,
            basic_llm=self.basic_llm,
            advanced_llm=self.advanced_llm,
            store=self.store,
            graph_summarizer=self.graph_summarizer,
            aggregator=self._aggregate_responses,
            finalizer=self._finalize_response,
        )

    # ------------------------------------------------------------------ #
    #  Aggregation & finalization (LangGraph nodes injected into pipeline) #
    # ------------------------------------------------------------------ #

    def _build_sources_footer(self, state: dict) -> str:
        sections = []
        pubmed_resp = state.get("pubmed_response")
        if pubmed_resp:
            links = [
                f"- [{p.get('title', p.get('pmid', 'Article'))}]({p['url']})"
                for p in pubmed_resp.get("items", []) if p.get("url")
            ]
            if links:
                sections.append("**PubMed Sources:**\n" + "\n".join(links))
        clinical_resp = state.get("clinical_trials_response")
        if clinical_resp:
            links = [
                f"- [{t.get('title', t.get('nct_id', 'Trial'))} ({t.get('nct_id', '')})]({t['url']})"
                for t in clinical_resp.get("items", []) if t.get("url")
            ]
            if links:
                sections.append("**ClinicalTrials.gov Sources:**\n" + "\n".join(links))
        return "\n\n".join(sections)

    def _aggregate_content_responses(self, state: dict, agent_outputs: list) -> Any:
        for key, agent_name, default_source in (
            ("rag_response", "rag_agent", KNOWLEDGE_BASE),
            ("galaxy_response", "galaxy_agent", GALAXY_PLATFORM),
            ("biogpt_response", "biogpt_agent", "biogpt"),
            ("pubmed_response", "pubmed_agent", "PubMed"),
            ("clinical_trials_response", "clinical_trials_agent", "ClinicalTrials.gov"),
        ):
            resp = state.get(key)
            if resp:
                text_content = resp.get("text", "")
                if text_content:
                    agent_outputs.append({
                        "agent": agent_name,
                        "source": resp.get("source", default_source),
                        "content": text_content,
                    })
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
                agent_outputs.append({
                    "agent": "content_retrieval_agent",
                    "source": ", ".join(content_resp.get("sources", ["external content"])),
                    "content": content_parts,
                })
        return state.get("resource")

    def _aggregate_responses(self, state: dict) -> Dict[str, Any]:
        if state.get("stop_pipeline") and state.get("response", {}).get("text"):
            logger.info("stop_pipeline with pre-built response — skipping aggregation")
            return {"response": state["response"]}

        hyp_resp = state.get("hypothesis_response") or {}
        if hyp_resp:
            hyp_succeeded = (
                isinstance(hyp_resp.get("resource"), dict)
                and hyp_resp["resource"].get("type") == "hypothesis"
            )
            if hyp_succeeded:
                final_text = hyp_resp.get("text", "").rstrip()
                sources_footer = self._build_sources_footer(state)
                if sources_footer:
                    final_text += "\n\n" + sources_footer
                return {
                    "response": {"text": final_text, "json_format": None, "organism": None},
                    "resource": hyp_resp.get("resource"),
                }
            hyp_text = (
                hyp_resp.get("text")
                or "The hypothesis service is not returning any results at the moment. "
                   "There is nothing I can help with directly, but I can search for similar "
                   "clinical trials and published research — please try asking about the topic directly."
            )
            return {"response": {"text": hyp_text, "json_format": None, "organism": None}}

        user_query = state.get("user_query", "")
        logger.info("Aggregating responses from multiple agents with source attribution")

        agent_outputs: List[Dict] = []
        json_format = None
        organism = None

        annotation_resp = state.get("annotation_response")
        needs_confirm = False
        confirmation_text = ""
        if annotation_resp:
            json_format, organism, needs_confirm = self.annotation_graph.aggregate_annotation_response(
                annotation_resp, agent_outputs
            )
            if needs_confirm:
                confirmation_text = annotation_resp.get("text", "")
                # Don't early-return — collect other agent outputs first so they aren't lost

        resource_to_save = self._aggregate_content_responses(state, agent_outputs)

        if json_format and not agent_outputs:
            nodes = json_format.get("nodes", [])
            logger.info("[note-check] node statuses: %s", {n.get("node_id"): n.get("status") for n in nodes})
            logger.info("[note-check] failed nodes: %s", [n.get("node_id") for n in nodes if n.get("status") is False])
            return {
                "response": {
                    "text": self.annotation_graph.build_annotation_text(json_format),
                    "json_format": json_format,
                    "organism": organism,
                }
            }

        if not agent_outputs:
            # No other agents produced output — return confirmation alone (or generic error)
            if needs_confirm:
                return {"response": {"text": confirmation_text, "json_format": None}}
            return {"response": {"text": "I couldn't find any relevant information to answer your query.", "json_format": None}}

        try:
            sources_info = []
            for output in agent_outputs:
                logger.info(
                    "=== [%s] source=%s ===\n%s",
                    output["agent"], output.get("source", "unknown"),
                    str(output.get("content", ""))[:300],
                )
                content = output.get("content", "")
                if isinstance(content, dict):
                    content = str(content)
                content = content.strip() if isinstance(content, str) else ""
                if content:
                    sources_info.append(f"From {output.get('source', 'unknown')}: {content}")

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
                prompt = hypothesis_aggregator_prompt.format(user_query=user_query, combined_text=combined_text)
            else:
                prompt = aggregator_prompt.format(user_query=user_query, combined_text=combined_text, json_note=json_note)

            aggregated_text = self.advanced_llm.generate(prompt)
            logger.info("Aggregated response: %s...", aggregated_text[:100])

            sources_footer = self._build_sources_footer(state)
            if sources_footer:
                aggregated_text = aggregated_text.rstrip() + "\n\n" + sources_footer

            # If annotation also needs confirmation, append it after the synthesized answer
            if needs_confirm and confirmation_text:
                aggregated_text = aggregated_text.rstrip() + "\n\n---\n\n" + confirmation_text

            return {
                "response": {"text": aggregated_text, "json_format": json_format, "organism": organism},
                "resource": resource_to_save,
            }

        except Exception as e:
            logger.error("Error in aggregation: %s", str(e), exc_info=True)
            fallback_parts = []
            for output in agent_outputs:
                content = output.get("content", "")
                if isinstance(content, dict):
                    content = str(content)
                if content:
                    fallback_parts.append(f"**From {output.get('source', 'unknown')}:**\n{content.strip()}")
            fallback_text = "\n\n".join(fallback_parts) if fallback_parts else "Annotation data retrieved."
            return {
                "response": {"text": fallback_text, "json_format": json_format, "organism": organism},
                "resource": resource_to_save,
            }

    def _finalize_response(self, state: dict) -> Dict[str, Any]:
        response = state.get("response", {})
        user_id = state.get("user_id")
        logger.info("Finalizing response for user: %s", user_id)
        logger.info("Response: %s", response)
        if not isinstance(response, dict):
            response = {"text": str(response), "json_format": None}
        response.setdefault("text", "")
        if isinstance(state.get("resource"), dict):
            response["resource"] = state.get("resource")
        emit_to_user(user=user_id, message=response, status="completed")
        return {"response": response}

    def agent(
        self,
        message: str,
        user_id: str,
        token: str,
        content_ids: Optional[List[str]] = None,
        graph_id: Optional[str] = None,
        urls: Optional[List[str]] = None,
        resource: Optional[Any] = None,
    ) -> Dict[str, Any]:
        logger.info(
            "Agent called: message=%s user_id=%s content_ids=%s graph_id=%s urls=%s",
            message, user_id, content_ids, graph_id, urls,
        )
        try:
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "user_query": message,
                "user_id": user_id,
                "token": token,
                "query_types": [],
                "response": {"text": "", "json_format": None},
                "error": "",
                "content_ids": content_ids,
                "graph_id": graph_id,
                "urls": urls,
                "resource": resource,
                "pipeline_details": {},
                "annotation_response": None,
                "rag_response": None,
                "galaxy_response": None,
                "biogpt_response": None,
                "content_retrieval_response": None,
                "hypothesis_response": None,
                "pubmed_response": None,
                "clinical_trials_response": None,
                "stop_pipeline": False,
                "agents_to_run": [],
                "agents_completed": [],
            }

            result = self.pipeline.run(initial_state)

            response = result.get("response", {"text": ""})
            if not isinstance(response, dict):
                response = {"text": str(response), "json_format": None}
            else:
                response.setdefault("text", "")
                response.setdefault("json_format", None)

            response["agents_completed"] = result.get("agents_completed", [])
            logger.info("Agent completed for user: %s", user_id)
            return response

        except Exception as e:
            logger.error("Error in agent processing", exc_info=True)
            error_response = {
                "text": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "json_format": None,
                "agents_completed": [],
            }
            emit_to_user(user=user_id, message=error_response, status="error")
            return error_response

    def _route_to_agent(
        self,
        response: str,
        query: str,
        user_id: str,
        token: str,
        graph_id,
        content_ids,
        urls,
        resource,
    ) -> Dict[str, Any]:
        if "response:" in response:
            final_response = response.split("response:")[1].strip().strip('"')
            self.store.create_history(
                user_id=user_id,
                user_message=query,
                assistant_answer=final_response,
                graph_id_referenced=graph_id,
                content_ids=content_ids,
                urls=urls,
                agents_used=[],
            )
            emit_to_user(user=user_id, message=final_response, status="completed")
            return {"text": final_response}

        if "question:" in response:
            refactored_question = response.split("question:")[1].strip()
            agent_response = self.agent(
                refactored_question, user_id, token,
                content_ids=content_ids, graph_id=graph_id, urls=urls, resource=resource,
            )
            if not isinstance(agent_response, dict):
                agent_response = {"text": str(agent_response), "agents_completed": []}

            resource_data = agent_response.get("resource")
            if isinstance(resource_data, dict) and resource_data.get("type"):
                logger.info("Resource created: %s", resource_data["type"])

            self.store.create_history(
                user_id=user_id,
                user_message=query,
                assistant_answer=agent_response.get("text", str(agent_response)),
                graph_id_referenced=graph_id,
                content_ids=content_ids,
                urls=urls,
                agents_used=agent_response.get("agents_completed", []),
                json_format=agent_response.get("json_format"),
            )
            emit_to_user(user=user_id, message=agent_response, status="completed")
            return agent_response

        logger.error("No response generated from LLM")
        error_msg = "I apologize, but I encountered an error while processing your request."
        self.store.create_history(
            user_id=user_id,
            user_message=query,
            assistant_answer=error_msg,
            graph_id_referenced=graph_id,
            content_ids=content_ids,
            urls=urls,
            agents_used=[],
        )
        emit_to_user(user=user_id, message={"text": error_msg}, status="completed")
        return {"text": error_msg}

    def _handle_repeat_question(
        self,
        query: str,
        history: list,
        user_id: str,
        graph_id,
        content_ids,
        urls,
    ):
        """Return a reflection response when the user re-asks the immediately preceding
        question unchanged. Returns None to fall through and re-run agents when:
        - no history exists
        - the last question differs from current query
        - the previous answer contained failure signals (service down, no results, etc.)
        """
        if not history:
            return None

        last_entry = history[-1]

        # Session gap: if the last exchange is older than 30 minutes the user has
        # effectively started a new conversation (e.g. page reload). Don't reflect.
        SESSION_GAP_SECONDS = 300
        last_time = last_entry.get("context", {}).get("time")
        if last_time:
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - last_time).total_seconds()
            if age > SESSION_GAP_SECONDS:
                logger.info("Repeat question: last exchange %.0fs ago — treating as new session", age)
                return None

        last_q = last_entry.get("question", "").strip().lower()
        if last_q != query.strip().lower():
            return None

        last_a = last_entry.get("context", {}).get("answer", "")
        if not last_a:
            return None

        reflection_marker = "\n\n---\n\nThat was my previous answer to this question."

        # If the stored answer already contains the reflection suffix, the user already
        # saw "did that cover it?" and is asking a third time — they want more. Re-run.
        if reflection_marker in last_a:
            logger.info("Repeat question: user already saw reflection — re-running agents for fresh answer")
            return None

        # Only inspect the main answer body (before any annotation confirmation prompt)
        # because confirmation text after "---" contains "couldn't find" by design.
        main_answer = last_a.split("\n\n---\n\n")[0]
        bad_signals = [
            "couldn't find", "could not find",
            "service is not returning", "no results",
            "unavailable", "failed", "error while processing",
            "i apologize", "no specific entities",
        ]
        if any(s in main_answer.lower() for s in bad_signals):
            logger.info("Repeat question detected but previous answer was poor — re-running agents")
            return None

        logger.info("Repeat question detected with good previous answer — reflecting")
        # Strip any confirmation prompts (appended after "---") — they are stale.
        core_answer = last_a.split("\n\n---\n\n")[0].rstrip()

        # Mention any knowledge graph that was generated, without re-sending it.
        last_json = last_entry.get("context", {}).get("json_format")
        graph_note = ""
        if last_json and last_json.get("nodes"):
            node_labels = []
            for n in last_json["nodes"]:
                props = n.get("properties", {})
                label = (
                    props.get("gene_name")
                    or props.get("term_name")
                    or props.get("name")
                    or n.get("type", "node")
                )
                node_labels.append(label)
            graph_note = (
                "\n\nA knowledge graph was also generated for this query"
                f" with {len(node_labels)} node(s): {', '.join(node_labels)}."
            )

        follow_up = (
            core_answer
            + graph_note
            + reflection_marker
            + " Did that cover what you needed, or would you like me to go deeper on any part?"
        )
        resp = {"text": follow_up, "json_format": None}
        self.store.create_history(
            user_id=user_id,
            user_message=query,
            assistant_answer=follow_up,
            graph_id_referenced=graph_id,
            content_ids=content_ids,
            urls=urls,
            agents_used=[],
        )
        emit_to_user(user=user_id, message=resp, status="completed")
        return resp

    def assistant_response(
        self,
        query: str,
        user_id: str,
        token: str,
        graph_id: Optional[str] = None,
        urls: Optional[List[str]] = None,
        content_ids: Optional[List[str]] = None,
        resource: Optional[Any] = None,
    ) -> Dict[str, Any]:
        try:
            logger.info(
                "assistant_response: query=%s user_id=%s graph_id=%s content_ids=%s urls=%s",
                query, user_id, graph_id, content_ids, urls,
            )

            if self.annotation_graph.has_pending_for(user_id):
                resp = self.annotation_graph.handle_confirmation_response(user_id, query)
                if resp is not None:
                    self.store.create_history(
                        user_id=user_id,
                        user_message=query,
                        assistant_answer=resp.get("text", ""),
                        graph_id_referenced=graph_id,
                        content_ids=content_ids,
                        urls=urls,
                        agents_used=resp.get("agents_completed", []),
                        json_format=resp.get("json_format"),
                    )
                    emit_to_user(user=user_id, message=resp, status="completed")
                    return resp

            # Fetch history once here — used by pending handlers below AND conversation_prompt
            try:
                user_information = self.store.get_context_and_memory(user_id)
                history = []
                memory = []
                for item in user_information:
                    history.append({"question": item["question"], "context": item["context"]})
                    memory.append(item["context"]["memory"])
            except Exception:
                history = []
                memory = []

            logger.info("Histories: %s  Memories: %s", history, memory)

            # Repeat-question detection: if the user asks the exact same question as their
            # immediately preceding one (nothing in between), reflect the previous answer
            # instead of re-running all agents. If the previous answer was poor (service
            # failure, no results), fall through and re-run.
            repeat_resp = self._handle_repeat_question(query, history, user_id, graph_id, content_ids, urls)
            if repeat_resp is not None:
                return repeat_resp

            if self.hypothesis_generation.has_pending_sample_offer_for(user_id):
                resp = self.hypothesis_generation.handle_sample_offer_response(user_id, query, token)
                if resp is not None:
                    self.store.create_history(
                        user_id=user_id,
                        user_message=query,
                        assistant_answer=resp.get("text", ""),
                        graph_id_referenced=graph_id,
                        content_ids=content_ids,
                        urls=urls,
                        agents_used=resp.get("agents_completed", ["hypothesis_agent"]),
                    )
                    emit_to_user(user=user_id, message=resp, status="completed")
                    return resp

            if self.hypothesis_generation.has_pending_tissue_for(user_id):
                resp = self.hypothesis_generation.handle_tissue_selection(user_id, query, token, history=history)
                if resp is not None:
                    self.store.create_history(
                        user_id=user_id,
                        user_message=query,
                        assistant_answer=resp.get("text", ""),
                        graph_id_referenced=graph_id,
                        content_ids=content_ids,
                        urls=urls,
                        agents_used=resp.get("agents_completed", ["hypothesis_agent"]),
                    )
                    emit_to_user(user=user_id, message=resp, status="completed")
                    return resp

            if self.hypothesis_generation.has_pending_go_for(user_id):
                resp = self.hypothesis_generation.handle_go_selection(user_id, query, token, history=history)
                if resp is not None:
                    self.store.create_history(
                        user_id=user_id,
                        user_message=query,
                        assistant_answer=resp.get("text", ""),
                        graph_id_referenced=graph_id,
                        content_ids=content_ids,
                        urls=urls,
                        agents_used=resp.get("agents_completed", ["hypothesis_agent"]),
                    )
                    emit_to_user(user=user_id, message=resp, status="completed")
                    return resp

            if history or any(memory):
                prompt = conversation_prompt.format(
                    memory=memory,
                    query=query,
                    conversation_history=history,
                    graph_id=graph_id or "",
                )
                logger.info("Running conversation_prompt through advanced LLM")
                response = self.advanced_llm.generate(prompt)
                logger.info("Advanced LLM response: %s", response)
            else:
                logger.info("No history/memory — skipping conversation_prompt LLM call")
                response = f"question: {query}"

            emit_to_user(user=user_id, message=ANALYZING_MSG)

            return self._route_to_agent(
                response or "", query, user_id, token, graph_id, content_ids, urls, resource
            )

        except Exception as e:
            logger.error("Error in assistant_response: %s", e, exc_info=True)
            error_msg = "I apologize, but I encountered an error while processing your request."
            try:
                self.store.create_history(
                    user_id=user_id,
                    user_message=query,
                    assistant_answer=error_msg,
                    graph_id_referenced=graph_id,
                    content_ids=content_ids,
                    urls=urls,
                    agents_used=[],
                )
            except Exception as save_error:
                logger.error("Failed to save error history: %s", save_error)
            return {"text": error_msg, "json_format": None}
