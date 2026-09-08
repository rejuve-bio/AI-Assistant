
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage

from app.agents.state import AgentState, ANNOTATION_DB, GALAXY_PLATFORM, KNOWLEDGE_BASE
from app.rag.utils.galaxy_content import GALAXY_COLLECTION, clean_galaxy_text
from app.socket_manager import emit_to_user

logger = logging.getLogger(__name__)


class AgentNodesMixin:
    def _annotation_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle annotation-related queries"""
        query_types = state.get("query_types", [])
        query_type = next((qt for qt in query_types if "annotation" in qt), "annotation_biological")
        
        logger.info(
            f"Annotation agent processing query: {state['user_query']} for user: {state['user_id']}, type: {query_type}"
        )
        
        try:
            if query_type == "annotation_biological":
                emit_to_user(
                    user=state["user_id"], 
                    message="Processing your biological query..."
                )

            pipeline_response = self.annotation_graph.process_annotation_query(
                query=state["user_query"],
                user_id=state["user_id"],
                query_type=query_type,
            )

            logger.info(f"Pipeline response: {pipeline_response}")

            if pipeline_response.get("needs_confirmation"):
                return {
                    "pending_confirmation": {
                        "agent": "annotation",
                        "confirmation_text": pipeline_response.get("confirmation_text", ""),
                        "data": pipeline_response.get("pending") or {},
                    },                    
                    "agents_completed": [],
                    "messages": [AIMessage(content="Annotation needs user confirmation")],
                }

            if pipeline_response.get("success", False):
                summary = pipeline_response.get("summary", "")
                json_format = pipeline_response.get("json_format", None)
                validation_report = pipeline_response.get("validation_report", {})
                organism = pipeline_response.get("organism", "human")

                response_dict = {
                    "text": summary if summary else "",
                    "json_format": json_format,
                    "validation_report": validation_report,
                    "organism": organism,
                    "source": ANNOTATION_DB
                }

                return {
                    "annotation_response": response_dict,
                    "agents_completed": ["annotation_agent"],
                    "messages": [AIMessage(content="Annotation processing completed")]
                }

            else:
                error_msg = pipeline_response.get("error", "Unknown error")
                logger.error(f"Annotation pipeline failed: {error_msg}")
                return {
                    "annotation_response": {
                        "text": f"Error: {error_msg}", 
                        "json_format": None,
                        "source": ANNOTATION_DB
                    },
                    "agents_completed": ["annotation_agent"],
                    "error": error_msg,
                }

        except Exception as e:
            logger.error("Unexpected error in annotation agent", exc_info=True)
            return {
                "annotation_response": {
                    "text": f"Error: {str(e)}",
                    "json_format": None,
                    "source": ANNOTATION_DB
                },
                "agents_completed": ["annotation_agent"],
                "error": str(e),
            }


    def _hypothesis_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle hypothesis generation queries"""
        logger.info(
            f"Hypothesis agent processing query: {state['user_query']} for user: {state['user_id']}"
        )
        try:
            emit_to_user(user=state["user_id"], message="Generating hypothesis...")
            response = self.hypothesis_generation.generate_hypothesis(
                token=state["token"],
                user_query=state["user_query"],
                user_id=state["user_id"],
            )

            hypothesis_text = response.get("text", "")
            # A real hypothesis always returns resource: {id, type, graph} — all fallback/failure paths omit it
            succeeded = isinstance(response.get("resource"), dict) and response["resource"].get("type") == "hypothesis"

            state_update = {
                "hypothesis_response": response,
                "messages": [AIMessage(content=f"Hypothesis generated: {hypothesis_text}")],
                "agents_completed": ["hypothesis_agent"],
            }

            if succeeded:
                current_agents = state.get("agents_to_run", [])
                extra = [a for a in ("clinical_trials_agent", "pubmed_agent") if a not in current_agents]
                if extra:
                    logger.info(f"Hypothesis succeeded — injecting literature agents: {extra}")
                    state_update["agents_to_run"] = current_agents + extra

            return state_update

        except Exception as e:
            logger.error("Error in hypothesis agent", exc_info=True)
            return {
                "hypothesis_response": {
                    "text": "The hypothesis service is not returning any results at the moment. There is nothing I can help with for this request.",
                    "resource": None,
                    "status": "failed",
                    "reason": str(e),
                },
                "stop_pipeline": True,
                "error": str(e),
                "messages": [AIMessage(content=f"Error in hypothesis generation: {str(e)}")],
                "agents_completed": ["hypothesis_agent"],
            }


    def _rag_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle general information queries"""
        logger.info(
            f"RAG agent processing query: {state['user_query']} for user: {state['user_id']}"
        )

        try:
            emit_to_user(user=state["user_id"], message="Retrieving information...")
            
            response = self.rag.get_result_from_rag(
                state["user_query"],
                state["user_id"],
                content_ids=state.get("content_ids"),
            )

            # Normalize response to dict with text key
            if response and isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response) if response else ""
            logger.debug(f"RAG response: {response_text}")

            # No useful results → inject PubMed as fallback
            if self._rag_has_no_results(response_text):
                current_agents = state.get("agents_to_run", [])
                if "pubmed_agent" not in current_agents:
                    logger.info("RAG found no results — injecting pubmed_agent as fallback")
                    emit_to_user(user=state["user_id"], message="Nothing found in knowledge base, searching PubMed...")
                    return {
                        "rag_response": {"text": response_text, "json_format": None, "source": KNOWLEDGE_BASE},
                        "agents_to_run": current_agents + ["pubmed_agent"],
                        "agents_completed": ["rag_agent"],
                        "messages": [AIMessage(content="RAG found no results — triggering PubMed fallback")],
                    }

            return {
                "rag_response": {
                    "text": response_text,
                    "json_format": None,
                    "source": KNOWLEDGE_BASE
                },
                "agents_completed": ["rag_agent"],
                "messages": [AIMessage(content="RAG query processed")],
            }
            
        except Exception as e:
            logger.error("Error in RAG agent", exc_info=True)
            return {
                "rag_response": {
                    "text": f"Error: {str(e)}", 
                    "json_format": None,
                    "source": KNOWLEDGE_BASE
                },
                "agents_completed": ["rag_agent"],
                "error": str(e),
            }


    def _galaxy_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle Galaxy tools and workflows queries"""
        logger.info(
            f"Galaxy agent processing query: {state['user_query']} for user: {state['user_id']}"
        )
        
        try:
            emit_to_user(
                user=state["user_id"], 
                message="Retrieving Galaxy tools information..."
            )
            
            response = self.galaxy_handler.get_galaxy_info(
                state["user_query"], 
                state["user_id"], 
                state["token"]
            )

            # Normalize response
            if isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response) if response else "No Galaxy information found"
            logger.debug(f"Galaxy response: {response_text}")
            return {
                "galaxy_response": {
                    "text": response_text, 
                    "json_format": None,
                    "source": GALAXY_PLATFORM
                },
                "agents_completed": ["galaxy_agent"],
                "messages": [AIMessage(content="Galaxy query processed")],
            }
            
        except Exception as e:
            logger.error("Error in galaxy agent", exc_info=True)
            return {
                "galaxy_response": {
                    "text": f"Error: {str(e)}", 
                    "json_format": None,
                    "source": GALAXY_PLATFORM
                },
                "agents_completed": ["galaxy_agent"],
                "error": str(e),
            }


    def _retrieve_from_graph(self, query, user_id, graph_id, token, resource, content_parts, sources):
        """Returns (early_return, entity_found) — early_return is a state update dict to
        short-circuit the pipeline on graph-fetch failure, or None to continue normally.
        entity_found is True/False/None (unknown) per whether the graph actually covered
        the query, used to decide whether a fresh annotation_agent run is still needed."""
        logger.info(f"Retrieving graph summary for graph_id: {graph_id}")
        graph_summary = self.answer_from_graph_summaries(
            query=query,
            user_id=user_id,
            graph_id=graph_id,
            token=token,
            resource=resource
        )
        if not graph_summary:
            return None, None
        entity_found = graph_summary.get("entity_found") if isinstance(graph_summary, dict) else None
        graph_text = graph_summary.get("text", str(graph_summary)) if isinstance(graph_summary, dict) else str(graph_summary)
        if graph_text and not graph_text.startswith("Failed to contact") and not graph_text.startswith("Error"):
            content_parts.append({"source": f"graph:{graph_id}", "content": graph_text})
            sources.append(f"graph:{graph_id}")
            return None, entity_found
        if graph_text:
            logger.warning(f"Graph fetch failed for {graph_id}: {graph_text}")
            last_topic = None
            try:
                history = self.store.get_context_and_memory(user_id)
                for item in reversed(history):
                    agents_used = item.get("context", {}).get("agents_used", [])
                    if "annotation_agent" in agents_used:
                        last_topic = item.get("question")
                        break
            except Exception:
                pass
            if last_topic:
                confirmation_text = (
                    f"I couldn't find the graph you referenced (ID: `{graph_id}`). "
                    f"Did you mean to ask about your previous annotation: *\"{last_topic}\"*? "
                    f"Or would you like to ask a different question?"
                )
            else:
                confirmation_text = (
                    f"I couldn't find the graph you referenced (ID: `{graph_id}`). "
                    f"Please check that the graph exists, or let me know what you'd like to explore."
                )
            return {
                "content_retrieval_response": {
                    "text": confirmation_text,
                    "json_format": None,
                    "sources": [],
                    "status": "needs_input",
                    "reason": "graph_not_found",
                },
                "agents_completed": ["content_retrieval_agent"],
                "stop_pipeline": True,
            }, None
        return None, entity_found


    def _retrieve_from_urls(self, query, urls, content_parts, sources):
        """Answer from URLs supplied by the client.

        Goes straight to RAG: Galaxy's handler is only needed for MCP. New URLs
        are scraped and stored on first use, then answered from storage.
        """
        logger.info(f"Retrieving URL content for: {urls}")
        self.rag.save_url_content(
            urls,
            collection_name=GALAXY_COLLECTION,
            cleaner=clean_galaxy_text,
            summarize=True,
            include_tables=True,
        )
        urls_text = self.rag.query_url_content(query, urls, GALAXY_COLLECTION)
        if urls_text:
            for file in (urls if isinstance(urls, list) else [urls]):
                content_parts.append({"source": f"file:{file}", "content": urls_text})
                sources.append(f"file:{file}")


    def _retrieve_from_rag(self, query, user_id, content_ids, content_parts, sources):
        logger.info(f"Retrieving RAG content for content_ids: {content_ids}")
        rag_content = self.rag.get_result_from_rag(query, user_id, content_ids)
        if rag_content:
            rag_text = rag_content.get("text", str(rag_content)) if isinstance(rag_content, dict) else str(rag_content)
            resources = rag_content.get("resource", {})
            content_parts.append({
                "source": f"content IDs: {', '.join(content_ids)}",
                "content": rag_text,
                "resource": resources
            })
            sources.append(f"content IDs: {', '.join(content_ids)}")


    def _content_retrieval_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve relevant content from multiple sources with source attribution
        """
        query = state.get("user_query")
        user_id = state.get("user_id")
        token = state.get("token")
        graph_id = state.get("graph_id")
        urls = state.get("urls")
        content_ids = state.get("content_ids")
        resource = state.get("resource")

        logger.info(f"ContentRetrievalAgent called for user: {user_id}")
        emit_to_user(user=user_id, message="Retrieving relevant content...")

        content_parts = []
        sources = []
        graph_covers_query = False

        try:
            if graph_id:
                early_return, entity_found = self._retrieve_from_graph(query, user_id, graph_id, token, resource, content_parts, sources)
                if early_return is not None:
                    return early_return

                if entity_found is True and "annotation_agent" in state.get("agents_to_run", []):
                    graph_covers_query = True

            if urls:
                self._retrieve_from_urls(query, urls, content_parts, sources)

            if content_ids:
                self._retrieve_from_rag(query, user_id, content_ids, content_parts, sources)

            response_dict = {
                "text": content_parts,
                "json_format": None,
                "sources": sources
            }
            logger.info(f"Content retrieval response prepared with {len(content_parts)} parts. response is {response_dict}")
            state_update = {
                "content_retrieval_response": response_dict,
                "agents_completed": ["content_retrieval_agent"],
                "messages": [AIMessage(content="Content retrieval completed")]
            }
            if graph_covers_query:
                logger.info("Existing attached graph already covers the query — skipping redundant annotation_agent run")
                current_agents = state.get("agents_to_run", [])
                state_update["agents_to_run"] = [a for a in current_agents if a != "annotation_agent"]
            return state_update

        except Exception as e:
            logger.error(f"Error in ContentRetrievalAgent: {str(e)}", exc_info=True)
            return {
                "content_retrieval_response": {
                    "text": [],
                    "json_format": None,
                    "sources": []
                },
                "agents_completed": ["content_retrieval_agent"],
                "error": str(e),
            }


    def _biogpt_agent(self, state: AgentState) -> dict:
        try:
            emit_to_user(user=state["user_id"], message="Analyzing biomedical information...")
            response = self.biogpt.generate_answer(state["user_query"])
            logger.info(f"BioGPT response: {response}")
            return {
                "biogpt_response": {
                    "text": response,
                    "source": "BioGPT"
                },
                "agents_completed": ["biogpt_agent"],
                "messages": [AIMessage(content="BioGPT query processed")]
            }
        except Exception as e:
            logger.error(f"Error in biogpt agent: {str(e)}", exc_info=True)
            return {
                "biogpt_response": {
                    "text": None,
                    "json_format": None,
                    "source": "BioGPT"
                },
                "agents_completed": ["biogpt_agent"],
                "error": str(e)
            }

    _NO_RESULT_PHRASES = (
        "couldn't find", "could not find", "no relevant", "no information",
        "no results", "not found", "no documents", "unable to find",
        "no data", "i don't have information", "i do not have",
        "no specific", "no details",
    )


    def _rag_has_no_results(self, text: str) -> bool:
        t = text.lower().strip()
        return len(t) < 120 or any(p in t for p in self._NO_RESULT_PHRASES)


    def _extract_search_term(self, user_query: str, context: str = "") -> str:
        """Distil a question (and optional context) into a concise API-friendly search term."""
        context_line = f"\nAdditional context: {context[:500]}" if context else ""
        prompt = (
            "Extract a short, keyword-based search term (3-7 words) suitable for searching "
            "PubMed or ClinicalTrials.gov. Focus on the biological topic, gene, drug, or condition. "
            "Do NOT include words like: clinical trials, studies, papers, literature, search, find, pubmed, research. "
            "Do NOT use only a variant rs number — expand to the gene name and condition it is associated with. "
            "Return ONLY the search term, no explanation, no punctuation.\n\n"
            f"User question: {user_query}{context_line}\n\nSearch term:"
        )
        try:
            term = self.basic_llm.generate(prompt).strip().strip('"').strip("'")
            logger.info(f"Extracted search term: '{term}'")
            return term if term else user_query
        except Exception:
            return user_query


    def _pubmed_agent(self, state: AgentState) -> Dict[str, Any]:
        from app.rag.literature import search_pubmed
        user_id = state["user_id"]
        hypothesis = state.get("hypothesis_response") or {}
        context = hypothesis.get("text", "")
        search_term = self._extract_search_term(state["user_query"], context=context)
        logger.info(f"PubMed agent searching for: {search_term}")
        try:
            emit_to_user(user=user_id, message="Searching PubMed literature...")
            result = search_pubmed(search_term, max_results=8)
            papers = result.get("papers", [])
            if not papers:
                text = "No relevant publications found in PubMed for this query."
            else:
                lines = [f"Found {len(papers)} relevant paper(s) from PubMed:\n"]
                for p in papers:
                    authors = ", ".join(p.get("authors", [])) or "Unknown authors"
                    lines.append(
                        f"- **{p.get('title', 'No title')}** ({p.get('year', '')}) — {authors}\n"
                        f"  {p.get('abstract', '')}\n"
                        f"  URL: {p.get('url', '')}"
                    )
                text = "\n".join(lines)
            return {
                "pubmed_response": {"text": text, "source": "PubMed", "items": papers},
                "agents_completed": ["pubmed_agent"],
                "messages": [AIMessage(content="PubMed search completed")],
            }
        except Exception as e:
            logger.error(f"PubMed agent error: {e}", exc_info=True)
            return {
                "pubmed_response": {"text": f"PubMed search unavailable: {str(e)}", "source": "PubMed", "items": []},
                "agents_completed": ["pubmed_agent"],
            }


    def _clinical_trials_agent(self, state: AgentState) -> Dict[str, Any]:
        from app.rag.literature import search_clinical_trials
        user_id = state["user_id"]
        hypothesis = state.get("hypothesis_response") or {}
        context = hypothesis.get("text", "")
        search_term = self._extract_search_term(state["user_query"], context=context)
        logger.info(f"ClinicalTrials agent searching for: {search_term}")
        try:
            emit_to_user(user=user_id, message="Searching ClinicalTrials.gov...")
            result = search_clinical_trials(search_term, status="RECRUITING", max_results=5)
            trials = result.get("trials", [])
            if not trials:
                result = search_clinical_trials(search_term, status="", max_results=5)
                trials = result.get("trials", [])
            if not trials:
                text = "No clinical trials found for this query on ClinicalTrials.gov."
            else:
                lines = [f"Found {len(trials)} clinical trial(s) on ClinicalTrials.gov:\n"]
                for t in trials:
                    phase = ", ".join(t.get("phase", [])) or "N/A"
                    conditions = ", ".join(t.get("conditions", [])) or "N/A"
                    interventions = ", ".join(t.get("interventions", [])) or "N/A"
                    lines.append(
                        f"- **{t.get('title', 'No title')}** ({t.get('nct_id', '')})\n"
                        f"  Phase: {phase} | Status: {t.get('status', '')} | Started: {t.get('start_date', 'N/A')}\n"
                        f"  Conditions: {conditions}\n"
                        f"  Interventions: {interventions}\n"
                        f"  URL: {t.get('url', '')}"
                    )
                text = "\n".join(lines)
            return {
                "clinical_trials_response": {"text": text, "source": "ClinicalTrials.gov", "items": trials},
                "agents_completed": ["clinical_trials_agent"],
                "messages": [AIMessage(content="ClinicalTrials search completed")],
            }
        except Exception as e:
            logger.error(f"ClinicalTrials agent error: {e}", exc_info=True)
            return {
                "clinical_trials_response": {"text": f"ClinicalTrials search unavailable: {str(e)}", "source": "ClinicalTrials.gov",  "items": []},
                "agents_completed": ["clinical_trials_agent"],
            }

    def answer_from_graph_summaries(self, query, user_id, resource, token, graph_id):
        """Legacy method for backward compatibility"""
        logger.info(
            f"Answer from graph summaries called with query: {query}, user_id: {user_id}, "
            f"resource: {resource}, graph_id: {graph_id}"
        )
        
        try:
            entity_found = None
            if resource == "annotation":
                summary_result = self.graph_summarizer.summary(
                    token=token, graph_id=graph_id, user_query=query
                )
                summary_text = summary_result.get('text', '') if isinstance(summary_result, dict) else summary_result
                if isinstance(summary_result, dict):
                    entity_found = summary_result.get('entity_found')
                emit_to_user(user=user_id, message=ANALYZING_MSG)

            elif resource == "hypothesis":
                summary_result = self.hypothesis_generation.get_by_hypothesis_id(
                    token, graph_id, user_id, query
                )
                summary_text = summary_result.get('text', '') if isinstance(summary_result, dict) else summary_result
                emit_to_user(user=user_id, message=ANALYZING_MSG)
            else:
                return "Invalid resource type specified."

            return {"text": summary_text, "json_format": None, "entity_found": entity_found}
            
        except Exception as e:
            logger.error("Error in answer_from_graph_summaries", exc_info=True)
            return {
                "text": f"Error processing query: {str(e)}",
                "json_format": None
            }