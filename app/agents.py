from .llm_handle.llm_models import (
    LLMInterface,
    OpenAIModel,
    get_llm_model,
    openai_embedding_model,
)
from .prompts.classifier_prompt import (
    aggeregator_prompt,
    VALIDATION_PROMPT,
    PLANNER_PROMPT,
    agent_descriptions
)
from .prompts.dependency_prompts import DEPENDENCY_SUMMARIZATION_PROMPT
from .annotation_graph.annotated_graph import Graph
from .annotation_graph.schema_handler import SchemaHandler
from .rag.rag import RAG
from .rag.utils.web_search import SimpleWebSearch
from .prompts.conversation_handler import conversation_prompt
from .summarizer import Graph_Summarizer
from .hypothesis_generation.hypothesis import HypothesisGeneration
from .socket_manager import emit_to_user
from .Galaxy_integration.galaxy import GalaxyHandler
from .biogpt_agent.biogpt import BioGPTAgent
from .utils import RichLogger
from typing import TypedDict, List, Annotated, Any, Dict, Optional
from flask_socketio import emit
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import json
import operator
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    user_id: str
    token: str
    query_types: List[str]
    response: Dict[str, Any]
    error: str
    content_ids: Optional[List[str]]
    graph_id: Optional[str]
    urls: Optional[List[str]]
    resource: Optional[str]
    pipeline_details: Dict[str, Any]
    annotation_response: Optional[Dict[str, Any]]
    rag_response: Optional[Dict[str, Any]]
    galaxy_response: Optional[Dict[str, Any]]
    content_retrieval_response: Optional[Dict[str, Any]]
    biogpt_response: Optional[Dict[str, Any]]
    agents_to_run: List[str]
    agents_completed: Annotated[List[str], operator.add]
    suggested_questions: Optional[List[str]]
    plan: Optional[List[Dict[str, Any]]]
    current_step_index: int
    step_input: Optional[str]
    step_outputs: Dict[int, str]
    # --- Grouped execution fields ---
    execution_groups: Optional[List[Dict[str, Any]]]
    current_group_index: int
    current_step_in_group: int




class AgentManager:
    def __init__(
        self,
        advanced_llm,
        basic_llm,
        schema_handler,
        qdrant_client=None,
        embedding_model=None,
        mongo_db_manager=None,
    ) -> None:
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.store = mongo_db_manager
        self.embedding_model = embedding_model
        
        # Initialize sub-agents/handlers
        self.annotation_graph = Graph(advanced_llm, schema_handler)
        self.graph_summarizer = Graph_Summarizer(self.advanced_llm)
        self.rag = RAG(llm=advanced_llm, qdrant_client=qdrant_client)
        self.hypothesis_generation = HypothesisGeneration(advanced_llm)
        self.galaxy_handler = GalaxyHandler(advanced_llm, qdrant_client, embedding_model)
        self.biogpt = BioGPTAgent(llm=advanced_llm)

        logger.info(f"AgentManager initialized with advanced_llm: {type(self.advanced_llm).__name__}")
        logger.info(f"Galaxy handler initialized: {type(self.galaxy_handler).__name__}")

    def get_content_summaries(self, user_id, content_ids=None):
        """Get summaries for all content types (PDF and web)"""
        content_summaries = []
        all_content = self.store.get_user_content_files(user_id)

        if content_ids:
            filtered_content = [
                content
                for content in all_content
                if content.get("content_id") in content_ids
            ]
        else:
            filtered_content = all_content

        for content in filtered_content:
            if content.get("content_type") == "pdf":
                content_summaries.append(
                    {
                        "content_id": content.get("content_id"),
                        "content_type": "pdf",
                        "filename": content.get("filename"),
                        "summary": content.get("summary") or "",
                    }
                )
            elif content.get("content_type") == "web":
                content_summaries.append(
                    {
                        "content_id": content.get("content_id"),
                        "content_type": "web",
                        "url": content.get("url"),
                        "title": content.get("title"),
                        "summary": content.get("summary") or "",
                    }
                )

        return content_summaries

    def classify_query(self, state: AgentState) -> Dict[str, Any]:
        """
        Two-step classification:
        1. VALIDATION: Check if query is relevant.
        2. PLANNING: If relevant, create a grouped execution plan (Sequential or Parallel).
        """
        query = state["user_query"]
        user_id = state["user_id"]
        content_ids = state.get("content_ids")
        content_summaries = self.get_content_summaries(user_id, content_ids)
        
        logger.info(f"Classifying query: {query}")

        # --- STEP 1: VALIDATION ---
        validation_prompt = VALIDATION_PROMPT.format(
            query=query,
            agent_descriptions=agent_descriptions
        )
        
        try:
            validation_resp = self.advanced_llm.generate(validation_prompt)
            if isinstance(validation_resp, str):
                validation_resp_str = validation_resp.replace("```json", "").replace("```", "").strip()
                validation_result = json.loads(validation_resp_str)
            else:
                validation_result = validation_resp
        except Exception as e:
            logger.error(f"Validation parsing error: {e}")
            validation_result = {"is_valid": True}

        if not validation_result.get("is_valid", True):
            refusal = validation_result.get("refusal_message", "I can only help with biological queries.")
            logger.info(f"Query rejected: {refusal}")
            return {
                "response": {"text": refusal, "json_format": None},
                "agents_to_run": [],
                "plan": [],
                "execution_groups": [],
                "messages": [AIMessage(content=refusal)]
            }

        # --- STEP 2: PLANNING ---
        planner_prompt_text = PLANNER_PROMPT.format(
            query=query,
            agent_descriptions=agent_descriptions,
            content_summaries=content_summaries
        )

        try:
            plan_resp = self.advanced_llm.generate(planner_prompt_text)
            if isinstance(plan_resp, str):
                plan_resp_str = plan_resp.replace("```json", "").replace("```", "").strip()
                plan_result = json.loads(plan_resp_str)
            else:
                plan_result = plan_resp
            
            execution_groups = plan_result.get("execution_groups", [])
            
            if not execution_groups and plan_result.get("steps"):
                old_steps = plan_result["steps"]
                logger.info("Backward compat: wrapping old 'steps' format into a sequential group")
                execution_groups = [{
                    "group_id": 1,
                    "mode": "sequential",
                    "steps": old_steps
                }]

        except Exception as e:
            logger.error(f"Planning parsing error: {e}")
            execution_groups = [{
                "group_id": 1,
                "mode": "sequential",
                "steps": [{"agent": "rag_agent", "input": query, "id": 1, "dependency": None}]
            }]

        all_steps = []
        for group in execution_groups:
            all_steps.extend(group.get("steps", []))
        
        agents_to_run = [step["agent"] for step in all_steps]
        
        if not all_steps:
            execution_groups = [{
                "group_id": 1,
                "mode": "sequential",
                "steps": [{"agent": "rag_agent", "input": query, "id": 1, "dependency": None}]
            }]
            all_steps = execution_groups[0]["steps"]
            agents_to_run = ["rag_agent"]

        logger.info(f"Generated Plan with {len(execution_groups)} execution groups: {execution_groups}")
        
        first_group = execution_groups[0]
        first_step = first_group["steps"][0]
        first_step_input = first_step.get("input", query)

        return {
            "plan": all_steps,
            "execution_groups": execution_groups,
            "current_group_index": 0,
            "current_step_in_group": 0,
            "current_step_index": 0,
            "step_input": first_step_input,
            "step_outputs": {},
            "agents_to_run": agents_to_run,
            "messages": [HumanMessage(content=f"Plan generated with {len(execution_groups)} groups, {len(all_steps)} total steps")]
        }


    
    
    
    
    
    
    def annotation_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle annotation-related queries"""
        query_types = state.get("query_types", [])
        query_type = next((qt for qt in query_types if "annotation" in qt), "annotation_biological")
        
        user_query = state.get("step_input") or state["user_query"]
        
        logger.info(
            f"Annotation agent processing query: {user_query} for user: {state['user_id']}, type: {query_type}"
        )
        
        try:
            if query_type == "annotation_biological":
                emit_to_user(
                    user=state["user_id"], 
                    message="Processing your biological query..."
                )
            elif query_type == "annotation_general":
                emit_to_user(
                    user=state["user_id"], 
                    message="Analyzing database information..."
                )

            pipeline_response = self.annotation_graph.process_annotation_query(
                query=user_query,
                user_id=state["user_id"],
                query_type=query_type,
            )
            
            logger.info(f"Pipeline response: {pipeline_response}")
            
            if pipeline_response.get("success", False):
                summary = pipeline_response.get("summary", "")
                json_format = pipeline_response.get("json_format", None)

                response_dict = {
                    "text": summary if summary else "",
                    "json_format": json_format,
                    "source": "annotation database"
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
                        "source": "annotation database"
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
                    "source": "annotation database"
                },
                "agents_completed": ["annotation_agent"],
                "error": str(e),
            }

    def hypothesis_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle hypothesis generation queries"""
        logger.info(
            f"Hypothesis agent processing query: {state['user_query']} for user: {state['user_id']}"
        )
        try:
            return {"text": "Hypothesis generation agent is under development.",}
        except Exception as e:
            logger.error("Error in hypothesis agent", exc_info=True)
            return {
                "response": f"Error generating hypothesis: {str(e)}",
                "error": str(e),
                "messages": [
                    AIMessage(content=f"Error in hypothesis generation: {str(e)}")
                ],
            }

    def rag_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle general information queries"""
        user_query = state.get("step_input") or state["user_query"]
        logger.info(
            f"RAG agent processing query: {user_query} for user: {state['user_id']}"
        )

        try:
            emit_to_user(user=state["user_id"], message="Retrieving information...")
            
            response = self.rag.get_result_from_rag(
                user_query,
                state["user_id"],
                content_ids=state.get("content_ids"),
            )

            # Normalize response to dict with text key
            if response and isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response) if response else "No response generated"
            logger.info(f"RAG response: {response_text}")
            return {
                "rag_response": {
                    "text": response_text, 
                    "json_format": None,
                    "source": "knowledge base"
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
                    "source": "knowledge base"
                },
                "agents_completed": ["rag_agent"],
                "error": str(e),
            }

    def galaxy_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle Galaxy tools and workflows queries"""
        user_query = state.get("step_input") or state["user_query"]
        logger.info(
            f"Galaxy agent processing query: {user_query} for user: {state['user_id']}"
        )
        
        try:
            emit_to_user(
                user=state["user_id"], 
                message="Retrieving Galaxy tools information..."
            )
            
            response = self.galaxy_handler.get_galaxy_info(
                user_query, 
                state["user_id"], 
                state["token"]
            )

            # Normalize response
            if isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response) if response else "No Galaxy information found"
            logger.info(f"Galaxy response: {response_text}")
            return {
                "galaxy_response": {
                    "text": response_text, 
                    "json_format": None,
                    "source": "Galaxy platform"
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
                    "source": "Galaxy platform"
                },
                "agents_completed": ["galaxy_agent"],
                "error": str(e),
            }

    def content_retrieval_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve relevant content from multiple sources with source attribution
        """
        query = state.get("step_input") or state.get("user_query")
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

        try:
            # Graph summary
            if graph_id:
                logger.info(f"Retrieving graph summary for graph_id: {graph_id}")
                graph_summary = self.answer_from_graph_summaries(
                    query=query, 
                    user_id=user_id,
                    graph_id=graph_id, 
                    token=token, 
                    resource=resource
                )
                if graph_summary:
                    graph_text = graph_summary.get("text", str(graph_summary)) if isinstance(graph_summary, dict) else str(graph_summary)
                    content_parts.append({
                        "source": f"graph:{graph_id}",
                        "content": graph_text
                    })
                    sources.append(f"graph:{graph_id}")

            # Galaxy urls
            if urls:
                logger.info(f"Retrieving Galaxy urls for user: {user_id}")
                urls_response = self.galaxy_handler.get_galaxy_info(
                    query=query, user_id=user_id, token=token,urls=urls
                )
                if urls_response:
                    urls_text = urls_response.get("text", str(urls_response)) if isinstance(urls_response, dict) else str(urls_response)
                    for file in (urls if isinstance(urls, list) else [urls]):
                        content_parts.append({
                            "source": f"file:{file}",
                            "content": urls_text
                        })
                        sources.append(f"file:{file}")

            # RAG content
            if content_ids:
                logger.info(f"Retrieving RAG content for content_ids: {content_ids}")
                rag_content = self.rag.get_result_from_rag(query, user_id, content_ids)
                if rag_content:
                    rag_text = rag_content.get("text", str(rag_content)) if isinstance(rag_content, dict) else str(rag_content)
                    resources = rag_content.get("resource",{})
                    content_parts.append({
                        "source": f"content IDs: {', '.join(content_ids)}",
                        "content": rag_text,
                        "resource": resources
                    })
                    sources.append(f"content IDs: {', '.join(content_ids)}")

            # Build response with source attribution
            if content_parts:
                response_dict = {
                    "text": content_parts,
                    "json_format": None,
                    "sources": sources
                }
            else:
                response_dict = {
                    "text": [],
                    "json_format": None,
                    "sources": []
                }
            logger.info(f"Content retrieval response prepared with {len(content_parts)} parts. response is {response_dict}" )
            return {
                "content_retrieval_response": response_dict,
                "agents_completed": ["content_retrieval_agent"],
                "messages": [AIMessage(content="Content retrieval completed")]
            }

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

    def biogpt_agent(self, state: AgentState) -> dict:
        try:
            user_query = state.get("step_input") or state["user_query"]
            emit_to_user(user=state["user_id"], message="Analyzing biomedical information...")
            response = self.biogpt.generate_answer(user_query)
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

    def aggregate_responses(self, state: AgentState) -> Dict[str, Any]:
        """
        Aggregate responses from all agents with source attribution.
        Ensures that text content is combined coherently and structured JSON data (json_format)
        is always included when available.
        """
        user_query = state.get("user_query", "")
        logger.info("Aggregating responses from multiple agents with source attribution")

        agent_outputs = []
        json_format = None

        # ---------------- Annotation Agent ----------------
        annotation_resp = state.get("annotation_response")
        if annotation_resp:
            text_content = annotation_resp.get("text") or annotation_resp.get("summary") or ""
            if text_content:
                agent_outputs.append({
                    "agent": "annotation_agent",
                    "source": annotation_resp.get("source", "annotation database"),
                    "content": text_content
                })

            # Capture JSON structured data
            json_format = annotation_resp.get("json_format")

            # Add placeholder if only JSON exists
            if json_format and not text_content:
                agent_outputs.append({
                    "agent": "annotation_agent",
                    "source": annotation_resp.get("source", "annotation database"),
                    "content": "Annotation visualization structure format is created successfully (see structured data)."
                })

        # ---------------- RAG Agent ----------------
        rag_resp = state.get("rag_response")
        if rag_resp:
            text_content = rag_resp.get("text", "")
            if text_content:
                agent_outputs.append({
                    "agent": "rag_agent",
                    "source": rag_resp.get("source", "knowledge base"),
                    "content": text_content
                })

        # ---------------- Galaxy Agent ----------------
        galaxy_resp = state.get("galaxy_response")
        if galaxy_resp:
            text_content = galaxy_resp.get("text", "")
            if text_content:
                agent_outputs.append({
                    "agent": "galaxy_agent",
                    "source": galaxy_resp.get("source", "Galaxy platform"),
                    "content": text_content
                })

        biogpt_resp = state.get("biogpt_response")
        if biogpt_resp:
            text_content = biogpt_resp.get("text", "")
            if text_content:
                agent_outputs.append({
                    "agent": "biogpt_agent",
                    "source": biogpt_resp.get("source", "biogpt"),
                    "content": text_content
                })

        # ---------------- Content Retrieval Agent ----------------
        content_resp = state.get("content_retrieval_response")
        if content_resp:
            content_parts = content_resp.get("text", [])
            if isinstance(content_parts, list):
                for part in content_parts:
                    if isinstance(part, dict) and part.get("content"):
                        agent_outputs.append({
                            "agent": "content_retrieval_agent",
                            "source": part.get("source", "external content"),
                            "content": part["content"]
                        })
            elif isinstance(content_parts, str) and content_parts:
                sources = content_resp.get("sources", ["external content"])
                agent_outputs.append({
                    "agent": "content_retrieval_agent",
                    "source": ", ".join(sources),
                    "content": content_parts
                })

        # ---------------- Handle Empty Outputs ----------------
        if not agent_outputs and json_format:
            return {
                "response": {
                    "text": "I found the requested annotation data in the database.",
                    "json_format": json_format
                }
            }

        if not agent_outputs:
            return {
                "response": {
                    "text": "I couldn't find any relevant information to answer your query.",
                    "json_format": None
                }
            }

        # ---------------- LLM Aggregation ----------------
        try:
            sources_info = []
            for output in agent_outputs:
                content = output.get("content", "")
                # Handle if content is a dict (convert to string)
                if isinstance(content, dict):
                    content = str(content)
                content = content.strip() if isinstance(content, str) else ""
                if content:
                    sources_info.append(f"From {output.get('source', 'unknown')}: {content}")
           
            combined_text = "\\n\\n".join(sources_info)

            # Include json_format note if present
            json_note = ""
            if json_format:
                json_note = "\\n\\nNote: Structured annotation data is also available for this query."

            # Build execution context from plan for better aggregation
            plan = state.get("plan", [])
            step_outputs = state.get("step_outputs", {})
            execution_context = ""
            
            if plan and len(plan) > 1:  # Only add if multi-step plan
                execution_context = "\n\nExecution Flow:\n"
                for idx, step in enumerate(plan, 1):
                    agent = step.get("agent", "unknown").replace("_", " ").title()
                    step_id = step.get("id")
                    
                    # Check if step was executed
                    if step_id in step_outputs:
                        output_preview = step_outputs[step_id]
                        if output_preview.startswith("FAILED:"):
                            status = "✗ failed"
                        else:
                            status = "✓ completed"
                    else:
                        status = "⊘ skipped"
                    
                    execution_context += f"{idx}. {agent}: {status}\n"

            # Check if we have any actual content
            if not combined_text.strip():
                logger.warning("Aggregator received no content from agents.")
                fallback_msg = "I could not retrieve any relevant information from the agents to answer your query. Please check if the documents are uploaded or if the query is clear."
                
                # Check for execution context failure clues
                if "failed" in execution_context.lower():
                    fallback_msg += "\n\n(Internal Note: Some internal steps failed during execution.)"
                
                return {
                    "response": {
                        "text": fallback_msg,
                        "json_format": json_format
                    }
                }

            prompt = aggeregator_prompt.format(
                user_query=user_query, 
                combined_responses=combined_text, 
                json_note=json_note,
                execution_context=execution_context
            )
            
            # log prompt length for debugging
            logger.info(f"Aggregator prompt length: {len(prompt)}")

            current_resp = self.advanced_llm.generate(prompt)
            
            # Safety check: If LLM returns the user query verbatim (echo), reject it
            if isinstance(current_resp, str) and current_resp.strip() == user_query.strip():
                 logger.error("LLM echoed user query! Using fallback.")
                 aggregated_text = "I apologize, but I failed to generate a proper summary of the results."
            else:
                 aggregated_text = current_resp

            logger.info(f"Successfully aggregated response: {str(aggregated_text)[:100]}...")

            return {
                "response": {
                    "text": aggregated_text,
                    "json_format": json_format
                }
            }

        except Exception as e:
            logger.error(f"Error in aggregation: {str(e)}", exc_info=True)
            # Fallback: simple concatenation with sources
            fallback_parts = []
            for output in agent_outputs:
                content = output.get('content', '')
                if isinstance(content, dict):
                    content = str(content)
                if content:
                    content_str = content.strip() if isinstance(content, str) else str(content)
                    fallback_parts.append(f"**From {output.get('source', 'unknown')}:**\\n{content_str}")
           
            fallback_text = "\\n\\n".join(fallback_parts) if fallback_parts else "Annotation data retrieved."

            return {
                "response": {
                    "text": fallback_text,
                    "json_format": json_format
                }
            }

    def finalize_response(self, state: AgentState) -> Dict[str, Any]:
        """Finalize and return the response, including clarifying question generation"""
        response = state.get("response", {})
        query = state.get("user_query")
        user_id = state.get("user_id")
        
        logger.info(f"Finalizing response for user: {user_id}")
        
        if not isinstance(response, dict):
            response = {"text": str(response), "json_format": None}
        response.setdefault("text", "")

        # --- Generate Clarifying Questions (Integrated) ---
        response_text = response.get("text", "")
        # Only generate if response is substantial enough
        if response_text and len(response_text.strip()) > 20:
             try:
                 from app.prompts.rag_prompts import CLARIFYING_QUESTIONS_PROMPT
                 prompt = CLARIFYING_QUESTIONS_PROMPT.format(
                    user_query=query,
                    assistant_response=response_text
                 )
                 
                 result = self.basic_llm.generate(prompt)
                 questions = []
                 if result:
                    lines = result.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                            if '.' in line and line[0].isdigit():
                                question = line.split('.', 1)[-1].strip()
                            else:
                                question = line.strip('- •').strip()
                            
                            if question and len(question) > 5: 
                                questions.append(question)
                 
                 if questions:
                     response["suggested_questions"] = questions[:5]
                     logger.info(f"Added {len(questions)} suggested questions to response")
                     
             except Exception as e:
                 logger.error(f"Error generating clarifying questions: {e}", exc_info=True)

        emit_to_user(user=user_id, message=response, status="completed")
        
        return {"response": response}


    def answer_from_graph_summaries(self, query, user_id, resource, token, graph_id):
        """Legacy method for backward compatibility"""
        logger.info(
            f"Answer from graph summaries called with query: {query}, user_id: {user_id}, "
            f"resource: {resource}, graph_id: {graph_id}"
        )
        
        try:
            if resource == "annotation":
                summary_result = self.graph_summarizer.summary(
                    token=token, graph_id=graph_id, user_query=query
                )
                summary_text = summary_result.get('text', '') if isinstance(summary_result, dict) else summary_result
                emit_to_user(user=user_id, message="Analyzing...")

            elif resource == "hypothesis":
                summary_result = self.hypothesis_generation.get_by_hypothesis_id(
                    token, graph_id, user_id, query
                )
                summary_text = summary_result.get('text', '') if isinstance(summary_result, dict) else summary_result
                emit_to_user(user=user_id, message="Analyzing...")
            else:
                return "Invalid resource type specified."
                
            return {"text": summary_text, "json_format": None}
            
        except Exception as e:
            logger.error("Error in answer_from_graph_summaries", exc_info=True)
            return {
                "text": f"Error processing query: {str(e)}",
                "json_format": None
            }

    def _prepare_dependency_context(self, dep_output: str, dependency_id: int) -> str:
        """
        Intelligently prepare context from dependency, summarizing if too long.
        This prevents token overflow while preserving important information.
        """
        MAX_CHARS = 1500  # Roughly 300-400 tokens
        
        if len(dep_output) <= MAX_CHARS:
            logger.info(f"Context from step {dependency_id}: {len(dep_output)} chars (within limit)")
            return dep_output
        
        logger.info(f"Summarizing long context from step {dependency_id} ({len(dep_output)} chars → target: ~400 words)")
        
        summarize_prompt = DEPENDENCY_SUMMARIZATION_PROMPT.format(content=dep_output)
        
        try:
            summary = self.basic_llm.generate(summarize_prompt)
            logger.info(f"Reduced context: {len(dep_output)} → {len(summary)} chars")
            
            return f"[Summarized from previous step]: {summary}"
        except Exception as e:
            logger.error(f"Summarization failed: {e}, using truncation fallback")
            truncated = dep_output[:MAX_CHARS] + "... [truncated due to length]"
            logger.warning(f"Using truncation fallback: {len(dep_output)} → {MAX_CHARS} chars")
            return truncated


    def update_step_state(self, state: AgentState) -> Dict[str, Any]:
        """
        Group-aware step advancement.
        - For SEQUENTIAL groups: advances one step at a time within the group.
        - For PARALLEL groups: captures all agent outputs and advances to the next group.
        Falls back to flat-plan behavior if execution_groups is missing.
        """
        plan = state.get("plan", [])
        execution_groups = state.get("execution_groups", [])
        step_outputs = dict(state.get("step_outputs", {}))
        user_id = state.get("user_id")

        response_key_map = {
            "rag_agent": "rag_response",
            "annotation_agent": "annotation_response",
            "galaxy_agent": "galaxy_response",
            "biogpt_agent": "biogpt_response",
            "content_retrieval_agent": "content_retrieval_response",
            "_hypothesis_agent": "response"
        }

        current_group_idx = state.get("current_group_index", 0)
        current_step_in_group = state.get("current_step_in_group", 0)

        if current_group_idx >= len(execution_groups):
            logger.info(" All groups completed, routing to aggregator")
            return {"step_input": None, "step_outputs": step_outputs}

        current_group = execution_groups[current_group_idx]
        mode = current_group.get("mode", "sequential")
        group_steps = current_group.get("steps", [])

        if mode == "parallel":
            # Visualize the parallel group start
            agent_names = [step.get("agent", "unknown") for step in group_steps]
            RichLogger.log_group_start(current_group_idx, mode, agent_names)
            
            for step in group_steps:
                self._capture_step_output(state, step, step_outputs, response_key_map)
            
            logger.info(f" Parallel group {current_group_idx + 1} completed ({len(group_steps)} agents)")
        else:
            if current_step_in_group < len(group_steps):
                self._capture_step_output(state, group_steps[current_step_in_group], step_outputs, response_key_map)

        if mode == "parallel":
            return self._advance_to_next_group(
                state, execution_groups, current_group_idx + 1, step_outputs, response_key_map, user_id
            )
        else:
            next_step_in_group = current_step_in_group + 1
            if next_step_in_group < len(group_steps):
                next_step = group_steps[next_step_in_group]
                next_agent_name = next_step.get("agent", "unknown")
                
                final_input = self._resolve_step_input(next_step, step_outputs, state, user_id, plan)
                if final_input is None:
                    return {
                        "current_group_index": len(execution_groups),
                        "current_step_in_group": 0,
                        "current_step_index": len(plan),
                        "step_outputs": step_outputs,
                        "step_input": None,
                        "error": "Dependency failed, skipping remaining steps"
                    }

                global_index = self._get_global_step_index(execution_groups, current_group_idx, next_step_in_group)

                emit_to_user(
                    user=user_id,
                    message=f"Step {global_index + 1}/{len(plan)}: Running {next_agent_name.replace('_', ' ').title()}...",
                    status="processing"
                )
                RichLogger.log_agent_start(next_agent_name, global_index + 1, len(plan))
                logger.info(f"Sequential group {current_group_idx + 1}, step {next_step_in_group + 1}/{len(group_steps)}: {next_agent_name}")

                return {
                    "current_group_index": current_group_idx,
                    "current_step_in_group": next_step_in_group,
                    "current_step_index": global_index,
                    "step_input": final_input,
                    "step_outputs": step_outputs,
                }
            else:
                return self._advance_to_next_group(
                    state, execution_groups, current_group_idx + 1, step_outputs, response_key_map, user_id
                )

    def _capture_step_output(self, state, step_info, step_outputs, response_key_map):
        """Capture the output of a completed step into step_outputs."""
        agent_name = step_info.get("agent")
        step_id = step_info.get("id")
        key = response_key_map.get(agent_name)
        
        if not key:
            return
        
        resp_data = state.get(key)
        if resp_data and isinstance(resp_data, dict):
            output_text = resp_data.get("text", "")
            json_data = resp_data.get("json_format")
            is_empty = not output_text and not json_data
            has_error_msg = (isinstance(output_text, str) and output_text.startswith("Error:")) or state.get("error")
            
            if is_empty or (has_error_msg and not json_data):
                error_msg = state.get("error", output_text or "Agent returned no data.")
                step_outputs[step_id] = f"FAILED: {error_msg}"
                logger.error(f"Step {step_id} ({agent_name}) failed: {error_msg}")
            else:
                step_outputs[step_id] = output_text if output_text else "[Structured Data Generated]"
                if has_error_msg and json_data:
                    logger.warning(f"  Step {step_id} ({agent_name}) had warnings but produced JSON data.")
                else:
                    RichLogger.log_agent_complete(agent_name, str(step_outputs[step_id]), success=True)
                    logger.info(f" Step {step_id} ({agent_name}) completed successfully.")

    def _advance_to_next_group(self, state, execution_groups, next_group_idx, step_outputs, response_key_map, user_id):
        """Advance to the next execution group, preparing inputs for its first step(s)."""
        plan = state.get("plan", [])

        if next_group_idx >= len(execution_groups):
            logger.info(f"All {len(execution_groups)} groups completed, routing to aggregator")
            return {
                "current_group_index": next_group_idx,
                "current_step_in_group": 0,
                "current_step_index": len(plan),
                "step_input": None,
                "step_outputs": step_outputs,
            }

        next_group = execution_groups[next_group_idx]
        next_mode = next_group.get("mode", "sequential")
        next_steps = next_group.get("steps", [])

        if not next_steps:
            return self._advance_to_next_group(state, execution_groups, next_group_idx + 1, step_outputs, response_key_map, user_id)

        global_index = self._get_global_step_index(execution_groups, next_group_idx, 0)

        if next_mode == "parallel":
            agent_names = [s.get("agent", "?") for s in next_steps]
            emit_to_user(
                user=user_id,
                message=f" Running {len(next_steps)} agents in parallel: {', '.join(n.replace('_', ' ').title() for n in agent_names)}...",
                status="processing"
            )
            RichLogger.log_group_start(next_group_idx, "parallel", agent_names)
            logger.info(f" Parallel group {next_group_idx + 1}: {agent_names}")
            
            first_input = self._resolve_step_input(next_steps[0], step_outputs, state, user_id, plan)
            if first_input is None:
                return {
                    "current_group_index": len(execution_groups),
                    "current_step_in_group": 0,
                    "current_step_index": len(plan),
                    "step_outputs": step_outputs,
                    "step_input": None,
                    "error": "Dependency failed for parallel group"
                }
            
            return {
                "current_group_index": next_group_idx,
                "current_step_in_group": 0,
                "current_step_index": global_index,
                "step_input": first_input,
                "step_outputs": step_outputs,
            }
        else:
            first_step = next_steps[0]
            next_agent_name = first_step.get("agent", "unknown")
            final_input = self._resolve_step_input(first_step, step_outputs, state, user_id, plan)
            
            if final_input is None:
                return {
                    "current_group_index": len(execution_groups),
                    "current_step_in_group": 0,
                    "current_step_index": len(plan),
                    "step_outputs": step_outputs,
                    "step_input": None,
                    "error": "Dependency failed, skipping remaining steps"
                }

            emit_to_user(
                user=user_id,
                message=f" Step {global_index + 1}/{len(plan)}: Running {next_agent_name.replace('_', ' ').title()}...",
                status="processing"
            )
            RichLogger.log_group_start(next_group_idx, "sequential", [next_agent_name])
            RichLogger.log_agent_start(next_agent_name, global_index + 1, len(plan))
            logger.info(f"📍 Sequential group {next_group_idx + 1}, step 1/{len(next_steps)}: {next_agent_name}")

            return {
                "current_group_index": next_group_idx,
                "current_step_in_group": 0,
                "current_step_index": global_index,
                "step_input": final_input,
                "step_outputs": step_outputs,
            }

    def _resolve_step_input(self, step_info, step_outputs, state, user_id, plan):
        """
        Resolve the input for a step, injecting dependency context if needed.
        Returns None if a required dependency has failed.
        """
        raw_input = step_info.get("input", state.get("user_query"))
        dependency_id = step_info.get("dependency")
        
        if not dependency_id:
            return raw_input
        
        dep_ids = dependency_id if isinstance(dependency_id, list) else [dependency_id]
        combined_dep_output = []
        failed_deps = []
        
        for d_id in dep_ids:
            d_out = step_outputs.get(d_id, "")
            if isinstance(d_out, str) and d_out.startswith("FAILED:"):
                failed_deps.append(d_id)
            elif d_out:
                p_out = self._prepare_dependency_context(d_out, d_id)
                combined_dep_output.append(p_out)

        if failed_deps:
            logger.error(f" Dependency steps {failed_deps} failed")
            emit_to_user(
                user=user_id,
                message=f"Required dependency failed, skipping dependent steps...",
                status="error"
            )
            return None
        
        if combined_dep_output:
            context_str = "\n---\n".join(combined_dep_output)
            final_input = f"{raw_input}. Context: {context_str}"
            logger.info(f" Injected context from steps {dep_ids}")
            return final_input
        
        return raw_input

    def _get_global_step_index(self, execution_groups, group_idx, step_in_group):
        """Calculate the global step index from group_idx and step_in_group."""
        count = 0
        for i, group in enumerate(execution_groups):
            if i < group_idx:
                count += len(group.get("steps", []))
            elif i == group_idx:
                count += step_in_group
                break
        return count



