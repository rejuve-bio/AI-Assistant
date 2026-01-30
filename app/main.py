from .llm_handle.llm_models import (
    LLMInterface,
    OpenAIModel,
    get_llm_model,
    openai_embedding_model,
)
from .prompts.conversation_handler import conversation_prompt
from .socket_manager import emit_to_user
from typing import TypedDict, List, Annotated, Any, Dict, Optional
from flask_socketio import emit
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import asyncio
import traceback
import json
import os
import operator
import logging
import logging.handlers as loghandlers
import time
from .agents import AgentManager, AgentState

logger = logging.getLogger(__name__)
log_dir = "/AI-Assistant/logfiles"

log_file = os.path.join(log_dir, "Assistant.log")
logger.setLevel(logging.DEBUG)
loghandle = loghandlers.TimedRotatingFileHandler(
    filename="logfiles/Assistant.log",
    when="D",
    interval=1,
    backupCount=7,
    encoding="utf-8",
)
loghandle.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(loghandle)

load_dotenv()


class AiAssistance:

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
        
        # Instantiate the AgentManager
        self.agents = AgentManager(
            advanced_llm=advanced_llm,
            basic_llm=basic_llm,
            schema_handler=schema_handler,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            mongo_db_manager=mongo_db_manager
        )

        logger.info(
            f"AiAssistance initialized with advanced_llm: {type(self.advanced_llm).__name__}"
        )

        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with iterative execution"""
        logger.info("Creating LangGraph workflow with iterative execution")

        workflow = StateGraph(AgentState)

        workflow.add_node("classifier", self.agents.classify_query)
        workflow.add_node("increment_step", self.increment_step)
        
        workflow.add_node("annotation_agent", self.agents.annotation_agent)
        workflow.add_node("rag_agent", self.agents.rag_agent)
        workflow.add_node("galaxy_agent", self.agents.galaxy_agent)
        workflow.add_node("biogpt_agent", self.agents.biogpt_agent)
        workflow.add_node("content_retrieval_agent", self.agents.content_retrieval_agent)
        workflow.add_node("_hypothesis_agent", self.agents.hypothesis_agent)
        
        workflow.add_node("aggregator", self.agents.aggregate_responses)
        # workflow.add_node("clarifying_questions", self.agents.generate_clarifying_questions) # Removed
        workflow.add_node("finalizer", self.agents.finalize_response)
        
        workflow.set_entry_point("classifier")
        
        # Router logic from classifier and after each step
        workflow.add_conditional_edges(
            "classifier",
            self._route_to_agents,
            [
                "annotation_agent", 
                "rag_agent", 
                "galaxy_agent", 
                "biogpt_agent", 
                "content_retrieval_agent",
                "_hypothesis_agent",
                "aggregator",
                "finalizer"
            ]
        )
        
        # Agents route to increment_step
        workflow.add_edge("annotation_agent", "increment_step")
        workflow.add_edge("rag_agent", "increment_step")
        workflow.add_edge("galaxy_agent", "increment_step")
        workflow.add_edge("biogpt_agent", "increment_step")
        workflow.add_edge("content_retrieval_agent", "increment_step")
        workflow.add_edge("_hypothesis_agent", "increment_step")
        
        # Loop back to router logic
        workflow.add_conditional_edges(
            "increment_step",
            self._route_to_agents,
             [
                "annotation_agent", 
                "rag_agent", 
                "galaxy_agent", 
                "biogpt_agent", 
                "content_retrieval_agent",
                "_hypothesis_agent",
                "aggregator",
                "finalizer"
            ]
        )
        
        workflow.add_edge("aggregator", "finalizer")
        # workflow.add_edge("aggregator", "clarifying_questions")
        # workflow.add_edge("clarifying_questions", "finalizer")
        workflow.add_edge("finalizer", END)
        
        return workflow

    def increment_step(self, state: AgentState) -> Dict[str, Any]:
        """Wrap the step update"""
        return self.agents.update_step_state(state)

    def _route_to_agents(self, state: AgentState) -> str:
        """
        Determine which agent to run next based on the plan and current step.
        """
        plan = state.get("plan", [])
        current_index = state.get("current_step_index", 0)
        
        # Enhanced logging
        logger.info(f"🎯 Router called: index={current_index}, plan_length={len(plan)}")
        
        # Case: Invalid query (refusal)
        if not plan:
             # If response text exists (refusal) and no plan, go to finalizer.
             # If completely empty default to AGGREGATOR (which might handle "no info")
             if state.get("response", {}).get("text"):
                  logger.info("❌ No plan (query was rejected). Routing to finalizer.")
                  return "finalizer"
             logger.info("⚠️  No plan and no response. Routing to aggregator.")
             return "aggregator"

        # Case: Plan execution done
        if current_index >= len(plan):
            logger.info(f"✅ Plan execution completed ({len(plan)} steps). Routing to aggregator.")
            return "aggregator"
        
        current_step = plan[current_index]
        agent_name = current_step.get("agent")
        
        logger.info(f"➡️  Routing to step {current_index + 1}/{len(plan)}: {agent_name}")
    
        # Check if valid node
        if agent_name in [
            "annotation_agent", "rag_agent", "galaxy_agent", 
            "biogpt_agent", "content_retrieval_agent", "_hypothesis_agent"
        ]:
            return agent_name
            
        logger.warning(f"⚠️  Unknown agent in plan: {agent_name}, skipping to aggregator")
        return "aggregator"

    def agent(
        self,
        message: str,
        user_id: str,
        token: str,
        content_ids: Optional[List[str]] = None,
        graph_id: Optional[str] = None,
        urls: Optional[List[str]] = None,
        resource: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main entry point for processing queries"""
        logger.info(
            f"Agent called with message: {message}, user_id: {user_id}, "
            f"content_ids: {content_ids}, graph_id: {graph_id}, urls: {urls}"
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
                "agents_to_run": [],
                "agents_completed": [],
                "suggested_questions": None,
            }

            result = self.app.invoke(initial_state)

            # Extract the structured response
            response = result.get("response", {"text": ""})
            
            # Ensure consistent structure
            if not isinstance(response, dict):
                response = {"text": str(response), "json_format": None}
            else:
                response.setdefault("text", "")
                response.setdefault("json_format", None)

            response["agents_completed"] = result.get("agents_completed", [])
            
            logger.info(f"Agent completed successfully for user: {user_id}")
            return response

        except Exception as e:
            logger.error("Error in agent processing", exc_info=True)
            error_response = {
                "text": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "json_format": None,
                "agents_completed": []
            }
            emit_to_user(user=user_id, message=error_response, status="error")
            return error_response


    def assistant_response(
        self, 
        query: str, 
        user_id: str, 
        token: str, 
        graph_id: Optional[str] = None,
        urls: Optional[List[str]] = None,
        content_ids: Optional[List[str]] = None,
        resource: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for assistant responses.
        Routes to agent execution system.
        """
        try:
            logger.info(
                f"Assistant response called with query={query}, user_id={user_id}, "
                f"graph_id={graph_id}, content_ids={content_ids}, urls={urls}"
            )
            
            # Get conversation history and memory
            try:
                user_information = self.store.get_context_and_memory(user_id)
                history = []
                memory = []
                for item in user_information:
                    q = item["QUESTION"]["question"]
                    c = item["QUESTION"]["context"]
                    m = item["MEMORIES"]
                    history.append({"question": q, "context": c})
                    memory.append(m)
            except Exception as e:
                history = []
                memory = []

            logger.info(f"Histories of the user are: {history} and memories are {memory}")

            # Generate LLM response to decide routing
            prompt = conversation_prompt.format(
                memory=memory,
                query=query,
                conversation_history=history,
            )
            logger.info("Advanced llm response")
            response = self.advanced_llm.generate(prompt)
            logger.info(f"Response from the advanced LLM: {response}")
            emit_to_user(user=user_id, message="Analyzing...")
            
            if response:
                # Case 1: Direct response (no agent needed)
                if "response:" in response:
                    result = response.split("response:")[1].strip()
                    final_response = result.strip('"')
                    
                    # ✅ Save history with all available info
                    self.store.create_history(
                        user_id=user_id,
                        user_message=query,
                        assistant_answer=final_response,
                        graph_id_referenced=graph_id,
                        content_ids=content_ids,
                        urls=urls,
                        agents_used=[],  # No agents used for direct response
                    )
                    
                    emit_to_user(user=user_id, message=final_response, status="completed")
                    return {"text": final_response}

                # Case 2: Agent response (needs processing)
                elif "question:" in response:
                    refactored_question = response.split("question:")[1].strip()
                    
                    # Call agent with all parameters
                    agent_response = self.agent(
                        refactored_question,
                        user_id,
                        token,
                        content_ids=content_ids,
                        graph_id=graph_id,
                        urls=urls,
                        resource=resource,
                    )
                    
                    # Normalize response to dict
                    if isinstance(agent_response, str):
                        agent_response = {"text": agent_response, "agents_completed": []}
                    elif not isinstance(agent_response, dict):
                        agent_response = {"text": str(agent_response), "agents_completed": []}

                    # Log resource type if available
                    resource_type = agent_response.get("resource", {}).get("type")
                    if resource_type:
                        logger.info(f"Resource successfully created: {resource_type}")

                    # Extract answer
                    assistant_answer = agent_response.get("text", str(agent_response))
                    
                    # Extract agents that were used
                    agents_used = agent_response.get("agents_completed", [])
                    
                    # ✅ Save complete history with ALL information
                    self.store.create_history(
                        user_id=user_id,
                        user_message=query,  # Original query, not refactored
                        assistant_answer=assistant_answer,
                        graph_id_referenced=graph_id,
                        content_ids=content_ids,
                        urls=urls,
                        agents_used=agents_used,
                    )
                    
                    emit_to_user(user=user_id, message=agent_response, status="completed")
                    return agent_response
                    
            else:
                # No response generated
                logger.error("No response generated from LLM")
                error_msg = "I apologize, but I encountered an error while processing your request."
                
                # ✅ Save the error attempt
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
        
        except Exception as e:
            logger.error(f"Error in assistant_response: {e}", exc_info=True)
            error_msg = "I apologize, but I encountered an error while processing your request."
            
            # ✅ Try to save error history
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
                logger.error(f"Failed to save error history: {save_error}")
            
            return {
                "text": error_msg,
                "json_format": None
            }