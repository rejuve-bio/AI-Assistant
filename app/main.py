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
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import os
import logging
import logging.handlers as loghandlers
from .agents import AgentManager, AgentState
from .utils import RichLogger

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
        """Create the LangGraph workflow with parallel + sequential execution support"""
        logger.info("Creating LangGraph workflow with parallel + sequential execution")

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
        
    agent_nodes = [
        "annotation_agent", 
        "rag_agent", 
        "galaxy_agent", 
        "biogpt_agent", 
        "content_retrieval_agent",
        "_hypothesis_agent",
        "aggregator",
        "finalizer"
    ]
        workflow.add_conditional_edges(
            "classifier",
            self._route_to_agents,
            agent_nodes
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
            agent_nodes
        )
        
        workflow.add_edge("aggregator", "finalizer")
        # workflow.add_edge("aggregator", "clarifying_questions")
        # workflow.add_edge("clarifying_questions", "finalizer")
        workflow.add_edge("finalizer", END)
        
        return workflow

    def increment_step(self, state: AgentState) -> Dict[str, Any]:
        """Wrap the step update"""
        return self.agents.update_step_state(state)

    def _route_to_agents(self, state: AgentState):
        """
        Group-aware router. Determines which agent(s) to run next.
        - For PARALLEL groups: returns a LIST of agent names (LangGraph runs them concurrently)
        - For SEQUENTIAL groups: returns a single agent name
        - When all groups are done: routes to aggregator
        """
        execution_groups = state.get("execution_groups", [])
        
        current_group_idx = state.get("current_group_index", 0)
        current_step_in_group = state.get("current_step_in_group", 0)
        
        if current_group_idx == 0 and current_step_in_group == 0:
            plan = state.get("plan", [])
            if plan:
                RichLogger.log_plan(plan)

        if current_group_idx >= len(execution_groups):
            if state.get("response", {}).get("text"):
                logger.info("No plan (query was rejected). Routing to finalizer.")
                return "finalizer"
            logger.info("All groups done. Routing to aggregator.")
            return "aggregator"
        
        current_group = execution_groups[current_group_idx]
        mode = current_group.get("mode", "sequential")
        group_steps = current_group.get("steps", [])
        
        if not group_steps:
            logger.warning("Empty group, routing to aggregator")
            return "aggregator"
        
        if mode == "parallel":
            # Return ALL agent names in this group — LangGraph runs them concurrently
            agent_names = []
            for step in group_steps:
                agent_name = step.get("agent")
                if agent_name in [
                    "annotation_agent", "rag_agent", "galaxy_agent",
                    "biogpt_agent", "content_retrieval_agent", "_hypothesis_agent"
                ]:
                    agent_names.append(agent_name)
                else:
                    logger.warning(f" Unknown agent in parallel group: {agent_name}")
            
            if not agent_names:
                logger.warning("No valid agents in parallel group, routing to aggregator")
                return "aggregator"
            
            logger.info(f"PARALLEL routing to: {agent_names}")
            RichLogger.log_router_decision(f"{len(agent_names)} Agents (Parallel)", str(agent_names))
            return agent_names
        else:
            if current_step_in_group >= len(group_steps):
                logger.info(" Sequential group exhausted, routing to aggregator")
                return "aggregator"
            
            step = group_steps[current_step_in_group]
            agent_name = step.get("agent")
            
            if agent_name in [
                "annotation_agent", "rag_agent", "galaxy_agent",
                "biogpt_agent", "content_retrieval_agent", "_hypothesis_agent"
            ]:
                logger.info(f" SEQUENTIAL routing to: {agent_name}")
                RichLogger.log_router_decision(agent_name, "Sequential Step")
                return agent_name
            
            logger.warning(f"Unknown agent: {agent_name}, routing to aggregator")
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
                "execution_groups": [],
                "current_group_index": 0,
                "current_step_in_group": 0,
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
            RichLogger.log_workflow_start(query)
            
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