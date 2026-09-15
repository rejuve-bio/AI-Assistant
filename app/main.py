from .prompts.conversation_handler import conversation_prompt
from .annotation_graph.annotated_graph import Graph
from .rag.rag import RAG
from .annotation_graph.summarizer import GraphSummarizer
from .hypothesis_generation.hypothesis import HypothesisGeneration
from .socket_manager import emit_to_user
from .Galaxy_integration.galaxy import GalaxyHandler
from .biogpt_agent.biogpt import BioGPTAgent
from .agents import (
    AgentNodesMixin,
    AggregationMixin,
    ConfirmationMixin,
    ThreadMemoryMixin,
    WorkflowMixin,
)
from .agents.state import ANALYZING_MSG
from typing import List, Any, Dict, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import logging


logger = logging.getLogger(__name__)

load_dotenv()




class AiAssistance(
    WorkflowMixin,
    AgentNodesMixin,
    ConfirmationMixin,
    AggregationMixin,
    ThreadMemoryMixin,
):

    def __init__(
        self,
        advanced_llm,
        basic_llm,
        schema_handler,
        fly_schema_handler=None,
        qdrant_client=None,
        embedding_model=None,
        mongo_db_manager=None,
        checkpointer=None,
    ) -> None:
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.annotation_graph = Graph(advanced_llm, schema_handler, fly_schema_handler=fly_schema_handler)
        self.graph_summarizer = GraphSummarizer(self.advanced_llm)
        self.rag = RAG(llm=advanced_llm, qdrant_client=qdrant_client)
        self.store = mongo_db_manager
        self.hypothesis_generation = HypothesisGeneration(advanced_llm)
        self.galaxy_handler = GalaxyHandler(advanced_llm, qdrant_client, embedding_model)
        self.embedding_model = embedding_model
        self.biogpt = BioGPTAgent(basic_llm=basic_llm, advanced_llm=advanced_llm)

        logger.info(
            f"AiAssistance initialized with advanced_llm: {type(self.advanced_llm).__name__}"
        )
        logger.info(f"Galaxy handler initialized: {type(self.galaxy_handler).__name__}")

        # Initialize the LangGraph workflow
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile(checkpointer=checkpointer)


    def agent(
        self,
        message: str,
        user_id: str,
        token: str,
        content_ids: Optional[List[str]] = None,
        graph_id: Optional[str] = None,
        urls: Optional[List[str]] = None,
        resource: Optional[Any] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main entry point for processing queries with parallel agent execution.
        """
        if not thread_id:
            raise ValueError(
                "thread_id is required — it names the conversation this message "
                "belongs to. Generate one per conversation on the client."
            )
        logger.info(
            f"Agent called with message: {message}, user_id: {user_id}, thread_id: {thread_id}, "
            f"content_ids: {content_ids}, graph_id: {graph_id}, urls: {urls}"
        )
        try:
            # Create initial state
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
                "pending_confirmation": None,
                "confirmation_outcome": None,
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

            config = {"configurable": {"thread_id": thread_id}}
            result = self.app.invoke(initial_state, config=config)

            response = self._extract_response_or_interrupt(result, config)
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

    def _extract_response_or_interrupt(self, result: Dict[str, Any], config: dict) -> Dict[str, Any]:
        """Build the response dict from an invoke() result, whether the graph
        finished normally or paused on any agent's confirmation interrupt().
        """
        state = self.app.get_state(config)
        if state.next:
            pending_confirmation = state.values.get("pending_confirmation") or {}
            data = pending_confirmation.get("data", {})
            return {
                "text": pending_confirmation.get("confirmation_text", ""),
                "json_format": None,
                "needs_confirmation": True,
                "confirmation": {
                    "options": self._confirmation_options(data),
                    "allow_free_text": True,
                },
                "agents_completed": result.get("agents_completed", []),
            }

        self._reset_thread(config)

        response = result.get("response", {"text": ""})
        if not isinstance(response, dict):
            response = {"text": str(response), "json_format": None}
        else:
            response.setdefault("text", "")
            response.setdefault("json_format", None)

        response["agents_completed"] = result.get("agents_completed", [])
        return response


    def _route_to_agent(self, response: str, query: str, user_id: str, token: str,
                        graph_id, content_ids, urls, resource, thread_id: Optional[str] = None) -> Dict[str, Any]:
        if "response:" in response:
            result = response.split("response:")[1].strip()
            final_response = result.strip('"')
            self.store.create_history(
                user_id=user_id,
                user_message=query,
                assistant_answer=final_response,
                graph_id_referenced=graph_id,
                content_ids=content_ids,
                urls=urls,
                agents_used=[],
            )
            self._record_thread_turn(thread_id, user_id, query, {"text": final_response}, graph_id=graph_id, resource=resource)
            emit_to_user(user=user_id, message=final_response, status="completed")
            return {"text": final_response}

        if "question:" in response:
            refactored_question = response.split("question:")[1].strip()
            agent_response = self.agent(
                refactored_question,
                user_id,
                token,
                content_ids=content_ids,
                graph_id=graph_id,
                urls=urls,
                resource=resource,
                thread_id=thread_id,
            )
            if isinstance(agent_response, str):
                agent_response = {"text": agent_response, "agents_completed": []}
            elif not isinstance(agent_response, dict):
                agent_response = {"text": str(agent_response), "agents_completed": []}
            resource_data = agent_response.get("resource")
            if isinstance(resource_data, dict):
                resource_type = resource_data.get("type")
                if resource_type:
                    logger.info(f"Resource successfully created: {resource_type}")
            assistant_answer = agent_response.get("text", str(agent_response))
            agents_used = agent_response.get("agents_completed", [])
            self.store.create_history(
                user_id=user_id,
                user_message=query,
                assistant_answer=assistant_answer,
                graph_id_referenced=graph_id,
                content_ids=content_ids,
                urls=urls,
                agents_used=agents_used,
            )
            self._record_thread_turn(thread_id, user_id, query, agent_response, graph_id=graph_id, resource=resource)
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

    def assistant_response(
        self,
        query: str,
        user_id: str,
        token: str,
        graph_id: Optional[str] = None,
        urls: Optional[List[str]] = None,
        content_ids: Optional[List[str]] = None,
        resource: Optional[Any] = None,
        resume: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        
        thread_id = thread_id or user_id
        try:
            logger.info(
                f"Assistant response called with query={query}, user_id={user_id}, thread_id={thread_id}, "
                f"graph_id={graph_id}, content_ids={content_ids}, urls={urls}, resource={resource}, resume={resume}"
            )

            if resume is not None:
                resp = self.resume_confirmation_with_value(resume, user_id, thread_id=thread_id)
                self.store.create_history(
                    user_id=user_id,
                    user_message=f"[resume:{resume}]",
                    assistant_answer=resp.get("text", ""),
                    graph_id_referenced=graph_id,
                    content_ids=content_ids,
                    urls=urls,
                    agents_used=resp.get("agents_completed", []),
                )
                self._record_thread_turn(thread_id, user_id, f"[resume:{resume}]", resp, graph_id=graph_id, resource=resource)
                emit_to_user(user=user_id, message=resp, status="completed")
                return resp
            try:
                is_pending = bool(self.app.get_state({"configurable": {"thread_id": thread_id}}).next)
            except Exception as e:
                logger.warning(f"Could not check for a pending annotation confirmation: {e}")
                is_pending = False

            if is_pending:
                resp = self.resume_pending_confirmation(query, user_id, thread_id=thread_id)
                if resp is not None:
                    self.store.create_history(
                        user_id=user_id,
                        user_message=query,
                        assistant_answer=resp.get("text", ""),
                        graph_id_referenced=graph_id,
                        content_ids=content_ids,
                        urls=urls,
                        agents_used=resp.get("agents_completed", []),
                    )
                    self._record_thread_turn(thread_id, user_id, query, resp, graph_id=graph_id, resource=resource)
                    emit_to_user(user=user_id, message=resp, status="completed")
                    return resp

            try:
                thread_doc = self.store.get_thread(thread_id, user_id)
                running_summary = thread_doc.get("running_summary", "")
                messages = thread_doc.get("messages", [])
                history = []
                i = 0
                while i < len(messages):
                    if messages[i].get("role") == "user":
                        question = messages[i].get("text", "")
                        answer = ""
                        if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                            answer = messages[i + 1].get("text", "")
                            i += 1
                        history.append({"question": question, "context": {"answer": answer}})
                    i += 1
                memory = [running_summary] if running_summary else []
            except Exception:
                history = []
                memory = []

            logger.info(f"Histories of the user are: {history} and memories are {memory}")

            prompt = conversation_prompt.format(
                memory=memory,
                query=query,
                conversation_history=history,
                graph_id=graph_id or "",
            )
            logger.info("Advanced llm response")
            response = self.advanced_llm.generate(prompt)
            logger.info(f"Response from the advanced LLM: {response}")
            emit_to_user(user=user_id, message=ANALYZING_MSG)

            return self._route_to_agent(
                response or "",
                query, user_id, token, graph_id, content_ids, urls, resource,
                thread_id=thread_id,
            )

        except Exception as e:
            logger.error(f"Error in assistant_response: {e}", exc_info=True)
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
                logger.error(f"Failed to save error history: {save_error}")
            return {
                "text": error_msg,
                "json_format": None
            }

