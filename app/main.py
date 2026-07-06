from .socket_manager import emit_to_user
from .agents import Orchestrator, AgentState
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage
from app.utils.langsmith_tracking import (
    current_usage_tracker,
    llm_usage_step,
    start_query_tracking,
    tracked_node,
)
import json
import os
import uuid
import logging
import logging.handlers as loghandlers

logger = logging.getLogger(__name__)
loghandle = loghandlers.TimedRotatingFileHandler(
    filename="logfiles/Assistant.log",
    when="D", interval=1, backupCount=7, encoding="utf-8",
)
loghandle.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.setLevel(logging.DEBUG)
logger.addHandler(loghandle)

load_dotenv()


class AiAssistance:

    def __init__(self, advanced_llm, basic_llm, schema_handler,
                 qdrant_client=None, embedding_model=None, mongo_db_manager=None):
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.store = mongo_db_manager

        self.agents = Orchestrator(
            advanced_llm=advanced_llm,
            basic_llm=basic_llm,
            schema_handler=schema_handler,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            mongo_db_manager=mongo_db_manager,
        )
        # In-memory store for pending annotation confirmations: {user_id: pending_json}
        self._pending_confirmations: Dict[str, Any] = {}

        logger.info(f"AiAssistance initialized: {type(advanced_llm).__name__}")
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()

    # ─────────────────────────────────────────────────────────────────────────
    # LangGraph workflow
    # ─────────────────────────────────────────────────────────────────────────

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node(
            "classifier",
            tracked_node("classifier", self.agents.classify_query),
        )
        workflow.add_node(
            "dag_scheduler",
            tracked_node("dag_scheduler", self._dag_scheduler_node),
        )
        workflow.add_node(
            "step_executor",
            tracked_node("step_executor", self.agents.step_executor),
        )
        workflow.add_node(
            "sync_node",
            tracked_node("sync_node", self.agents.sync_node),
        )
        workflow.add_node(
            "replanner",
            tracked_node("replanner", self.agents.replanner),
        )
        workflow.add_node(
            "aggregator",
            tracked_node("aggregator", self.agents.aggregate_responses),
        )
        workflow.add_node(
            "finalizer",
            tracked_node("finalizer", self.agents.finalize_response),
        )

        workflow.set_entry_point("classifier")

        # After classifier: if plan is empty (rejected/direct) → finalizer, else → dag_scheduler
        workflow.add_conditional_edges(
            "classifier",
            self._after_classify,
            ["dag_scheduler", "finalizer"],
        )

        # dag_scheduler dispatches ready steps via Send or exits to aggregator
        workflow.add_conditional_edges(
            "dag_scheduler",
            self.agents.dag_scheduler,
            ["step_executor", "aggregator"],
        )

        # Each step_executor instance → sync_node (barrier, waits for all parallel instances)
        workflow.add_edge("step_executor", "sync_node")

        # sync_node: more steps ready → dag_scheduler, plan complete (first time) → replanner
        workflow.add_conditional_edges(
            "sync_node",
            self.agents.should_continue_dag,
            {"dag_scheduler": "dag_scheduler", "replanner": "replanner", "aggregator": "aggregator"},
        )

        # replanner: added new steps → dag_scheduler, nothing to add → aggregator
        workflow.add_conditional_edges(
            "replanner",
            self.agents.after_replan,
            ["dag_scheduler", "aggregator"],
        )

        workflow.add_edge("aggregator", "finalizer")
        workflow.add_edge("finalizer", END)

        return workflow

    def _dag_scheduler_node(self, state: AgentState) -> Dict[str, Any]:
        """Thin wrapper so the node exists; routing is in the conditional edge function."""
        return {}

    def _after_classify(self, state: AgentState) -> str:
        plan = state.get("plan", [])
        if not plan:
            return "finalizer"
        return "dag_scheduler"

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry points
    # ─────────────────────────────────────────────────────────────────────────

    def agent(self, message: str, user_id: str, token: str,
              content_ids: Optional[List[str]] = None,
              graph_id: Optional[str] = None,
              urls: Optional[List[str]] = None,
              resource: Optional[Any] = None,
              conversation_history: Optional[List] = None) -> Dict[str, Any]:

        logger.info(f"agent() called: user={user_id} query={message[:80]}")

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
            # DAG fields
            "plan": None,
            "current_step": None,
            "completed_step_ids": [],
            "step_outputs": {},
            "step_agent_outputs": [],
            "action_retry_count": 0,
            "replan_count": 0,
            "agents_to_run": [],
            "agents_completed": [],
            "suggested_questions": None,
            "session_id": str(uuid.uuid4()),
            "conversation_history": conversation_history or [],
            "_usage_tracker": current_usage_tracker(),
        }

        try:
            result = self.app.invoke(initial_state)
            response = result.get("response", {"text": ""})
            if not isinstance(response, dict):
                response = {"text": str(response), "json_format": None}
            response.setdefault("text", "")
            response.setdefault("json_format", None)
            response["agents_completed"] = result.get("agents_completed", [])
            logger.info(f"agent() done for user={user_id}")
            return response
        except Exception as e:
            logger.error("agent() error", exc_info=True)
            err = {"text": f"Error processing request: {e}", "json_format": None, "agents_completed": []}
            emit_to_user(user=user_id, message=err, status="error")
            return err

    def assistant_response(self, query: str, user_id: str, token: str,
                           graph_id: Optional[str] = None,
                           urls: Optional[List[str]] = None,
                           content_ids: Optional[List[str]] = None,
                           resource: Optional[Any] = None) -> Dict[str, Any]:
        query = query or ""
        metadata = {
            "graph_id": graph_id,
            "resource": resource,
            "content_count": len(content_ids or []),
            "url_count": len(urls or []),
        }
        with start_query_tracking(
            user_id=str(user_id),
            query=query or "",
            metadata=metadata,
        ) as tracker:
            response = self._assistant_response_impl(
                query=query,
                user_id=user_id,
                token=token,
                graph_id=graph_id,
                urls=urls,
                content_ids=content_ids,
                resource=resource,
            )
            if not isinstance(response, dict):
                response = {
                    "text": str(response),
                    "json_format": None,
                }
            response["usage"] = tracker.as_dict()
            return response

    def _assistant_response_impl(self, query: str, user_id: str, token: str,
                                 graph_id: Optional[str] = None,
                                 urls: Optional[List[str]] = None,
                                 content_ids: Optional[List[str]] = None,
                                 resource: Optional[Any] = None) -> Dict[str, Any]:

        logger.info(f"assistant_response(): user={user_id} query={query[:80]}")

        try:
            emit_to_user(user=user_id, message="Analyzing...")

            # Check if this message is a reply to a pending annotation confirmation
            pending = self._pending_confirmations.get(user_id)
            if pending:
                verdict = self._classify_confirmation(query)
                if verdict == "confirm":
                    resolved_json = self._apply_pending_substitutions(pending, apply=True)
                    text = "Got it! I've built the annotation structure using the confirmed match. The structured data is ready."
                    del self._pending_confirmations[user_id]
                    self._save_history(user_id, query, text, graph_id, content_ids, urls, ["annotation_agent"])
                    resp = {"text": text, "json_format": resolved_json, "agents_completed": ["annotation_agent"]}
                    emit_to_user(user=user_id, message=resp, status="completed")
                    return resp
                elif verdict == "reject":
                    resolved_json = self._apply_pending_substitutions(pending, apply=False)
                    text = "Understood! I've built the annotation structure without the unidentified node. The structured data is ready."
                    del self._pending_confirmations[user_id]
                    self._save_history(user_id, query, text, graph_id, content_ids, urls, ["annotation_agent"])
                    resp = {"text": text, "json_format": resolved_json, "agents_completed": ["annotation_agent"]}
                    emit_to_user(user=user_id, message=resp, status="completed")
                    return resp
                else:
                    # New unrelated query — clear stale pending state
                    del self._pending_confirmations[user_id]

            past = self.store.retrieve_user_history(user_id, limit=2) if self.store else {}
            conversation_history = list(reversed(past.get(str(user_id), [])))  # oldest first
            agent_resp = self.agent(
                query, user_id, token,
                content_ids=content_ids, graph_id=graph_id, urls=urls, resource=resource,
                conversation_history=conversation_history,
            )
            if not isinstance(agent_resp, dict):
                agent_resp = {"text": str(agent_resp), "agents_completed": []}

            answer = agent_resp.get("text", "")
            agents_used = agent_resp.get("agents_completed", [])

            # If the annotation pipeline needs confirmation, hold pending_json in memory
            if agent_resp.get("needs_confirmation"):
                pending_json = agent_resp.pop("pending_json", None)
                agent_resp.pop("unconfirmed_nodes", None)
                if pending_json:
                    self._pending_confirmations[user_id] = pending_json

            self._save_history(user_id, query, answer, graph_id, content_ids, urls, agents_used)

            # Strip any remaining pending fields before returning to caller
            agent_resp.pop("pending_json", None)
            agent_resp.pop("unconfirmed_nodes", None)

            emit_to_user(user=user_id, message=agent_resp, status="completed")
            return agent_resp

        except Exception as e:
            logger.error(f"assistant_response() error: {e}", exc_info=True)
            err_msg = "I apologize, but I encountered an error while processing your request."
            try:
                self._save_history(user_id, query, err_msg, graph_id, content_ids, urls, [])
            except Exception:
                pass
            return {"text": err_msg, "json_format": None}

    def _save_history(self, user_id, query, answer, graph_id, content_ids, urls, agents_used):
        try:
            self.store.create_history(
                user_id=user_id,
                user_message=query,
                assistant_answer=answer,
                graph_id_referenced=graph_id,
                content_ids=content_ids,
                urls=urls,
                agents_used=agents_used,
            )
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Confirmation flow helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_confirmation(self, message: str) -> str:
        """Returns 'confirm', 'reject', or 'new_query' using the LLM."""
        prompt = (
            f"The assistant asked a yes/no confirmation question about substituting an unrecognized "
            f"gene name with a suggested match. The user replied:\n\n\"{message}\"\n\n"
            f"Classify the reply as exactly one of:\n"
            f"- confirm  (user agrees to use the suggested match)\n"
            f"- reject   (user wants to skip/exclude it and build without it)\n"
            f"- new_query  (user is asking something unrelated, not answering the confirmation)\n\n"
            f"Reply with only one word: confirm, reject, or new_query."
        )
        try:
            with llm_usage_step("confirmation_classifier"):
                result = self.agents.basic_llm.generate(prompt)
            verdict = result.strip().lower().split()[0] if result else "new_query"
            if verdict in ("confirm", "reject", "new_query"):
                return verdict
            return "new_query"
        except Exception:
            return "new_query"

    def _apply_pending_substitutions(self, pending_json: dict, apply: bool = True) -> dict:
        import copy
        result = copy.deepcopy(pending_json)
        for node in result.get("nodes", []):
            subs = node.pop("pending_substitutions", {})
            node.pop("needs_confirmation", None)
            node.pop("all_list_values", None)
            node.pop("not_validated", None)

            props = node.get("properties", {})
            if apply and subs:
                for prop_key, prop_val in list(props.items()):
                    if isinstance(prop_val, str):
                        items = [v.strip() for v in prop_val.split(",") if v.strip()]
                        # Single value: replace if it matches
                        if len(items) == 1 and items[0] in subs:
                            props[prop_key] = subs[items[0]]
                        else:
                            # List: append confirmed substitutions
                            confirmed = [subs[orig] for orig in subs if orig not in items]
                            props[prop_key] = ", ".join(items + confirmed)
            elif not apply and subs:
                # Rejection: originals were already excluded from the list; nothing to add
                pass

            node["status"] = True
        return result
