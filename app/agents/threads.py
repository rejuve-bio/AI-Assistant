
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ThreadMemoryMixin:
    _THREAD_SUMMARY_CHUNK = 20

    def _record_thread_turn(self, thread_id: str, user_id: str, query: str, resp: Dict[str, Any],
                            graph_id: Optional[str] = None, resource: Optional[Any] = None,
                            client_message_id: Optional[str] = None,
                            files_part: Optional[list] = None,
                            context_to_record: Optional[Dict[str, Any]] = None,
                            quick_replies: Optional[list] = None) -> None:
        try:
            if context_to_record:
                self.store.append_message(thread_id, user_id, "system", "", context_part=context_to_record)

            self.store.append_message(
                thread_id, user_id, "user", query,
                graph_id=graph_id,
                resource_type=resource if isinstance(resource, str) else None,
                files_part=files_part,
            )

            resource = resp.get("resource")
            answered_about = graph_id
            resource_type = resource if isinstance(resource, str) else None
            if isinstance(resource, dict):
                resource_type = resource.get("type") or resource_type
                if resource.get("id"):
                    answered_about = resource["id"]
            if not resource_type:
                agents = resp.get("agents_completed") or []
                if "hypothesis_agent" in agents:
                    resource_type = "hypothesis"
                elif "annotation_agent" in agents:
                    resource_type = "annotation"

            thread_doc = self.store.append_message(
                thread_id,
                user_id,
                "assistant",
                resp.get("text", ""),
                json_format=resp.get("json_format"),
                agents=resp.get("agents_completed"),
                graph_id=answered_about,
                resource_type=resource_type,
                client_message_id=client_message_id,
                quick_replies=quick_replies,
                proposals=resp.get("proposals"),
            )

            label = (resp.get("text") or "")[:80]
            if isinstance(resource, dict) and resource.get("id"):
                self.store.record_tool_call(thread_id, user_id, resource.get("type", "unknown"), resource["id"], label)
            elif graph_id and resp.get("text"):
                agents = resp.get("agents_completed") or []
                agent = "content_retrieval" if "content_retrieval_agent" in agents else (agents[0] if agents else "unknown")
                self.store.record_tool_call(thread_id, user_id, agent, graph_id, label)

            watermark = thread_doc.get("summary_watermark", 0)
            if thread_doc.get("message_count", 0) - watermark >= self._THREAD_SUMMARY_CHUNK:
                self._extend_thread_summary(thread_id, user_id, thread_doc, watermark)
        except Exception as e:
            logger.warning(f"Failed to record thread turn for {thread_id}: {e}")

    def _extend_thread_summary(self, thread_id: str, user_id: str, thread_doc: Dict[str, Any], watermark: int) -> None:
        """Folds the messages since the last watermark into the thread's
        running summary -- never deletes or replaces the stored messages
        themselves (see _THREAD_SUMMARY_CHUNK's comment). Storing
        everything doesn't hurt; this summary exists only to give the LLM a
        compact digest of a long thread without it having to re-read every
        message every turn."""
        messages = thread_doc.get("messages", [])
        new_messages = messages[watermark:]
        if not new_messages:
            return
        transcript = "\n".join(f"{m.get('role', '?')}: {m.get('text', '')}" for m in new_messages)
        prior_summary = thread_doc.get("running_summary", "")
        prompt = (
            "Summarize this conversation excerpt into a concise running summary an "
            "assistant can use as context later. Merge it with the prior summary "
            "below rather than replacing it — keep anything from the prior summary "
            "still relevant.\n\n"
            f"Prior summary: {prior_summary or '(none yet)'}\n\n"
            f"Conversation excerpt:\n{transcript}\n\n"
            "Write the merged summary, 3-6 sentences, factual, no meta-commentary."
        )
        try:
            new_summary = self.advanced_llm.generate(prompt)
            new_summary = new_summary.strip() if isinstance(new_summary, str) else str(new_summary)
        except Exception as e:
            logger.warning(f"Thread summary extension failed for {thread_id}: {e}")
            return
        self.store.extend_running_summary(thread_id, user_id, new_summary, len(messages))

    def _reset_thread(self, config: dict) -> None:

        thread_id = config.get("configurable", {}).get("thread_id")
        checkpointer = getattr(self.app, "checkpointer", None)
        if not thread_id or not checkpointer:
            return
        try:
            checkpointer.delete_thread(thread_id)
        except Exception as e:
            logger.warning(f"Failed to reset checkpoint thread {thread_id}: {e}")
