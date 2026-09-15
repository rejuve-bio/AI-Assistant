
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ThreadMemoryMixin:
    _THREAD_COMPACT_THRESHOLD = 20
    _THREAD_KEEP_RECENT = 10

    def _record_thread_turn(self, thread_id: str, user_id: str, query: str, resp: Dict[str, Any],
                            graph_id: Optional[str] = None, resource: Optional[Any] = None) -> None:
        try:
            self.store.append_message(
                thread_id, user_id, "user", query,
                graph_id=graph_id,
                resource_type=resource if isinstance(resource, str) else None,
            )

            resource = resp.get("resource")
            answered_about = graph_id
            resource_type = resource if isinstance(resource, str) else None
            if isinstance(resource, dict) and resource.get("id"):
                answered_about = resource["id"]
                resource_type = resource.get("type") or resource_type
            if answered_about and not resource_type:
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
            )

            label = (resp.get("text") or "")[:80]
            if isinstance(resource, dict) and resource.get("id"):
                self.store.record_tool_call(thread_id, user_id, resource.get("type", "unknown"), resource["id"], label)
            elif graph_id and resp.get("text"):
                agents = resp.get("agents_completed") or []
                agent = "content_retrieval" if "content_retrieval_agent" in agents else (agents[0] if agents else "unknown")
                self.store.record_tool_call(thread_id, user_id, agent, graph_id, label)

            if thread_doc.get("message_count", 0) >= self._THREAD_COMPACT_THRESHOLD:
                self._compact_thread(thread_id, user_id, thread_doc)
        except Exception as e:
            logger.warning(f"Failed to record thread turn for {thread_id}: {e}")

    def _compact_thread(self, thread_id: str, user_id: str, thread_doc: Dict[str, Any]) -> None:

        messages = thread_doc.get("messages", [])
        keep = self._THREAD_KEEP_RECENT
        if len(messages) <= keep:
            return
        older, recent = messages[:-keep], messages[-keep:]
        transcript = "\n".join(f"{m.get('role', '?')}: {m.get('text', '')}" for m in older)
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
            logger.warning(f"Thread compaction summarization failed for {thread_id}: {e}")
            return
        self.store.compact_thread(thread_id, user_id, new_summary, recent)

    def _reset_thread(self, config: dict) -> None:

        thread_id = config.get("configurable", {}).get("thread_id")
        checkpointer = getattr(self.app, "checkpointer", None)
        if not thread_id or not checkpointer:
            return
        try:
            checkpointer.delete_thread(thread_id)
        except Exception as e:
            logger.warning(f"Failed to reset checkpoint thread {thread_id}: {e}")
