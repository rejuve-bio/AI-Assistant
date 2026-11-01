
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.state import AgentState
from app.socket_manager import emit_to_user

logger = logging.getLogger(__name__)


class AggregationMixin:
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

    def _finalize_response(self, state: AgentState) -> Dict[str, Any]:
        user_id = state.get("user_id")
        messages = state.get("messages", [])
        last = messages[-1] if messages else None

        if isinstance(last, AIMessage) and (last.content or "").strip():
            text = last.content
        else:
            recent = []
            for m in reversed(messages):
                if isinstance(m, ToolMessage):
                    recent.append(m.content)
                elif recent:
                    break
            text = "\n\n".join(reversed(recent)) if recent else "I don't have a response for that."

        response = {"text": text, "json_format": None}

        annotation_resp = state.get("annotation_response")
        if annotation_resp and annotation_resp.get("json_format"):
            response["json_format"] = annotation_resp["json_format"]
            response["organism"] = annotation_resp.get("organism")

        hypothesis_resp = state.get("hypothesis_response") or {}
        if isinstance(hypothesis_resp.get("resource"), dict):
            response["resource"] = hypothesis_resp["resource"]
        elif state.get("resource"):
            response["resource"] = state.get("resource")

        footer = self._build_sources_footer(state)
        if footer:
            response["text"] = response["text"].rstrip() + "\n\n" + footer

        if state.get("error"):
            response.setdefault("status", "error")

        logger.info(f"Finalizing response for user: {user_id}")
        emit_to_user(user=user_id, message=response, status="completed")
        return {"response": response}
