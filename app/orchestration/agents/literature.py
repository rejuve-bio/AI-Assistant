"""PubMed and ClinicalTrials.gov research agents."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from app.rag.literature import search_clinical_trials, search_pubmed

from ..contracts import AgentState
from .dependencies import AgentDependencies

logger = logging.getLogger(__name__)


class _LiteratureAgent:
    def __init__(self, dependencies: AgentDependencies) -> None:
        self._basic_llm = dependencies.basic_llm
        self._emit_status = dependencies.emit_status

    def _search_term(self, state: AgentState) -> str:
        context = (state.get("hypothesis_response") or {}).get("text", "")
        prompt = (
            "Extract a 3-7 word search term for PubMed or ClinicalTrials.gov. Focus on the biological topic, gene, drug, or condition. "
            "Return only the search term.\n\n"
            f"User question: {state['user_query']}\nAdditional context: {context[:500]}\n\nSearch term:"
        )
        try:
            response = self._basic_llm.generate(prompt)
            term = response.strip().strip('"').strip("'") if isinstance(response, str) else ""
            return term or state["user_query"]
        except Exception:
            return state["user_query"]


class PubMedAgent(_LiteratureAgent):
    def execute(self, state: AgentState) -> dict[str, Any]:
        try:
            self._emit_status(user=state["user_id"], message="Searching PubMed literature...")
            papers = search_pubmed(self._search_term(state), max_results=8).get("papers", [])
            text = self._format_papers(papers)
            return {"pubmed_response": {"text": text, "source": "PubMed", "items": papers}, "agents_completed": ["pubmed_agent"], "messages": [AIMessage(content="PubMed search completed")]}
        except Exception as exc:
            logger.exception("PubMed agent failed")
            return {"pubmed_response": {"text": f"PubMed search unavailable: {exc}", "source": "PubMed", "items": []}, "agents_completed": ["pubmed_agent"]}

    @staticmethod
    def _format_papers(papers: list[dict[str, Any]]) -> str:
        if not papers:
            return "No relevant publications found in PubMed for this query."
        lines = [f"Found {len(papers)} relevant paper(s) from PubMed:\n"]
        for paper in papers:
            lines.append(f"- **{paper.get('title', 'No title')}** ({paper.get('year', '')}) — {', '.join(paper.get('authors', [])) or 'Unknown authors'}\n  {paper.get('abstract', '')}\n  URL: {paper.get('url', '')}")
        return "\n".join(lines)


class ClinicalTrialsAgent(_LiteratureAgent):
    def execute(self, state: AgentState) -> dict[str, Any]:
        try:
            self._emit_status(user=state["user_id"], message="Searching ClinicalTrials.gov...")
            term = self._search_term(state)
            trials = search_clinical_trials(term, status="RECRUITING", max_results=5).get("trials", [])
            if not trials:
                trials = search_clinical_trials(term, status="", max_results=5).get("trials", [])
            return {"clinical_trials_response": {"text": self._format_trials(trials), "source": "ClinicalTrials.gov", "items": trials}, "agents_completed": ["clinical_trials_agent"], "messages": [AIMessage(content="ClinicalTrials search completed")]}
        except Exception as exc:
            logger.exception("ClinicalTrials agent failed")
            return {"clinical_trials_response": {"text": f"ClinicalTrials search unavailable: {exc}", "source": "ClinicalTrials.gov", "items": []}, "agents_completed": ["clinical_trials_agent"]}

    @staticmethod
    def _format_trials(trials: list[dict[str, Any]]) -> str:
        if not trials:
            return "No clinical trials found for this query on ClinicalTrials.gov."
        lines = [f"Found {len(trials)} clinical trial(s) on ClinicalTrials.gov:\n"]
        for trial in trials:
            lines.append(f"- **{trial.get('title', 'No title')}** ({trial.get('nct_id', '')})\n  Phase: {', '.join(trial.get('phase', [])) or 'N/A'} | Status: {trial.get('status', '')}\n  Conditions: {', '.join(trial.get('conditions', [])) or 'N/A'}\n  URL: {trial.get('url', '')}")
        return "\n".join(lines)
