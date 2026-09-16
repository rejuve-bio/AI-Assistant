
import logging
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from app.agents.state import KNOWLEDGE_BASE
from app.socket_manager import emit_to_user

logger = logging.getLogger(__name__)

RAG = "rag"
PUBMED = "pubmed"
CLINICAL_TRIALS = "clinical_trials"
_MIN_USEFUL_ANSWER = 120
_NO_RESULT_PHRASES = (
    "couldn't find", "could not find", "no relevant", "no information",
    "no results", "not found", "no documents", "unable to find",
    "no data", "i don't have information", "i do not have",
    "no specific", "no details",
)


class LiteratureState(TypedDict, total=False):
    # in
    user_query: str
    user_id: str
    content_ids: Optional[List[str]]
    sources: List[str]
    search_context: str
    min_year: Optional[int]
    rag_response: Optional[Dict[str, Any]]
    pubmed_response: Optional[Dict[str, Any]]
    clinical_trials_response: Optional[Dict[str, Any]]
    agents_completed: Annotated[List[str], operator.add]
    messages: Annotated[List[Any], operator.add]


def _found_nothing(text: str) -> bool:
    stripped = (text or "").lower().strip()
    return len(stripped) < _MIN_USEFUL_ANSWER or any(
        phrase in stripped for phrase in _NO_RESULT_PHRASES
    )


def build_literature_subgraph(rag, extract_search_term):

    def rag_node(state: LiteratureState) -> Dict[str, Any]:
        user_id = state["user_id"]
        emit_to_user(user=user_id, message="Retrieving information...")
        try:
            result = rag.get_result_from_rag(
                state["user_query"], user_id, content_ids=state.get("content_ids")
            )
            text = result["text"] if isinstance(result, dict) and "text" in result else (
                str(result) if result else ""
            )
            logger.debug(f"RAG response: {text}")
        except Exception as e:
            logger.error("Error in RAG node", exc_info=True)
            text = f"Error: {e}"

        return {
            "rag_response": {"text": text, "json_format": None, "source": KNOWLEDGE_BASE},
            "agents_completed": ["rag_agent"],
            "messages": [AIMessage(content="RAG query processed")],
        }

    def pubmed_node(state: LiteratureState) -> Dict[str, Any]:
        from app.rag.literature import search_pubmed

        user_id = state["user_id"]
        term = extract_search_term(state["user_query"], context=state.get("search_context", ""))
        min_year = state.get("min_year")
        logger.info(f"PubMed searching for: {term} (min_year={min_year})")
        emit_to_user(user=user_id, message="Searching PubMed literature...")
        try:
            papers = search_pubmed(term, max_results=8, min_year=min_year).get("papers", [])
            text = _format_papers(papers)
        except Exception as e:
            logger.error(f"PubMed error: {e}", exc_info=True)
            papers, text = [], f"PubMed search unavailable: {e}"

        return {
            "pubmed_response": {"text": text, "source": "PubMed", "items": papers},
            "agents_completed": ["pubmed_agent"],
            "messages": [AIMessage(content="PubMed search completed")],
        }

    def clinical_trials_node(state: LiteratureState) -> Dict[str, Any]:
        from app.rag.literature import search_clinical_trials

        user_id = state["user_id"]
        term = extract_search_term(state["user_query"], context=state.get("search_context", ""))
        logger.info(f"ClinicalTrials searching for: {term}")
        emit_to_user(user=user_id, message="Searching ClinicalTrials.gov...")
        try:
            trials = search_clinical_trials(term, status="RECRUITING", max_results=5).get("trials", [])
            if not trials:
                trials = search_clinical_trials(term, status="", max_results=5).get("trials", [])
            pubmed = state.get("pubmed_response")
            standing_in = pubmed is not None and not pubmed.get("items")
            text = _format_trials(trials, standing_in=standing_in)
        except Exception as e:
            logger.error(f"ClinicalTrials error: {e}", exc_info=True)
            trials, text = [], f"ClinicalTrials search unavailable: {e}"

        return {
            "clinical_trials_response": {
                "text": text, "source": "ClinicalTrials.gov", "items": trials
            },
            "agents_completed": ["clinical_trials_agent"],
            "messages": [AIMessage(content="ClinicalTrials search completed")],
        }


    def entry(state: LiteratureState) -> str:
        """Start at the first requested source, in RAG → PubMed → trials order."""
        sources = state.get("sources") or [RAG]
        for node in (RAG, PUBMED, CLINICAL_TRIALS):
            if node in sources:
                return node
        return RAG

    def after_rag(state: LiteratureState) -> str:
        """PubMed runs when it was asked for, or as a fallback when RAG found nothing."""
        sources = state.get("sources") or []
        if PUBMED in sources:
            return PUBMED
        if _found_nothing((state.get("rag_response") or {}).get("text", "")):
            logger.info("RAG found nothing — falling back to PubMed")
            emit_to_user(
                user=state["user_id"],
                message="Nothing found in knowledge base, searching PubMed...",
            )
            return PUBMED
        return END

    def after_pubmed(state: LiteratureState) -> str:
        if CLINICAL_TRIALS in (state.get("sources") or []):
            return CLINICAL_TRIALS
        if not (state.get("pubmed_response") or {}).get("items"):
            logger.info("PubMed found nothing — trying ClinicalTrials")
            return CLINICAL_TRIALS
        return END

    graph = StateGraph(LiteratureState)
    graph.add_node(RAG, rag_node)
    graph.add_node(PUBMED, pubmed_node)
    graph.add_node(CLINICAL_TRIALS, clinical_trials_node)

    graph.set_conditional_entry_point(
        entry, {RAG: RAG, PUBMED: PUBMED, CLINICAL_TRIALS: CLINICAL_TRIALS}
    )
    graph.add_conditional_edges(RAG, after_rag, {PUBMED: PUBMED, END: END})
    graph.add_conditional_edges(PUBMED, after_pubmed, {CLINICAL_TRIALS: CLINICAL_TRIALS, END: END})
    graph.add_edge(CLINICAL_TRIALS, END)

    return graph.compile()


def _format_papers(papers: list) -> str:
    if not papers:
        return "No relevant publications found in PubMed for this query."
    lines = [f"Found {len(papers)} relevant paper(s) from PubMed:\n"]
    for paper in papers:
        authors = ", ".join(paper.get("authors", [])) or "Unknown authors"
        lines.append(
            f"- **{paper.get('title', 'No title')}** ({paper.get('year', '')}) — {authors}\n"
            f"  {paper.get('abstract', '')}\n"
            f"  URL: {paper.get('url', '')}"
        )
    return "\n".join(lines)


def _format_trials(trials: list, standing_in: bool = False) -> str:
    if not trials:
        return (
            "Nothing found in PubMed or ClinicalTrials.gov for this query."
            if standing_in
            else "No clinical trials found for this query on ClinicalTrials.gov."
        )
    heading = (
        f"Nothing in PubMed for this — {len(trials)} related clinical trial(s):\n"
        if standing_in
        else f"Found {len(trials)} clinical trial(s) on ClinicalTrials.gov:\n"
    )
    lines = [heading]
    for trial in trials:
        phase = ", ".join(trial.get("phase", [])) or "N/A"
        conditions = ", ".join(trial.get("conditions", [])) or "N/A"
        interventions = ", ".join(trial.get("interventions", [])) or "N/A"
        lines.append(
            f"- **{trial.get('title', 'No title')}** ({trial.get('nct_id', '')})\n"
            f"  Phase: {phase} | Status: {trial.get('status', '')} | "
            f"Started: {trial.get('start_date', 'N/A')}\n"
            f"  Conditions: {conditions}\n"
            f"  Interventions: {interventions}\n"
            f"  URL: {trial.get('url', '')}"
        )
    return "\n".join(lines)
