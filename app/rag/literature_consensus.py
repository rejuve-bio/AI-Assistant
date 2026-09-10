"""
Analyzes retrieved PubMed papers for agreement or disagreement with a claim.
Classifies each paper as SUPPORT / OPPOSE / INCONCLUSIVE and surfaces
contradictions when evidence is divided.
"""

import json
import logging
import re

from app.prompts.literature_consensus_prompt import LITERATURE_CONSENSUS_PROMPT

logger = logging.getLogger(__name__)


class LiteratureConsensusAnalyzer:
    """Analyze retrieved PubMed papers for agreement/disagreement with a claim."""

    def __init__(self, llm):
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_consensus(self, claim: str, papers: list) -> dict:
        """
        Classify all papers' stances against *claim* in a single batched LLM
        call and return a structured consensus result.

        Parameters
        ----------
        claim : str
            The scientific assertion to evaluate (e.g. hypothesis summary,
            user question, or annotation relationship).
        papers : list[dict]
            PubMed paper dicts, each with at least ``pmid``, ``title``,
            ``abstract``.

        Returns
        -------
        dict
            {
              "consensus_label": str,
              "paper_stances": [{"pmid", "stance", "rationale"}, ...],
              "support_count": int,
              "oppose_count": int,
              "inconclusive_count": int,
              "has_contradiction": bool,
              "warning_text": str | None,
              "summary": str,
            }
        """
        if not papers or not claim:
            return self._empty_result()

        # Build the papers block for the prompt
        papers_block = self._format_papers_block(papers)

        prompt = LITERATURE_CONSENSUS_PROMPT.format(
            claim=claim,
            papers_block=papers_block,
        )

        try:
            raw = self.llm.generate(prompt)
            parsed = self._parse_response(raw)
            if parsed is None:
                logger.warning("Consensus LLM returned unparseable output — falling back to empty result")
                return self._empty_result()
            return self._build_result(parsed, papers)
        except Exception as e:
            logger.error(f"Literature consensus analysis failed: {e}", exc_info=True)
            return self._empty_result()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _format_papers_block(papers: list) -> str:
        """Format papers into a numbered list for the prompt."""
        lines = []
        for i, p in enumerate(papers, 1):
            pmid = p.get("pmid", "unknown")
            title = p.get("title", "No title")
            abstract = p.get("abstract", "No abstract available.")
            lines.append(
                f"Paper {i} (PMID: {pmid}):\n"
                f"  Title: {title}\n"
                f"  Abstract: {abstract}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _parse_response(raw) -> dict | None:
        """Parse the LLM response into a dict, handling the dict-or-string quirk."""
        if isinstance(raw, dict):
            return raw

        if not isinstance(raw, str):
            return None

        cleaned = raw.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            # Try to find embedded JSON object
            match = re.search(r'\{[\s\S]+\}', cleaned)
            if match:
                try:
                    return json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    pass
        return None

    def _build_result(self, parsed: dict, papers: list) -> dict:
        """Build the final consensus result from the parsed LLM output."""
        paper_stances = parsed.get("paper_stances", [])
        consensus = parsed.get("consensus", {})

        support_count = consensus.get("support_count", 0)
        oppose_count = consensus.get("oppose_count", 0)
        inconclusive_count = consensus.get("inconclusive_count", 0)
        label = consensus.get("label", "INCONCLUSIVE")
        summary = consensus.get("summary", "")

        # If LLM didn't provide counts, derive them from stances
        if support_count == 0 and oppose_count == 0 and inconclusive_count == 0 and paper_stances:
            for ps in paper_stances:
                stance = ps.get("stance", "").upper()
                if stance == "SUPPORT":
                    support_count += 1
                elif stance == "OPPOSE":
                    oppose_count += 1
                else:
                    inconclusive_count += 1

        has_contradiction = support_count >= 1 and oppose_count >= 1

        warning_text = None
        if has_contradiction:
            supporting = [ps for ps in paper_stances if ps.get("stance", "").upper() == "SUPPORT"]
            opposing = [ps for ps in paper_stances if ps.get("stance", "").upper() == "OPPOSE"]

            # Look up titles from papers list by pmid
            pmid_to_title = {str(p.get("pmid", "")): p.get("title", "Unknown") for p in papers}

            support_titles = [pmid_to_title.get(str(ps.get("pmid", "")), ps.get("pmid", "?")) for ps in supporting]
            oppose_titles = [pmid_to_title.get(str(ps.get("pmid", "")), ps.get("pmid", "?")) for ps in opposing]

            warning_text = (
                "⚠️ **Contradiction detected in retrieved literature:**\n"
                f"  Papers supporting: {', '.join(support_titles)}\n"
                f"  Papers opposing: {', '.join(oppose_titles)}\n"
                "  Interpret results with caution."
            )

        return {
            "consensus_label": label,
            "paper_stances": paper_stances,
            "support_count": support_count,
            "oppose_count": oppose_count,
            "inconclusive_count": inconclusive_count,
            "has_contradiction": has_contradiction,
            "warning_text": warning_text,
            "summary": summary,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "consensus_label": "INCONCLUSIVE",
            "paper_stances": [],
            "support_count": 0,
            "oppose_count": 0,
            "inconclusive_count": 0,
            "has_contradiction": False,
            "warning_text": None,
            "summary": "",
        }
