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

            support_lines = []
            for ps in supporting:
                title = pmid_to_title.get(str(ps.get("pmid", "")), ps.get("pmid", "?"))
                rationale = ps.get("rationale", "")
                line = f"  - {title}"
                if rationale:
                    line += f" — {rationale}"
                support_lines.append(line)

            oppose_lines = []
            for ps in opposing:
                title = pmid_to_title.get(str(ps.get("pmid", "")), ps.get("pmid", "?"))
                rationale = ps.get("rationale", "")
                line = f"  - {title}"
                if rationale:
                    line += f" — {rationale}"
                oppose_lines.append(line)

            warning_text = (
                "⚠️ **Contradiction detected in retrieved literature:**\n"
                f"  Papers supporting:\n" + "\n".join(support_lines) + "\n"
                f"  Papers opposing:\n" + "\n".join(oppose_lines) + "\n"
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
    def extract_claim_from_annotation(annotation_response: dict) -> str | None:
        """Convert annotation structured output into a natural-language claim.

        Extracts the most concrete, falsifiable statement from the annotation's
        ``json_format`` (nodes + predicates).  Falls back to the annotation's
        ``text`` summary when predicates are absent.

        Returns
        -------
        str | None
            A concise claim such as "TP53 interacts with MDM2", or ``None`` if
            no meaningful claim can be derived.
        """
        if not annotation_response:
            return None

        json_format = annotation_response.get("json_format")
        if not json_format:
            return annotation_response.get("text") or None

        predicates = json_format.get("predicates", [])
        nodes = json_format.get("nodes", [])

        if not predicates:
            # No relationships — only node lookups; not a falsifiable claim.
            return annotation_response.get("text") or None

        # Build a node_id → human-readable label map
        node_labels: dict[str, str] = {}
        for n in nodes:
            nid = n.get("node_id", "")
            props = n.get("properties", {})
            # Prefer 'name', then 'id', then first prop value, then node_id
            label = (
                props.get("name")
                or props.get("id")
                or next(iter(props.values()), None)
                or nid
            )
            node_labels[nid] = str(label)

        # Convert the first few predicates into "source relationship target"
        claims = []
        for pred in predicates[:3]:  # cap at 3 to keep the claim concise
            src = node_labels.get(pred.get("source", ""), pred.get("source", ""))
            tgt = node_labels.get(pred.get("target", ""), pred.get("target", ""))
            rel = pred.get("type", "relates to").replace("_", " ")
            if src and tgt:
                claims.append(f"{src} {rel} {tgt}")

        return "; ".join(claims) if claims else (annotation_response.get("text") or None)

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
