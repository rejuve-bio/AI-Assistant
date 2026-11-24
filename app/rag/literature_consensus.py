
import json
import logging
import re
from typing import Optional, Dict, Any, List

from app.prompts.literature_consensus_prompt import LITERATURE_CONSENSUS_PROMPT

logger = logging.getLogger(__name__)


def format_consensus_block(consensus: Optional[Dict[str, Any]]) -> str:
    if not consensus or not consensus.get("consensus_label"):
        return ""

    label = consensus["consensus_label"]
    support = consensus.get("support_count", 0)
    oppose = consensus.get("oppose_count", 0)
    inconclusive = consensus.get("inconclusive_count", 0)
    summary = consensus.get("summary", "")

    parts = [
        f"📚 **Literature consensus: {label.replace('_', ' ').title()}** "
        f"({support} supporting, {oppose} opposing, {inconclusive} inconclusive)"
    ]
    if summary:
        parts.append(f"   {summary}")

    warning = consensus.get("warning_text")
    if warning:
        parts.append("")
        parts.append(warning)

    return "\n".join(parts)


def format_consensus_narrative(consensus: Optional[Dict[str, Any]], papers: list, trials: list) -> str:
    if not consensus or not consensus.get("evidence_stances"):
        return ""

    id_to_item: Dict[str, Dict[str, str]] = {}
    for p in papers:
        id_to_item[str(p.get("pmid", ""))] = {"title": p.get("title", "No title"), "url": p.get("url", "")}
    for t in trials:
        id_to_item[str(t.get("nct_id", ""))] = {"title": t.get("title", "No title"), "url": t.get("url", "")}

    def _bucket(stance_name: str) -> list:
        return [es for es in consensus["evidence_stances"] if str(es.get("stance", "")).upper() == stance_name]

    def _render(es_list: list) -> str:
        lines = []
        for es in es_list:
            item = id_to_item.get(str(es.get("id", "")), {})
            title = item.get("title") or es.get("id", "?")
            url = item.get("url", "")
            link = f"[{title}]({url})" if url else title
            rationale = es.get("rationale", "")
            lines.append(f"- {link}" + (f" — {rationale}" if rationale else ""))
        return "\n".join(lines)

    supporting = _bucket("SUPPORT")
    opposing = _bucket("OPPOSE")
    inconclusive = _bucket("INCONCLUSIVE")
    total = len(supporting) + len(opposing) + len(inconclusive)

    label = consensus.get("consensus_label", "INCONCLUSIVE")
    summary = consensus.get("summary", "")

    sections = [
        f"📚 **Overall: {label.replace('_', ' ').title()}**" + (f" — {summary}" if summary else ""),
        f"Found {total} piece(s) of evidence: {len(supporting)} supporting, "
        f"{len(opposing)} opposing, {len(inconclusive)} inconclusive.",
    ]

    sides = [("Research opposing this:", opposing), ("Research supporting this:", supporting)]
    if len(supporting) >= len(opposing):
        sides.reverse()
    for heading, bucket in sides:
        if bucket:
            sections.append(f"{heading}\n" + _render(bucket))

    if inconclusive:
        if supporting or opposing:
            titles = ", ".join(id_to_item.get(str(es.get("id", "")), {}).get("title") or es.get("id", "?") for es in inconclusive)
            sections.append(f"Also reviewed, but inconclusive: {titles}")
        else:
            sections.append("The evidence found is inconclusive:\n" + _render(inconclusive))

    return "\n\n".join(sections)


class LiteratureConsensusAnalyzer:

    def __init__(self, llm):
        self.llm = llm

    def analyze_consensus(self, claim: str, papers: Optional[list] = None, trials: Optional[list] = None) -> dict:
        papers = papers or []
        trials = trials or []
        evidence = self._as_evidence_items(papers, trials)
        if not evidence or not claim:
            return self._empty_result()

        evidence_block = self._format_evidence_block(evidence)

        prompt = LITERATURE_CONSENSUS_PROMPT.format(
            claim=claim,
            evidence_block=evidence_block,
        )

        try:
            raw = self.llm.generate(prompt)
            parsed = self._parse_response(raw)
            if parsed is None:
                logger.warning("Consensus LLM returned unparseable output — falling back to empty result")
                return self._empty_result()
            return self._build_result(parsed, evidence)
        except Exception as e:
            logger.error(f"Literature consensus analysis failed: {e}", exc_info=True)
            return self._empty_result()

    @staticmethod
    def _as_evidence_items(papers: list, trials: list) -> list:
        items = [
            {
                "id": p.get("pmid", "unknown"),
                "source": "PUBMED",
                "title": p.get("title", "No title"),
                "detail": f"Abstract: {p.get('abstract', 'No abstract available.')}",
            }
            for p in papers
        ]
        items += [
            {
                "id": t.get("nct_id", "unknown"),
                "source": "CLINICAL_TRIAL",
                "title": t.get("title", "No title"),
                "detail": (
                    f"Status: {t.get('status', 'unknown')} | "
                    f"Conditions: {', '.join(t.get('conditions', [])) or 'N/A'} | "
                    f"Interventions: {', '.join(t.get('interventions', [])) or 'N/A'}"
                ),
            }
            for t in trials
        ]
        return items

    @staticmethod
    def _format_evidence_block(evidence: list) -> str:
        lines = []
        for i, e in enumerate(evidence, 1):
            lines.append(
                f"Item {i} ({e['source']}, ID: {e['id']}):\n"
                f"  Title: {e['title']}\n"
                f"  {e['detail']}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _parse_response(raw) -> dict | None:
        if isinstance(raw, dict):
            return raw

        if not isinstance(raw, str):
            return None

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            match = re.search(r'\{[\s\S]+\}', cleaned)
            if match:
                try:
                    return json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    pass
        return None

    def _build_result(self, parsed: dict, evidence: list) -> dict:
        evidence_stances = parsed.get("evidence_stances", parsed.get("paper_stances", []))
        consensus = parsed.get("consensus", {})

        if evidence_stances:
            support_count = 0
            oppose_count = 0
            inconclusive_count = 0
            for es in evidence_stances:
                stance = str(es.get("stance", "")).upper()
                if stance == "SUPPORT":
                    support_count += 1
                elif stance == "OPPOSE":
                    oppose_count += 1
                else:
                    inconclusive_count += 1
        else:
            support_count = consensus.get("support_count", 0)
            oppose_count = consensus.get("oppose_count", 0)
            inconclusive_count = consensus.get("inconclusive_count", 0)

        label = consensus.get("label", "INCONCLUSIVE")
        summary = consensus.get("summary", "")

        has_contradiction = support_count >= 1 and oppose_count >= 1

        warning_text = None
        if has_contradiction:
            supporting = [es for es in evidence_stances if str(es.get("stance", "")).upper() == "SUPPORT"]
            opposing = [es for es in evidence_stances if str(es.get("stance", "")).upper() == "OPPOSE"]

            id_to_title = {str(e.get("id", "")): e.get("title", "Unknown") for e in evidence}

            def _line(es: dict) -> str:
                title = id_to_title.get(str(es.get("id", "")), es.get("id", "?"))
                rationale = es.get("rationale", "")
                return f"  - {title}" + (f" — {rationale}" if rationale else "")

            warning_text = (
                "⚠️ **Contradiction detected in retrieved evidence:**\n"
                "  Supporting:\n" + "\n".join(_line(es) for es in supporting) + "\n"
                "  Opposing:\n" + "\n".join(_line(es) for es in opposing) + "\n"
                "  Interpret results with caution."
            )

        return {
            "consensus_label": label,
            "evidence_stances": evidence_stances,
            "support_count": support_count,
            "oppose_count": oppose_count,
            "inconclusive_count": inconclusive_count,
            "has_contradiction": has_contradiction,
            "warning_text": warning_text,
            "summary": summary,
        }

    @staticmethod
    def extract_claim_from_annotation(annotation_response: dict) -> str | None:
        if not annotation_response:
            return None

        json_format = annotation_response.get("json_format")
        if not json_format:
            return annotation_response.get("text") or None

        predicates = json_format.get("predicates", [])
        nodes = json_format.get("nodes", [])

        if not predicates:
            return annotation_response.get("text") or None

        node_labels: dict[str, str] = {}
        for n in nodes:
            nid = n.get("node_id", "")
            props = n.get("properties", {}) or {}
            label = (
                props.get("name")
                or props.get("id")
                or n.get("id")
                or next(iter(props.values()), None)
                or nid
            )
            node_labels[nid] = str(label)

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
            "evidence_stances": [],
            "support_count": 0,
            "oppose_count": 0,
            "inconclusive_count": 0,
            "has_contradiction": False,
            "warning_text": None,
            "summary": "",
        }

