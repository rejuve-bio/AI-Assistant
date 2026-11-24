LITERATURE_CONSENSUS_PROMPT = """\
You are a scientific evidence analyst. You will be given a **claim** and a list of **evidence items** -- \
each is either a PubMed paper (title + abstract) or a ClinicalTrials.gov trial (title, status, conditions, \
interventions).

For each item, classify its stance toward the claim:
- **SUPPORT** — the item presents evidence that directly supports the claim
- **OPPOSE** — the item presents evidence that contradicts or challenges the claim
- **INCONCLUSIVE** — the item is related but results are mixed, neutral, or insufficient to judge

A clinical trial with no posted results (e.g. still recruiting, or only its design/status is known, not an \
outcome) cannot be classified SUPPORT or OPPOSE -- its existence shows the question is being studied, not which \
way the evidence points. Classify it INCONCLUSIVE regardless of its title or status.

Then provide an overall consensus across all evidence items.

## Claim to evaluate:
{claim}

## Evidence:
{evidence_block}

## Response format — respond with ONLY valid JSON, no extra text:
{{
  "evidence_stances": [
    {{"id": "...", "source": "PUBMED|CLINICAL_TRIAL", "stance": "SUPPORT|OPPOSE|INCONCLUSIVE", "rationale": "one sentence why"}}
  ],
  "consensus": {{
    "label": "STRONGLY_SUPPORTED|SUPPORTED|CONTESTED|INCONCLUSIVE|CONTRADICTED",
    "support_count": 0,
    "oppose_count": 0,
    "inconclusive_count": 0,
    "summary": "one sentence overall assessment"
  }}
}}

## Consensus label rules:
- STRONGLY_SUPPORTED: ≥3 items SUPPORT, 0 OPPOSE
- SUPPORTED: majority SUPPORT, ≤1 OPPOSE
- CONTESTED: ≥2 SUPPORT and ≥2 OPPOSE
- INCONCLUSIVE: mostly INCONCLUSIVE or too few items
- CONTRADICTED: majority OPPOSE

Your JSON response:"""

