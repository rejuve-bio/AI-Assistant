"""
Prompts for literature contradiction & consensus reasoning.

One batched prompt classifies all papers' stances against a claim in a single
LLM call, avoiding 1-call-per-paper overhead.
"""

LITERATURE_CONSENSUS_PROMPT = """\
You are a scientific literature analyst. You will be given a **claim** and a list of **papers** (title + abstract).

For each paper, classify its stance toward the claim:
- **SUPPORT** — the paper presents evidence that directly supports the claim
- **OPPOSE** — the paper presents evidence that contradicts or challenges the claim
- **INCONCLUSIVE** — the paper is related but results are mixed, neutral, or insufficient to judge

Then provide an overall consensus across all papers.

## Claim to evaluate:
{claim}

## Papers:
{papers_block}

## Response format — respond with ONLY valid JSON, no extra text:
{{
  "paper_stances": [
    {{"pmid": "...", "stance": "SUPPORT|OPPOSE|INCONCLUSIVE", "rationale": "one sentence why"}}
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
- STRONGLY_SUPPORTED: ≥3 papers SUPPORT, 0 OPPOSE
- SUPPORTED: majority SUPPORT, ≤1 OPPOSE
- CONTESTED: ≥2 SUPPORT and ≥2 OPPOSE
- INCONCLUSIVE: mostly INCONCLUSIVE or too few papers
- CONTRADICTED: majority OPPOSE

Your JSON response:"""
