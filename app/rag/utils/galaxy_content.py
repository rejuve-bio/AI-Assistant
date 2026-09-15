"""Galaxy-specific content handling.

Galaxy tool pages are noisier than ordinary web sources: they embed Google Chart
API images, chart data arrays and wide parameter tables, none of which survive
embedding usefully. Extraction, chunking and storage stay in ContentProcessor /
RAG -- only the parts that are genuinely Galaxy-shaped live here.
"""
import logging
import re

logger = logging.getLogger(__name__)

# Galaxy tool pages are public reference material, so they are embedded once
# into a shared collection rather than per-user.
GALAXY_COLLECTION = "1_AI_ASSISTANT_GALAXY_DATASETS"


# Applied in order. Each entry is (pattern, replacement).
_NOISE_PATTERNS = [
    # Google Chart API images and markdown image leftovers
    (r"http://chart\.apis\.google\.com[^\s\)]+", ""),
    (r"!\[\]\([^)]+\)", ""),
    # HTML entities
    (r"&nbsp;?", " "),
    (r"&[a-zA-Z0-9#]+;", ""),
    # Chart data arrays and parameter codes
    (r"chd=e:[A-Za-z0-9\.]+", ""),
    (r"ch[a-z]{2,3}=[^\s&]+", ""),
    # Table formatting debris
    (r"\|+", "|"),
    (r"(\|\s*\|){2,}", "| "),
    (r"\|\s*-+\s*\|", ""),
    # Rows that are only single letters or only digits separated by pipes
    (r"^[A-Z]{1,3}(?:\s*\|\s*[A-Z]{1,3})+\s*$", ""),
    (r"^\d+\s*\|(?:\s*\d+\s*\|?)+$", ""),
    (r"^[-\s\|]+$", ""),
    # Rules and separators
    (r"-{3,}", ""),
    (r"_{3,}", ""),
    (r"\*{3,}", ""),
    # Position/count arrays emitted by Galaxy's plotting tools
    (r"Position,\d+(?:,\d+)*", ""),
    (r"Count,\d+(?:,\d+)*", ""),
    # Whitespace
    (r"\n\s*\n\s*\n+", "\n\n"),
    (r"[ \t]+", " "),
    (r"\n +", "\n"),
]

# A line with less than this share of alphanumeric characters is table debris
# rather than prose, and only adds noise to the embedding.
_MIN_ALNUM_RATIO = 0.3


def clean_galaxy_text(text: str) -> str:
    """Strip Galaxy tool-page noise so the remaining prose embeds cleanly."""
    if not text:
        return ""

    for pattern, replacement in _NOISE_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        alnum = sum(character.isalnum() for character in line)
        if alnum / len(stripped) > _MIN_ALNUM_RATIO:
            kept.append(line)

    return "\n".join(kept).strip()


_SUMMARY_PROMPT = """
You are analyzing scientific/technical content. Create a comprehensive, detailed summary.
This summary will be embedded and used to answer questions about the document, so
preserve ALL important details.

REQUIREMENTS:
1. Capture ALL important details, numbers, statistics, findings
2. Include specific values, percentages, counts mentioned
3. Preserve technical terms and methodology details
4. List all key topics and concepts discussed
5. Keep the summary well-structured and clear

Return ONLY valid JSON (no markdown, no extra text) with this EXACT structure:
{{"summary": "detailed multi-sentence summary capturing all key information and specific values"}}

TEXT TO ANALYZE:
{text}

Previous summary from the preceding chunk (for context):
{previous_summary}

Remember: Return ONLY the JSON object, nothing else.
"""


def summarize_chunk(llm, text, previous_summary=None):
    """Summarize one chunk, threading the previous chunk's summary as context.

    Falls back to the chunk's opening text if the model returns something
    unparseable -- an imperfect summary still beats losing the chunk.
    """
    import json

    prompt = _SUMMARY_PROMPT.format(
        text=text,
        previous_summary=previous_summary if previous_summary else "N/A",
    )

    try:
        result = llm.generate(prompt)

        if isinstance(result, str):
            result = re.sub(r"```(?:json)?\s*", "", result).strip()
            result = json.loads(result)

        if isinstance(result, dict) and "summary" in result:
            return result["summary"]
        return str(result)

    except Exception as e:
        logger.warning(f"Summary generation failed, falling back to raw text: {e}")
        return text[:500].replace("\n", " ").strip()
