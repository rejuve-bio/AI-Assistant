DEPENDENCY_SUMMARIZATION_PROMPT = """
You are a scientific summarizer. Condense the following agent output into a concise summary
(maximum 400 words) that preserves all key biological findings, numbers, gene names,
variant IDs, pathway names, and actionable results.

Remove verbose explanations, repeated information, and formatting artifacts.
Keep all specific identifiers (rs####, gene names, p-values, file paths, numerical results) intact.

Content to summarize:
{content}

Provide only the summary, no preamble.
"""
