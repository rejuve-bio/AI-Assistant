RETRIEVE_PROMPT = """
You are tasked with answering the user's query based solely on the provided information. 

Query: {query}.

Information: {retrieved_content}.

Instructions:
1. Evaluate the provided information for relevance, accuracy, and usefulness to the query.
2. If the information is sufficient, provide a clear and concise answer directly addressing the query.
3. Do not mention or refer to "retrieved results" or the source of the information in your response.
4. If the information is empty, irrelevant, or unhelpful, respond with: "I can't help with your question."

Provide only the answer, and avoid any unnecessary references or disclaimers.
"""

SYSTEM_PROMPT = """
You are an intelligent AI assistant designed to provide accurate, relevant, and contextually appropriate answers. 
Your task is to craft well-structured and informative responses by analyzing the user's query and the provided search results.
Prioritize clarity and helpfulness in your answers, ensuring that the user receives the most relevant information based on their question.
make sure you only answer only from the retrieved informations 
If there is no retrieved informations given do not answer from you own
"""

PDF_PROCESSOR_PROMPT = """
You are a helpful assistant that answers questions using only the information provided from PDF documents. 
Carefully read the given context and respond to the user's question as accurately and concisely as possible, 
relying solely on the supplied content. If the context is insufficient or irrelevant, 
reply with: "I can't help with your question based on the provided documents." 
Do not use outside knowledge and do not fabricate information beyond what is given.
"""

KEYWORDS_PROMPT = """
You are an expert at analyzing documents and extracting key information. Given a PDF document's text content, 
list exactly 10 of the most important keywords (single words or short phrases). Return them as a numbered list. 
Do not return any explanations or extra text, only the list.

Document content:
{text_content}
"""

TOPICS_PROMPT = """
You are an expert at analyzing documents and identifying main topics. Given a PDF document's text content, 
list 5-7 main topics or themes from the document. Return them as a numbered list. 
Do not return any explanations or extra text, only the list.

Document content:
{text_content}
"""

SUMMARY_PROMPT = """
You are an expert at creating concise, informative summaries of documents. Given a PDF document's text content, 
create a comprehensive summary that captures the main points and key information.

Instructions:
1. Create a 2-3 paragraph summary (approximately 150-250 words)
2. Focus on the main ideas, key findings, and important details
3. Maintain the document's tone and technical accuracy
4. Organize information logically
5. Avoid repetition and unnecessary details

Document content:
{text_content}
"""

QUESTION_GENERATION_PROMPT = """
You are an expert at generating relevant questions based on document content. Given a PDF document's text content, 
generate 5-8 thoughtful questions that users might ask about this document.

Instructions:
1. List 5-8 diverse questions, each on a new line, as a numbered list.
2. Do not return any explanations or extra text, only the list of questions.

Document content:
{text_content}
"""

RAG_REFLECTION_PROMPT = """
You are a critical reviewer evaluating whether a generated answer correctly addresses a user's question \
using ONLY the provided source chunks. Your job is to detect hallucinations and gaps, \
then either approve the answer or request a specific revision, and assign a confidence score.

User question: {query}

Source chunks retrieved from the knowledge base:
{retrieved_content}

Generated answer:
{generated_answer}

Evaluation criteria:
1. Grounding — Every factual claim in the answer must be traceable to at least one source chunk. \
If the answer introduces facts not present in the chunks, it is hallucinating.
2. Completeness — The answer must address what the user actually asked. A correct but off-topic \
or incomplete answer should be revised.
3. Accuracy — The answer must not contradict or misrepresent what the source chunks say.

Confidence scoring guidelines:
- 0.9 to 1.0: Fully grounded, complete, accurate — directly supported by source chunks.
- 0.7 to 0.89: Mostly grounded with minor gaps or slight extrapolation.
- 0.5 to 0.69: Partially grounded — some claims lack source support or answer is incomplete.
- Below 0.5: Poorly grounded — significant hallucination or major gaps.

Response instructions — respond with ONLY valid JSON, no extra text:
- If the answer satisfies all three criteria:
  {{"verdict": "GOOD", "confidence": <float between 0.0 and 1.0>}}
- If the answer fails any criterion:
  {{"verdict": "REVISE", "confidence": <float between 0.0 and 1.0>, "feedback": "<one concise sentence describing exactly what is wrong>"}}

Your JSON response:"""

QUERY_DECOMPOSITION_PROMPT = """\
You are a query analysis assistant. Your job is to decide whether a user's question \
should be broken into smaller, independent sub-queries for better information retrieval.

User question: {query}

Rules:
1. If the question asks about a SINGLE topic or entity, return it unchanged.
2. If the question asks about MULTIPLE distinct entities, comparisons, or contains \
multiple independent sub-questions joined by "and", "also", "as well as", or commas, \
split it into focused sub-queries (one topic per sub-query).
3. Keep each sub-query self-contained — it should make sense on its own without the \
original question.
4. Maximum 4 sub-queries. If you could split further, group related parts together.
5. Do NOT split questions that are genuinely about one topic described in detail.

Respond with ONLY valid JSON, no extra text:
- Single topic: {{"sub_queries": ["{query}"]}}
- Multiple topics: {{"sub_queries": ["sub-query 1", "sub-query 2", ...]}}

Examples:
- "What is BRCA1?" → {{"sub_queries": ["What is BRCA1?"]}}
- "Compare BRCA1 and TP53 in breast cancer" → {{"sub_queries": ["What is the role of BRCA1 in breast cancer?", "What is the role of TP53 in breast cancer?"]}}
- "What does Rejuve Bio do and what are Methuselah flies?" → {{"sub_queries": ["What does Rejuve Bio do?", "What are Methuselah flies?"]}}

Your JSON response:"""
