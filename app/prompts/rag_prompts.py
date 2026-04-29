RETRIEVE_PROMPT = """
Answer the query using only the provided information. Be direct and brief — the audience is a biomedical researcher who does not need background explained.

Query: {query}
Information: {retrieved_content}

- Answer in 2-4 sentences maximum unless the query requires a structured list.
- If the information is insufficient or irrelevant, respond with exactly: "I can't help with your question."
- Do not restate the question. Do not add disclaimers or source references.
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

CLARIFYING_QUESTIONS_PROMPT = """
A biomedical researcher just received this response. Suggest 3-4 follow-up actions they would actually take next.

User query: {user_query}
Response: {assistant_response}

Rules:
- Each suggestion must be specific and directly actionable given what was just returned.
- Prioritize: validating findings, running a related analysis, comparing with another dataset or method, drilling into a specific result.
- Do NOT generate generic questions like "Can you explain X?" or "What is Y?" — the researcher already knows the basics.
- Frame as questions the researcher would type into this system (e.g. "Run DESeq2 on the filtered genes from this result", "Search PubMed for rs1421085 obesity studies").
- Return ONLY a numbered list, no preamble.
"""
