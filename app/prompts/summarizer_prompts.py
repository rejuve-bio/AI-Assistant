SUMMARY_PROMPT_BASED_ON_USER_QUERY = """
                                You are an expert biology assistant on summarizing graph data.\n\n
                                User Query: {user_query}\n\n"
                                Given the following data visualization:\n{description}\n\n"
                                Your task is to analyze the graph and summarize the most important trends, patterns, and relationships.\n
                                Instructions:\n"
                                - Focus on identifying key trends, relationships, or anomalies directly related to the user's question.\n
                                - Highlight specific comparisons (if applicable) or variables shown in the graph.\n
                                - Format the response in a clear, concise, and easy-to-read manner.\n\n
                                Please provide a summary based solely on the information shown in the graph.
                                Addressed with clear and concise descriptions. Make sure not to use bullet points or numbered lists, but instead focus on delivering the content in paragraph form for the user question
                                """
SUMMARY_PROMPT_CHUNKING = """
                You are an expert biology assistant on summarizing graph data.\n\n
                Given the following graph data:\n{description}\n\n
                Given the following previous summary:\n{prev_summery}\n\n"
                Your task is to analyze the graph ,including the previous summary and summarize the most important trends, patterns, and relationships.\n
                Instructions:\n
                  - Count and list important metrics, such as the number of nodes and edges.    
                  - Identify any central nodes and explain their role in the network.     
                  - Mention any notable structures in the graph, such as chains, hubs, or clusters.      
                  - Discuss any specific characteristics of the data, such as alternative splicing or regulatory mechanisms that may be involved.     
                  - Format the response clearly and concisely.\n\n
                Count and list important metrics
                Identify any central nodes or relationships and highlight any important patterns.
                Also, mention key relationships between nodes and any interesting structures (such as chains or hubs).
                Please provide a summary based solely on the graph information.
                Addressed points in a separate paragraph, with clear and concise descriptions. Make sure not to use bullet points or numbered lists, but instead focus on delivering the content in paragraph form.
 """
SUMMARY_PROMPT_CHUNKING_USER_QUERY ="""
 You are an expert biology assistant on summarizing graph data.\n\n
                                User Query: {user_query}\n\n"
                                Given the following data visualization:\n{description}\n\n" 
                                Given the following previous summary:\n{prev_summery}\n\n"
                                Your task is to analyze the graph ,including the previous summary and summarize the most important trends, patterns, and relationships.\n
                                Instructions:\n"
                                - Focus on identifying key trends, relationships, or anomalies directly related to the user's question.\n
                                - Highlight specific comparisons (if applicable) or variables shown in the graph.\n
                                - Format the response in a clear, concise, and easy-to-read manner.\n\n
                                Please provide a summary based solely on the information shown in the graph.
                                Addressed with clear and concise descriptions. Make sure not to use bullet points or numbered lists, but instead focus on delivering the content in paragraph form for the user question
                             """


SUMMARY_PROMPT = """
You are an expert biology assistant on summarizing graph data.

Given the following graph data:
{description}

Your task is to analyze and summarize the most important trends, patterns, and relationships in a list of paragraphs. 
Each paragraph should address one of the following points:
- Identify key trends and relationships in the graph data.
- Count and list important metrics, such as the number of nodes and edges.
- Identify any central nodes and explain their role in the network.
- Mention any notable structures in the graph, such as chains, hubs, or clusters.
- Discuss any specific characteristics of the data, such as alternative splicing or regulatory mechanisms that may be involved.
- Explain any notable relationships, including nodes that have a higher number of associated related nodes or complex processes.
Addressed points in a separate paragraph, with clear and concise descriptions. Make sure not to use bullet points or numbered lists, but instead focus on delivering the content in paragraph form.
"""


GRAPH_BIOLOGICAL_INSIGHT_PROMPT = """
You are a biomedical knowledge-graph interpreter writing for a researcher.

User request: {user_query}

Use ONLY this parsed graph summary:
{parsed_graph_json}

Write a detailed 2-3 paragraph explanation focused on biological interpretation.

Requirements:
- Do NOT describe raw graph structure, UI elements, or visualization layout.
- Prioritize biological insight: what the relationships may imply mechanistically.
- Explicitly include important categories/types, including coding vs non-coding patterns when available.
- If transcript subtype distribution is present (for example protein_coding, nonsense_mediated_decay, protein_coding_lof), interpret what it may indicate.
- Include practical exploration guidance (where to start, what path to follow first).
- Include at least one concrete example identifier from the provided sample IDs.
- Use cautious scientific language (for example: suggests, may indicate, is consistent with).
- Avoid bullet points and avoid symbolic edge notation.

Return plain text only.
"""


GRAPH_ID_BIOLOGICAL_QA_PROMPT = """
You are a biomedical graph analyst.

Graph biological summary:
{graph_summary}

Category counts:
{node_count_by_label}

Relationship counts:
{edge_count_by_label}

User request: {user_query}

Answer in 1-2 paragraphs with biological insight.

Requirements:
- Do NOT describe raw graph structure.
- Explain biological meaning, not only visible counts.
- Mention category/type interpretation, including coding vs non-coding implications when relevant.
- Use cautious language where appropriate.

Return plain text only.
"""
