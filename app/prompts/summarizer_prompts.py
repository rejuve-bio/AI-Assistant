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

GRAPH_QUERY_FROM_TEXT_PROMPT = (
    "Based on this graph data:\n{graph_text}\n\n"
    "Question: {query}\n\n"
    "If this graph data does not actually contain the specific entity or "
    "information the question asks about, begin your reply with the exact "
    "tag NOT_IN_GRAPH: followed by a brief note — do NOT answer using "
    "unrelated entities from the data as if they were relevant. Otherwise, "
    "answer the question using the real data -- but don't stop at just "
    "restating node/edge counts and names. Point out what's actually "
    "notable: a hub node with unusually many connections, an irregular "
    "structure (e.g. one transcript with far more exons than the others), "
    "or a real pattern worth flagging. If you genuinely recognize the "
    "specific gene/entity from your own general knowledge, you may add a "
    "brief, clearly-labeled note on why it's biologically relevant (e.g. "
    "known family, function, disease association) -- but label it plainly "
    "as general background, never as something the graph itself showed, "
    "and only state it if you're actually confident, not a guess.\n"
    "Answer:"
)

GRAPH_QUERY_FROM_SUMMARY_PROMPT = (
    "Based on this graph data:\n"
    "Summary: {summary}\n"
    "Total nodes: {node_count}\n"
    "Total edges: {edge_count}\n"
    "Node types: {node_count_by_label}\n"
    "Edge types: {edge_count_by_label}\n\n"
    "Question: {query}\n\n"
    "If this graph data does not actually contain the specific entity or "
    "information the question asks about, begin your reply with the exact "
    "tag NOT_IN_GRAPH: followed by a brief note. Otherwise, answer the "
    "question using the real data -- but don't stop at just restating the "
    "counts above. Point out what's actually notable (a dominant node type, "
    "an unusual ratio of nodes to edges, etc.). If you genuinely recognize "
    "a specific gene/entity from your own general knowledge, you may add a "
    "brief, clearly-labeled note on why it's biologically relevant -- but "
    "label it plainly as general background, never as something this graph "
    "itself showed, and only state it if you're actually confident.\n"
    "with no tag.\nAnswer:"
)

KG_OVERVIEW_QUERY_PROMPT = (
    "Based on this knowledge graph overview data:\n{info_json}\n\n"
    "Question: {query}\n\n"
    "Answer using only the numbers/facts present in this data. If the data "
    "doesn't contain what's asked, say so plainly instead of guessing.\nAnswer:"
)
