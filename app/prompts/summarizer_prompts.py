SUMMARY_PROMPT_BASED_ON_USER_QUERY = """
                                You are an expert biology assistant on summarizing graph data.\n\n
                                User Query: {user_query}\n\n"
                                Given the following data visualization:\n{description}\n\n"
                                Your task is to analyze the graph and summarize the most important trends, patterns, and relationships.\n
                                Instructions:\n"
                                - Focus on identifying key trends, relationships, or anomalies directly related to the user's question.\n
                                - Highlight specific comparisons (if applicable) or variables shown in the graph.\n
                                - Use bullet points or numbered lists to break down core details when necessary.\n
                                - Format the response in a clear, concise, and easy-to-read manner.\n\n
                                Please provide a summary based solely on the information shown in the graph.
                                """
SUMMARY_PROMPT_CHUNKING = """"""
SUMMARY_PROMPT_CHUNKING_USER_QUERY =""""""

                
SUMMARY_PROMPT = """
                You are an expert biology assistant on summarizing graph data.\n\n
                Given the following graph data:\n{description}\n\n
                Your task is to analyze and summarize the most important trends, patterns, and relationships.\n
                Instructions:\n
                - Identify key trends, relationships.\n
                - Use bullet points or numbered lists to break down core details when necessary.\n
                - Format the response clearly and concisely.\n\n
                Count and list important metrics
                Identify any central nodes or relationships and highlight any important patterns.
                Also, mention key relationships between nodes and any interesting structures (such as chains or hubs).
                Please provide a summary based solely on the graph information.
                Start with: 'The graph shows:'
                """
            
