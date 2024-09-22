# Handles graph-related operations like processing nodes, edges, generating responses ...

from llm_handler import Gemini
from helpers import schema_description, sample_json_format,sample_graph
from summarizer import Graph_Summarizer
import json
import re

class QueryHandler:

    def __init__(self, query):
        self.llm = Gemini()
        self.query = query
        # enhance the query        


class Annotation_service_GraphHandler(QueryHandler):

    def __init__(self, query):
        super().__init__(query)  # Call the parent class's __init__ method
        self.schema_description = schema_description
        self.sample_json_format = sample_json_format

    def extract_nodes_and_edges(self):
        try:
            prompt = f'''
                    Given the user query
                    
                    {self.query}
                    
                    you are supposed to identify and create a graph based on the following schema and refering to the description on the schema
                    
                    {self.schema_description}

                    1. dont use the examples in the description just return the result only from the query
                    2. return the result in json format in key value pairs where the keys are the schema name and value is the user query we will set to query the graph
                    3. only return a relationship related with user question
                    3. dont explain anything just return the result in json format
                    4. return me in a cypher query format
                    '''
        
            self.nodes_and_edges = self.llm(prompt)
        except Exception as e:
            print(f"Error in extract_nodes_and_edges: {str(e)}")

    def sort_json_format(self):
        try:
            prompt = f"""
            The user question is:
            {self.query}
            Given the following extracted information from a query:
            {self.nodes_and_edges}
            Convert this information into the following JSON format:
            {self.sample_json_format}
            Please follow these rules when creating the output:
            1. For each node, include node_id, id, type, and properties fields. These are mandatory and should be the only fields for nodes.
            2. Generate node_ids by auto-incrementing for each node (n1, n2, n3, ...).
            3. If a specific ID is given in the extracted information, use it in the 'id' field. Otherwise, leave it as an empty string.
            4. Add the label of the node in the type filed
            5. Include all relevant properties from the extracted information in the properties field of each node.
            6. For predicates (relationships), include type, source, and target fields. The source and target should reference the node_ids.
            7. Ensure that the output JSON structure matches the sample format provided.
            8. Make sure all relationships in the extracted information are represented as predicates in the output.
            9. Make sure all nodes in the relation ship are available in the nodes list
            10.Dont mention id in the dicitionary
            Provide only the resulting JSON as your response, without any additional explanation or commentary.
            """
            json_format = self.llm(prompt)
            match = re.search(r'```json\n(\{.+\})\n```', json_format, re.DOTALL | re.IGNORECASE)
            cleaned_json = match.group(1) if match else None
            self.parsed_json = json.loads(cleaned_json)
        except (json.JSONDecodeError, AttributeError, Exception) as e:
            print(f"Error in sort_json_format: {str(e)}")

        try:
            prompt =f"""
                Given a json format 
                {self.parsed_json}
                if only a node is present in the json format add 
                a key "id": with a value of ""
                for the given nodes
                """
            response = self.llm(prompt)
            json_ = response.text
            match = re.search(r'```json\n(\{.+\})\n```', json_, re.DOTALL | re.IGNORECASE)
            cleaned_json = match.group(1) if match else None
            self.parsed_json = json.loads(cleaned_json)
        except:
            print(f"Error in generating json format from query: {str(e)}")
    
    def query_annotation_service(self):
        try:
            # localhost://annotation_service endpoint
            pass
        except Exception as e:
            print(f"Error in query_annotation_service: {str(e)}")

    def query_and_refactoring_json_format(self):
        try:
            self.query_annotation_service()
            # Assuming this assigns self.graph based on some results
            self.graph = "graph result here"
            self.graph = sample_graph
        except Exception as e:
            # if there is schema issue do this all over again or ... ?
            print(f"Error in query_and_refactoring_json_format: {str(e)}")


    def summarizer_prompt(self):
        try:
            summary = Graph_Summarizer()
            graph_summary = summary.summarizer_prompt(self.query, self.graph)
            return graph_summary, self.graph
        except Exception as e:
            print(f"Error in summarizer_prompt: {str(e)}")
            return None, self.graph

    def generate_annotation_result(self):
        try:
            self.extract_nodes_and_edges()
            self.sort_json_format()
            self.query_and_refactoring_json_format()
            summary = self.summarizer_prompt()
            return summary, self.graph
        except Exception as e:
            print(f"Error in generate_annotation_result: {str(e)}")
            return None


query ="What is gene ENSG00000237491 transcribed_to?"

a = Annotation_service_GraphHandler(query)
print(a.generate_annotation_result())

