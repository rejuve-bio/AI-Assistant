import os
from app.rag.query import RAG
from dotenv import load_dotenv
from .annotation_graph.annotated_graph import Graph
from .cache import ConversationCache
from .summarizer import Graph_Summarizer
from .llm_handle.llm_models import LLMInterface

load_dotenv()

VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION")
class AiAssistance:

    def __init__(self,llm:LLMInterface,schema) -> None:
        self.llm = llm
        self.schema = schema
        self.cache = ConversationCache()

    def clarification(self):
        pass    

    def classify_user_question(self,query,user_id):
        history = self.cache.get_conversation(user_id)

        prompt = f'''
                  '''

        llm = self.llm.generate(prompt)
        return llm  


    def summarize_graph(self,query,graph):
        summary = Graph_Summarizer().summary(query,graph)
        return summary



    def annotate_graph(self,query):
        try:
            annotated_response = Graph(self.llm,self.schema).generate_graph(query)
            return annotated_response
        except:
            pass



    def call_rag(self,query):
        response = RAG(self.llm).result(query, VECTOR_COLLECTION)
        graph = None          # graph is none since the result is from rag
        return response,graph



    def assistant_response(self,query,graph,user_id):
        if graph:
            summary = self.summarize_graph(query,graph)
            return summary

        user_query = self.classify_user_question(query,user_id)
        if user_query =="graph":
            try:
                response = self.annotate_graph(query,user_id)
            except:
                pass
        elif user_query =="llm":
            try:
                response = self.call_rag(query,user_id)
            except:
                pass
        else:
            response =  self.clarification()

        user_history = self.cache.add_conversation(user_id,response)
        return response

