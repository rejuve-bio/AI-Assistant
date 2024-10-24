import os
from app.rag.query import RAG
from dotenv import load_dotenv

load_dotenv()

VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION")
class AiAssistance:

    def __init__(self) -> None:
        pass
        self.rag = RAG()

    def call_rag(self,query):
        response = self.rag.result(query, VECTOR_COLLECTION)
        return response

