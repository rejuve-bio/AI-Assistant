import os
import logging
from flask import current_app
from app.annotation_graph.schema_handler import SchemaHandler
from app.rag.rag import RAG
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from .annotation_graph.annotated_graph import Graph
from .summarizer import GraphSummarizer
from .llm_handle.llm_models import LLMInterface, get_llm_model


from typing import Annotated
from app.storage.qdrant import Qdrant
from app.prompts.conversation_handler import conversation_prompt
from app.llm_handle.llm_models import openai_embedding_model
from app.memory_layer import MemoryManager
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm_config = {
    "model": "gpt-4",
    "api_key": OPENAI_API_KEY,
    "cache_seed": None
}


class AiAssistance:

    def __init__(self, advanced_llm:LLMInterface, basic_llm:LLMInterface, schema_handler:SchemaHandler) -> None:
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.annotation_graph = Graph(advanced_llm, schema_handler)
        self.graph_summarizer = GraphSummarizer(basic_llm)
        self.rag = RAG(advanced_llm)

        self.message_history = {
            "rag_agent": [],
            "graph_agent": [],
            "user_agent": []
            }
     
    def summarize_graph(self,graph,query):
        summary = self.graph_summarizer.summary(graph,query)
        return summary,None

    def agent(self,message,user_id):
        
        graph_agent = AssistantAgent(
            name="gragh_generate",
            llm_config=llm_config,
            system_message=(
           "You are a knowledgeable assistant specializing in answering questions related to biological annotations. This includes identifying genes, proteins, terms, SNPs, transcripts, and interactions."
           "You have access to a bio knowledge graph to retrieve relevant data."
           "Please note that you can only use the functions provided to you. When your task is complete, respond with 'TERMINATE' to indicate that no further action is required."
            )
        )

        rag_agent = AssistantAgent(
            name="rag_retrival",
            llm_config=llm_config,
            system_message=(
                "You are a helpful assistant responsible for retrieving general informations"
                "You can only use the functions provided to you. Reply 'TERMINATE' when the task is done."
               ),)

        user_agent = UserProxyAgent(
            name="user",
            llm_config=False,
            code_execution_config=False,
            human_input_mode="NEVER",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"))


        @user_agent.register_for_execution()
        @rag_agent.register_for_llm(description="Retrieve information for general knowledge queries.")
        def get_general_response(query:Annotated[str,"always pass the question it self"], user_id: str) -> str:
            try:
                response = self.rag.result(query, user_id)
                return response + "TERMINATE"
            except Exception as e:
                logger.error("Error in retrieving response", exc_info=True)
                return "Error in retrieving response." + "TERMINATE"

        
        @user_agent.register_for_execution()
        @graph_agent.register_for_llm(description="Generate and handle bio-knowledge graphs for annotation-related queries.")
        def generate_graph(query:Annotated[str,f"always pass the question it self"]):
            try:
                response = self.annotation_graph.generate_graph(query)
                return response + "TERMINATE"
            except Exception as e:
                logger.error("Error in generating graph", exc_info=True)
                return "Error in generating graph." +"TERMINATE"


        group_chat = GroupChat(agents=[user_agent, rag_agent, graph_agent], messages=[],max_round=120)
        group_manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=llm_config,
            human_input_mode="NEVER")

        user_agent.initiate_chat(group_manager, message=message, clear_history=False)

        response = group_chat.messages[2]['content']
        return response

    async def save_memory(self,query,user_id):
        # saving the new query of the user to a memorymanager
        memory_manager = MemoryManager(self.llm, embedding_model=openai_embedding_model)
        memory_manager.add_memory(query, user_id)

    async def assistant(self,query,user_id):
        # retrieving saved memories
        context = self.client._retrieve_memory(user_id=user_id)
        prompt = conversation_prompt.format(context=context,query=query)
        response = self.llm.generate(prompt)

        if response:
            if "response:" in response:
                result = response.split("response:")[1].strip()
                return result
            elif "question:" in response:
                refactored_question = response.split("question:")[1].strip()

        await self.save_memory(query,user_id)
        response = self.agent(refactored_question, user_id)
        return response

    def assistant_response(self,query,graph,user_id,graph_id):
        
        if graph:
            logger.info("summarizing graph")
            summary = self.summarize_graph(graph=graph,query=query)
            return summary
            
        if graph_id and query:
            logger.info("summarizing graph")
            summary = self.summarize_graph(graph_id=graph_id,query=query)
            return summary

        if query:
            logger.info("agent calling")
            response = asyncio.run(self.assistant(query, user_id))
            return response

        else:
            return "please provide appropriate question"


