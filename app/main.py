import os
from app.rag.query import RAG
from dotenv import load_dotenv
from .annotation_graph.annotated_graph import Graph
from .cache import ConversationCache
from .summarizer import Graph_Summarizer
from .llm_handle.llm_models import LLMInterface,OpenAIModel
import traceback
import logging
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from typing import Annotated

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

    def __init__(self,llm:LLMInterface,schema) -> None:
        self.llm = llm
        self.schema = schema
        self.classifier = Classification_model()
        self.annotation_graph = Graph(llm,schema)
        self.rag = RAG(self.llm)

        self.message_history = {
            "rag_agent": [],
            "graph_agent": [],
            "user_agent": []
            }
     
    def summarize_graph(self,query,graph):
        summary = Graph_Summarizer().summary(query,graph)
        return summary,None

    def agent(self,message,user_id):
        
        def save_history(history):
            self.message_history = history
            
        def get_history():
            return self.message_history

        def get_general_response(query:str, user_id: str) -> str:
            try:
                response = self.rag.result(query, user_id)
                return response
            except Exception as e:
                logger.error("Error in retrieving response", exc_info=True)
                return "Error in retrieving response."
   
        def generate_graph(query:str):
            try:
                response = self.annotation_graph.generate_graph(query)
                return response
            except Exception as e:
                logger.error("Error in generating graph", exc_info=True)
                return "Error in generating graph."

        graph_agent = AssistantAgent(
            name="gragh_generate",
            llm_config=llm_config,
            system_message="you are helpful assistant capable of callign graph generation function to assist users asking graph",
            description="An AI assistant for handling graph generation",
            human_input_mode="NEVER")
        graph_agent.register_for_llm(name="generate_graph", description="Generates a graph for graph-based queries")(generate_graph)
        
        rag_agent = AssistantAgent(
            name="rag_retrival",
            llm_config=llm_config,
            system_message="you are helpful assistant on retriving similiar information from rag",
            description="answer for general questions",
            human_input_mode="NEVER")
        rag_agent.register_for_llm(name="get_general_response", description="Provides responses for general queries")(get_general_response)

        def should_terminate_user(message):
            return "tool_calls" not in message and message["role"] != "tool"

        user_agent = UserProxyAgent(
            name="user",
            llm_config=False,
            description="A human user capable of interacting with AI agents.",
            code_execution_config=False,
            human_input_mode="NEVER",
            is_termination_msg=should_terminate_user
        )

        user_agent.register_for_execution(name="generate_graph")(generate_graph)
        user_agent.register_for_execution(name="get_general_response")(get_general_response)

        group_chat = GroupChat(agents=[user_agent, rag_agent, graph_agent], messages=[],max_round=120)
        group_manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=llm_config,
            human_input_mode="NEVER")


        history = get_history()
        rag_agent._oai_messages={group_manager:history["rag_agent"]}
        graph_agent._oai_messages={group_manager:history["graph_agent"]}
        user_agent._oai_messages = {group_manager:history["user_agent"]}
        user_agent.initiate_chat(group_manager, message=message, clear_history=False)

        save_history({
            "rag_agent":rag_agent.chat_messages.get(group_manager),
            "graph_agent":graph_agent.chat_messages.get(group_manager),
            "user_agent":user_agent.chat_messages.get(group_manager)
        })
        response = group_chat.messages[-1]
        return response

    def assistant_response(self,query,graph,user_id):
        
        if graph:
            logger.info("summarizing graph")
            summary = self.summarize_graph(query,graph)
            return summary

        logger.info("agent calling")
        response = self.agent(query, user_id)
        return response

# import os
# openai_api_key = os.getenv('OPENAI_API_KEY')
# llm = OpenAIModel(openai_api_key)
# a = AiAssistance(llm,"").assistant_response("What is the purpose of the Rejuve platform?", "","1","annotation")
# print(a)

