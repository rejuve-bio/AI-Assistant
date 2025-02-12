
import logging
from dotenv import load_dotenv
from app.annotation_graph.schema_handler import SchemaHandler
from app.rag.rag import RAG
from .annotation_graph.annotated_graph import Graph
from .llm_handle.llm_models import LLMInterface,OpenAIModel,get_llm_model,openai_embedding_model
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent
from typing import Annotated
from app.storage.qdrant import Qdrant
from app.prompts.conversation_handler import conversation_prompt
from app.memory_layer import MemoryManager
from app.summarizer import Graph_Summarizer
import asyncio
import traceback
import autogen

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class AiAssistance:

    def __init__(self, advanced_llm:LLMInterface, basic_llm:LLMInterface, schema_handler:SchemaHandler) -> None:
        self.advanced_llm = advanced_llm  
        self.basic_llm = basic_llm
        self.annotation_graph = Graph(advanced_llm, schema_handler)
        self.graph_summarizer = Graph_Summarizer(self.advanced_llm)
        self.client = Qdrant()
        self.rag = RAG(client=self.client,llm=advanced_llm)

        
        if self.advanced_llm.model_provider == 'gemini':
            self.llm_config = [{"model":"gemini-pro",
                                "api_key": self.advanced_llm.api_key, 
                                  "base_url": "https://generativelanguage.googleapis.com/v1beta",
                                  "api_type": "google" }]
            print("gemini model selected")
        else:
            self.llm_config = [{"model": self.advanced_llm.model_name, "api_key":self.advanced_llm.api_key}]


    def agent(self,message,user_id, token):
        graph_agent = ConversableAgent(
            name="gragh_generate",
            llm_config = {"config_list" : self.llm_config},
            system_message=(        
           "You are a knowledgeable assistant specializing in answering questions related to biological annotations. This includes identifying genes, proteins, terms, SNPs, transcripts, and interactions."
           "You have access to a bio knowledge graph to retrieve relevant data."
           "Please note that you can only use the functions provided to you specifically to retrieve relevant data using the users query given to you below. When your task is complete, Reply 'TERMINATE' when the task is done."
            )
        )

        rag_agent = AssistantAgent(
            name="rag_retrival",
            llm_config = {"config_list" : self.llm_config},
            system_message=(
                "You are a helpful assistant responsible for retrieving general informations"
                "You can only use the functions provided to you. Reply 'TERMINATE' when the task is done."
               ),)

        user_agent = UserProxyAgent(
            name="user",
            llm_config={"config_list" : self.llm_config},
            system_message=(
                "You are a user who is interacting with the assistant. You can ask questions and get answers from the assistant."
                "Please note that you can only use the two agents(rag_agent, and the graph_agent). Generate an answer that is clear and understandable for humans with the response still containing its core content."
                " When your task is complete, Reply 'TERMINATE' when the task is done."
                ),
            code_execution_config=False,
            human_input_mode="NEVER",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"))


        @user_agent.register_for_execution()
        @rag_agent.register_for_llm(description="Retrieve information for general knowledge queries.")
        def get_general_response() -> str:
            try:
                response = self.rag.get_result_from_rag(message, user_id)
                return response
            except Exception as e:
                logger.error("Error in retrieving response", exc_info=True)
                return "Error in retrieving response."


        @user_agent.register_for_execution()
        @graph_agent.register_for_llm(description="Generate and handle bio-knowledge graphs for annotation-related queries.")
        def generate_graph():
            print("generate graph called")
            try:
                response = self.annotation_graph.generate_graph(message, token)
                return response
            except Exception as e:
                logger.error("Error in generating graph", exc_info=True)
                return f"I couldn't generate a graph for the given question {message} please try again."

        print("agents created")
        group_chat = GroupChat(agents=[user_agent, graph_agent, rag_agent], messages=[],max_round=3)
        group_manager = GroupChatManager(
            groupchat=group_chat,
            llm_config = {"config_list" : self.llm_config},
            human_input_mode="NEVER")
        print("group manager created")
        user_agent.initiate_chat(group_manager, message=message, clear_history=False)
        response = group_chat.messages[2]['content']
        if response:
            return response
        return group_chat.messages[1]['content']

    async def save_memory(self,query,user_id):
        # saving the new query of the user to a memorymanager
        memory_manager = MemoryManager(self.advanced_llm,client=self.client)
        memory_manager.add_memory(query, user_id)

    async def assistant(self,query,user_id, token):
        # retrieving saved memories
        try:
            context = self.client._retrieve_memory(user_id=user_id)
        except:
            context = {""}
        prompt = conversation_prompt.format(context=context,query=query)
        response = self.advanced_llm.generate(prompt)

        if response:
            if "response:" in response:
                result = response.split("response:")[1].strip()
                return {"text":result.strip('"')}
            elif "question:" in response:
                refactored_question = response.split("question:")[1].strip()
        await self.save_memory(query,user_id)
        response = self.agent(refactored_question, user_id, token)
        return response 

    def assistant_response(self,query,user_id,token,graph=None,graph_id=None,file=None,resource="annotation"):
      
        try:
            if (file and query) or (file and graph):
                return {"text":"please pass a file to be uploaded or a query with/without graph ids not both"}

            if file:
                if file.filename.lower().endswith('.pdf'):
                    response = self.rag.save_retrievable_docs(file,user_id,filter=True)            
                    return response
                else:
                    response = {
                        'text': "Only PDF files are supported."
                        }
                    return response, 400
                
            if graph_id and query:
                logger.info("explaining nodes")
                if resource=="annotation":
                    summary = self.graph_summarizer.summary(token=token,graph_id=graph_id, user_query=query)
                    return summary
                if resource=="hypothesis":
                    logger.info("no hypothesis graph ids")
                    return {"text":"null"}
                else:
                    return {"text":f"Unsupported resource type: '{resource}'"}

            if query and graph:
                summary = self.graph_summarizer.summary(user_query=query,graph=graph)
                return summary

            if query:
                logger.info("agent calling")
                response = asyncio.run(self.assistant(query, user_id, token))
                return response               
                           
            if graph:
                summary = self.graph_summarizer.summary(user_query=query,graph=graph)
                return summary
        except:
            traceback.print_exc()

        


