
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
import os

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
        # initialize memory manager for saving user queries here
        # memory_manager = MemoryManager(self.advanced_llm,client=self.client)

        
        if self.advanced_llm.model_provider == 'gemini':
            self.llm_config = [{"model":f"{os.getenv('BASIC_LLM_VERSION')}",
                                "api_key": self.advanced_llm.api_key, 
                                  "base_url": "https://generativelanguage.googleapis.com/v1beta",
                                  "api_type": "google" }]
            print("gemini model selected")
        #adding local llm into the options
        elif self.advanced_llm.model_provider == 'local':
             self.llm_config = [
                                    {
                                        "model": f"{os.getenv('ADVANCED_LLM_VERSION')}",
                                        "base_url": "http://127.0.0.1:1234/v1",
                                        "api_key": "NULL"
                                    }
                                ]
        else:
            self.llm_config = [{"model": self.advanced_llm.model_name, "api_key":self.advanced_llm.api_key}]

#### Adding another agent to classify the users question and route it to the appropriate agent
    def preprocess_message(self,message):
        if " and " in message:
            message = message.replace(" and ", " ").strip()
            return message
        return message

    def agent(self,message,user_id, token):
        message = self.preprocess_message(message)

        
        # classify the users question nased on users explicit intent
        classifier_agent = AssistantAgent(
            name="classifier",
            llm_config={"config_list": self.llm_config},
            system_message=(
                "You are a text classifier. Your task is to classify each user query into one of two categories: 'bio_annotation' or 'general_knowledge'. "
                "A query should be classified as 'bio_annotation' only if it demonstrates an explicit intent to obtain detailed information or annotations "
                "about specific biological entities, such as a gene, protein, enhancer, exon, pathway, promoter, snp, super_enhancer, or transcript. "
                "For example, if the user is asking for specific knowledge, details, or annotations about one of these biological elements, then label the query as 'bio_annotation'. "
                "If the mentioned terms appear only in a general or casual context without a clear intent to obtain detailed biological knowledge, classify the query as 'general_knowledge'. "
                "Output only the classification label ('bio_annotation' or 'general_knowledge') without any additional text. "
                "When you have completed all classifications, reply with 'TERMINATE'."
            ),
            description="Classifies input text into 'bio_annotation' for queries with explicit intent to obtain detailed biological information on entities like genes, proteins, enhancers, exons, pathways, promoters, snps, super_enhancers, or transcripts; otherwise classifies as 'general_knowledge'."
        )
        graph_agent = AssistantAgent(
            name="gragh_generate",
            llm_config = {"config_list" : self.llm_config},
            system_message=(
                "You are a knowledgeable assistant specializing in answering questions related to biological annotations, such as identifying genes, proteins, terms, SNPs, transcripts, and interactions."
                " You have access to a bio knowledge graph to retrieve relevant data."
                " You can only use the functions provided to you. When your task is complete, reply 'TERMINATE' when the task is done."
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
                # Addding if condition to check if the response is a dictionary or a string
                if isinstance(response, dict):
                    return response["text"]
                elif isinstance(response, str):
                    return response
                
            except Exception as e:
                logger.error("Error in retrieving response", exc_info=True)
                return "Error in retrieving response."


        @user_agent.register_for_execution()
        @graph_agent.register_for_llm(description="Generate and handle bio-knowledge graphs for annotation-related queries.")
        def generate_graph():
            print("generate graph called")
            try:
                logger.info(f"Generating graph with arguments: {message}")  # Add this line to log the arguments
                response = self.annotation_graph.generate_graph(message, token)
                return response
            except Exception as e:
                logger.error("Error in generating graph", exc_info=True)
                return f"I couldn't generate a graph for the given question {message} please try again."

        print("agents created")
        # classify the users question
        classifiction_message=[{"role": "User","content":message}]
        classification= classifier_agent.generate_reply(classifiction_message)
        print(f"classification of the users question is {classification['content'].strip()}")
        if classification['content'].strip() == 'bio_annotation':
           # response using imitiate chat because they are just two agents interacting
           response = user_agent.initiate_chat(graph_agent,message=message, max_turns=3)
           response= response.chat_history[3]['content']
           
        elif classification['content'].strip() == 'general_knowledge':  
           response= user_agent.initiate_chat(rag_agent,message=message, max_turns=3)
           response= response.chat_history[3]['content']
        else:
            print("Invalid classification, Defaulting to Groupchat with both agents")
            group_chat = GroupChat(agents=[user_agent, graph_agent, rag_agent], messages=[],max_round=3)
            group_manager = GroupChatManager(
                groupchat=group_chat,
                llm_config = {"config_list" : self.llm_config},
                human_input_mode="NEVER")           
            print("group manager created")
            user_agent.initiate_chat(group_manager, message=message, clear_history=False)
            # response is the 2nd message in the group chat
            response = group_chat.messages[2]['content']


        if response:
            return response
        return group_chat.messages[1]['content']


    async def save_memory(self,query,user_id):
        # saving the new query of the user to a memorymanager

        logger.info(f"Conversation record:\n {query}")
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
        # Save both the both the users query and the Ai-assistants response
        response = self.agent(refactored_question, user_id, token)
        query_response= f"User: {query} \n Assistant: {response}" # conversation to be saved
        await self.save_memory(query_response,user_id)
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
                
            if graph_id:  
                logger.info("Explaining nodes")

                # Case 1: Both graph_id and query are provided
                if query:
                    logger.debug("Query provided with graph_id")
                    if resource == "annotation":
                        # Process summary with query
                        summary = self.graph_summarizer.summary(token=token, graph_id=graph_id, user_query=query)
                        return summary
                    elif resource == "hypothesis":
                        logger.info("Hypothesis resource with query")
                        return {"text": "Explanation for hypothesis resource with query."}
                    else:
                        logger.error(f"Unsupported resource type: '{resource}'")
                        return {"text": f"Unsupported resource type: '{resource}'"}

                # Case 2: Only graph_id is provided (no query)
                else:
                    logger.debug("No query provided, but graph_id is available")
                    if resource == "annotation":
                        # Process summary without query
                        summary = self.graph_summarizer.summary(token=token, graph_id=graph_id, user_query=None)
                        return summary
                    elif resource == "hypothesis":
                        logger.info("Hypothesis resource, no query provided")
                        return {"text": "Explanation for hypothesis resource without query."}
                    else:
                        logger.error(f"Unsupported resource type: '{resource}'")
                        return {"text": f"Unsupported resource type: '{resource}'"}

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

        


