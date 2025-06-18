
import logging
import logging.handlers as loghandlers
from dotenv import load_dotenv
from app.annotation_graph.schema_handler import SchemaHandler
from app.rag.rag import RAG
from .annotation_graph.annotated_graph import Graph
from .llm_handle.llm_models import LLMInterface,OpenAIModel,get_llm_model,openai_embedding_model
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from typing import Annotated
from app.storage.qdrant import Qdrant
from app.prompts.conversation_handler import conversation_prompt
from app.prompts.classifier_prompt import classifier_prompt
from app.memory_layer import MemoryManager
from app.summarizer import Graph_Summarizer
from app.history import History
import asyncio
import traceback
import json
import autogen


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
loghandle = loghandlers.TimedRotatingFileHandler(
                filename="logfiles/Assistant.log",
                when='D', interval=1, backupCount=7,
                encoding="utf-8")
loghandle.setFormatter(
    logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(loghandle)
load_dotenv()

class AiAssistance:

    def __init__(self, advanced_llm:LLMInterface, basic_llm:LLMInterface, schema_handler:SchemaHandler) -> None:
        self.advanced_llm = advanced_llm     
        self.basic_llm = basic_llm
        self.annotation_graph = Graph(advanced_llm, schema_handler)
        self.graph_summarizer = Graph_Summarizer(self.advanced_llm)
        self.client = Qdrant()
        self.rag = RAG(client=self.client,llm=advanced_llm)
        self.history = History()
        
        if self.advanced_llm.model_provider == 'gemini':
            self.llm_config = [{"model":"gemini-1.5-flash","api_key": self.advanced_llm.api_key}]
        else:
            self.llm_config = [{"model": self.advanced_llm.model_name, "api_key":self.advanced_llm.api_key}]

    def agent(self,message,user_id, token):
        message = self.preprocess_message(message)
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
            llm_config=False,
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
            try:
                logger.info(f"Generating graph with arguments: {message}")  # Add this line to log the arguments
                response = self.annotation_graph.generate_graph(message, token)
                return response
            except Exception as e:
                logger.error("Error in generating graph", exc_info=True)
                return f"I couldn't generate a graph for the given question {message} please try again."


        group_chat = GroupChat(agents=[user_agent, rag_agent, graph_agent], messages=[],max_round=3)
        group_manager = GroupChatManager(
            groupchat=group_chat,
            llm_config = {"config_list" : self.llm_config},
            human_input_mode="NEVER")

        user_agent.initiate_chat(group_manager, message=message, clear_history=False)

        response = group_chat.messages[2]['content']
        if response:
            return response
        return group_chat.messages[1]['content']

    async def save_memory(self,query,user_id):
        # saving the new query of the user to a memorymanager
        memory_manager = MemoryManager(self.advanced_llm,client=self.client)
        memory_manager.add_memory(query, user_id)

    async def assistant(self,query,user_id, token, user_context=None):
        # retrieving saved memories
        try:
            # context = self.client._retrieve_memory(user_id=user_id)
            context=None
            history = self.history.retrieve_user_history(user_id)
            user_context = user_context
        except:
            context = {""}
            history = {""}
        prompt = conversation_prompt.format(memory=context,query=query,history=history,user_context=user_context)
        response = self.advanced_llm.generate(prompt)

        if response:
            if "response:" in response:
                result = response.split("response:")[1].strip()
                response = result.strip('"')
                self.history.create_history(user_id, query, response)      
                return {"text":response}
            elif "question:" in response:
                refactored_question = response.split("question:")[1].strip()
        await self.save_memory(query,user_id)
        response = self.agent(refactored_question, user_id, token)
        self.history.create_history(user_id, query, response)     
        return response 

    def assistant_response(self,query,user_id,token,graph=None,graph_id=None,file=None,resource="annotation"):
      
        try:
            logger.info(f"passes parameters are query = {query}, user_id= {user_id}, graphid={graph_id}, graph = {graph}, resource = {resource}")
            if (file and query) or (file and graph):
                return {"text":"please pass a file to be uploaded or a query with/without graph ids not both"}

            if file:
                if file.filename.lower().endswith('.pdf'):
                    response = self.rag.save_retrievable_docs(file,user_id,filter=True)   
                    self.history.create_history(user_id, query, json.dumps(response))
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
                        summary = self.graph_summarizer.summary(token=token, graph_id=graph_id)
                        prompt = classifier_prompt.format(query=query,graph_summary=summary)
                        response = self.advanced_llm.generate(prompt)
                        if "related" in response:
                            logger.info("question is related with with the graph")
                            query_response = self.graph_summarizer.summary(token=token, graph_id=graph_id,  user_query=query)
                            # creating users history
                            self.history.create_history(user_id, query, query_response)    
                            logger.info(f"user query is {query} response is {query_response}")  
                            return query_response
                        elif "not" in response:
                            logger.info("question not related with the graph so sending the query {query} to agent")
                            response = asyncio.run(self.assistant(query, user_id, token, user_context=summary))
                            logger.info(f"user query is {query} response is {response}")  
                            return response           
                        else:
                            logger.warning(f"Unexpected classifier response: {response}. Defaulting to not related.")
                            return response

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
                        # creating users history
                        self.history.create_history(user_id, query, summary)
                        return summary
                    elif resource == "hypothesis":
                        logger.info("Hypothesis resource, no query provided")
                        return {"text": "Explanation for hypothesis resource without query."}
                    else:
                        logger.error(f"Unsupported resource type: '{resource}'")
                        return {"text": f"Unsupported resource type: '{resource}'"}
 
            if query:
                logger.info("agent calling")
                response = asyncio.run(self.assistant(query, user_id, token))
                return response 

            if query and graph:
                summary = self.graph_summarizer.summary(user_query=query,graph=graph)
                self.history.create_history(user_id, query, response)             
                return summary

            if graph:
                summary = self.graph_summarizer.summary(user_query=query,graph=graph)
                self.history.create_history(user_id, query, response)     
                return summary
        except:
            traceback.print_exc()


