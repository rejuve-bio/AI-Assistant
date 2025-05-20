

import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from .llm_handle.llm_models import LLMInterface,OpenAIModel,get_llm_model,openai_embedding_model
from .annotation_graph.annotated_graph import Graph
from app.annotation_graph.schema_handler import SchemaHandler
from app.rag.rag import RAG
from app.storage.qdrant import Qdrant
from app.prompts.conversation_handler import conversation_prompt
from app.prompts.classifier_prompt import classifier_prompt
from app.memory_layer import MemoryManager
from app.summarizer import Graph_Summarizer
from app.hypothesis_generation.hypothesis import HypothesisGeneration
from app.history import History
import logging.handlers as loghandlers
from dotenv import load_dotenv
from typing import Annotated
import traceback
import logging
import asyncio
import json
import os


logger = logging.getLogger(__name__)


log_dir = "/AI-Assistant/logfiles"
log_file = os.path.join(log_dir, "Assistant.log")

os.makedirs(log_dir, exist_ok=True)

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
        self.hypothesis_generation = HypothesisGeneration(self.advanced_llm)
        self.client = Qdrant()
        self.rag = RAG(client=self.client,llm=advanced_llm)
        self.history = History()
        
        if self.advanced_llm.model_provider == 'gemini':
            self.llm_config = [{"model":"gemini-1.5-flash","api_key": self.advanced_llm.api_key}]
        else:
            self.llm_config = [{"model": self.advanced_llm.model_name, "api_key":self.advanced_llm.api_key}]


    def preprocess_message(self,message):
        if " and " in message:
            message = message.replace(" and ", " ").strip()
            return message
        return message

    def agent(self,message,user_id, token):
        message = self.preprocess_message(message)

        # graph_agent = AssistantAgent(
        #     name="gragh_generate",
        #     llm_config = {"config_list" : self.llm_config},
        #     system_message=("""
        #                     You are a knowledgeable assistant that executes biological queries in JSON format.
        #                     You must not interpret or modify the JSON.
        #                     When you receive a JSON query, use the `generate_graph` tool to process it and return the output.
        #                     Do not respond with explanations or summaries—just run the tool and return its result.
        #                     End your response with 'TERMINATE'.
        #                 """))

        annotation_validate_agent = AssistantAgent(
            name="validate a json format for a validation",
            llm_config = {"config_list" : self.llm_config},
            system_message=("""
                You are responsible for handling ONLY factual annotation-related user queries. 
                YOUR PRIMARY ROLE:
                - Convert user questions into valid JSON format for Neo4j graph database execution
                - Handle entity identification and relationship queries
                
                TYPES OF QUERIES YOU HANDLE:
                - Gene ID lookups (e.g., "What is ENSG00000140718?")
                - Protein information retrieval (e.g., "Show me information about TP53 protein")
                - Known gene-gene interactions (e.g., "How does BRCA1 interact with BRCA2?")
                - Any query asking for ESTABLISHED FACTS or DOCUMENTED RELATIONSHIPS
                DO NOT generate any text-based responses using your internal knowledge
                ALWAYS use the function to process user queries about genomic information
                When receiving a query, immediately execute the function with the query parameters
                """),
                )
        
        hypothesis_generation_agent = AssistantAgent(
            name="hypothesis generations",
            llm_config = {"config_list" : self.llm_config},
            system_message=("""
                You are responsible for identifying hypothesis-generation queries about biological mechanisms and ALWAYS using the hypothesis_generation function to process them.
                
                YOUR PRIMARY ROLE:
                - Recognize when a user is asking for speculative biological reasoning
                - ALWAYS use the hypothesis_generation function to process these queries
                - Do not provide direct responses or explanations - use only the function
                - Return only what the hypothesis_generation function outputs
                
                QUERY IDENTIFICATION CRITERIA:
                - The query asks about potential mechanisms or causal relationships
                - The query uses speculative language (e.g., "how might," "could," "possibly")
                - The query seeks explanations rather than established facts
                - The user wants reasoning about biological processes or effects
                - ANY query asking to explain variants (rs numbers) or phenotypes
                
                KEY DETECTION PHRASES:
                - "How might rs345 contribute to obesity?"
                - "What mechanism could explain..."
                - "Why would gene X affect condition Y?"
                - "Hypothesize how..."
                - "What's the potential impact of..."
                - "Explain variant rs1421085"
                - "Can you explain the variant rs1421085?"
                
                IMPORTANT INSTRUCTIONS:
                1. Do NOT attempt to answer the biological query yourself
                2. ALWAYS use the hypothesis_generation function
                3. NEVER respond with your own explanation of variants or biological mechanisms
                4. Simply identify that the query matches your criteria and use the function
                5. After calling the function, respond with TERMINATE
                
                Example:
                User: "Can you explain the variant rs1421085?"
                Your action: Call the hypothesis_generation function
                Your response: Return ONLY the function's output + "TERMINATE"
                """),
            )
        
        rag_agent = AssistantAgent(
            name="rag_retrival",
            llm_config = {"config_list" : self.llm_config},
            system_message=("""
                You are responsible for identifying general information queries that fall outside specific biological entity lookups or mechanisms.
                YOUR PRIMARY ROLE:
                - Recognize general information requests that aren't targeted biological lookups or hypothesis generation
                - Route these general queries to the appropriate retrieval function
                - Handle queries that don't fit the specific criteria of the other specialized agents

                QUERY IDENTIFICATION CRITERIA:
                - The query requests general scientific or contextual information
                - The query doesn't focus on specific biological entity data retrieval
                - The query doesn't ask for speculative biological mechanisms
                KEY DETECTION PHRASES:
                "what is rejuve"
                "General information about this site?"
                IMPORTANT: You only identify and route queries to the appropriate function. The function will retrieve and present the actual information. Reply 'TERMINATE' when the identification and routing is complete.
               """
               ),)

        user_agent = UserProxyAgent(
            name="user",
            llm_config=False,
            code_execution_config=False,
            human_input_mode="NEVER",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"))

        @user_agent.register_for_execution()
        @annotation_validate_agent.register_for_llm(description="retrieve the json format provided from the tool")
        def get_json_format() -> str:
            try:
                logger.info(f"Generating graph with arguments: {message}")  # Add this line to log the arguments
                response = self.annotation_graph.generate_graph(message,token)
                return response
            except Exception as e:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                logger.error("Error in generating graph", exc_info=True)
                return f"I couldn't generate a graph for the given question {message} please try again."
        
        # @user_agent.register_for_execution()
        # @graph_agent.register_for_llm(description="Generate and handle bio-knowledge graphs for annotation-related queries.")
        # def generate_graph():
        #     try:
        #         logger.info(f"Generating graph with arguments: {message}")  # Add this line to log the arguments
        #         response = self.annotation_graph.generate_graph("message",message,token)
        #         return response
        #     except Exception as e:
        #         logger.error("Error in generating graph", exc_info=True)
        #         return f"I couldn't generate a graph for the given question {message} please try again."

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
        @hypothesis_generation_agent.register_for_llm(description="generation of hypothesis")
        def hypothesis_generation() -> str:
            try:
                logger.info(f"Here is the user query passed to the agent {message}")
                response = self.hypothesis_generation.generate_hypothesis(token=token,user_query=message)
                return response
            except:
                traceback.print_exc()
       
        group_chat = GroupChat(agents=[user_agent, rag_agent,annotation_validate_agent,hypothesis_generation_agent], messages=[],max_round=3)
        group_manager = GroupChatManager(
            groupchat=group_chat,
            llm_config = {"config_list" : self.llm_config},
            human_input_mode="NEVER")

        user_agent.initiate_chat(group_manager, message=message, clear_history=True)

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
            history =self.history.retrieve_user_history(user_id)
            user_context = user_context
        except:
            context = {""}
            history = {""}
        prompt = conversation_prompt.format(memory=context,query=query,history=history,user_context=user_context)
        response = self.advanced_llm.generate(prompt)

        if "response:" in response:
            result = response.split("response:")[1].strip()
            response = result.strip('"')
            self.history.create_history(user_id, query, response)      
            return {"text":response}
        elif "question:" in response:
            refactored_question = response.split("question:")[1].strip()

        await self.save_memory(query,user_id)
        logger.info(f"agent calling for the given user query {query}")
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
                logger.info("Graph id is passed")

                # Case 1: Both graph_id and query are provided
                if query:
                    logger.debug("Query provided with graph_id")
                    '''
                    TODO
                    if resource is opposite to the requested question implement a mechanism to analyse the user question by differentiating the resources id.
                    '''
                    if resource == "annotation":
                        """
                        TODO
                        save annotation graphs ids along summary if same graph is asked again we won't send an api call instead we will just refer from the db by the id
                        """
                        # Process summary with query
                        summary = self.graph_summarizer.summary(token=token, graph_id=graph_id)
                        prompt = classifier_prompt.format(query=query, graph_summary=summary)
                        response = self.advanced_llm.generate(prompt)
                        
                        if response.startswith("related:"):
                            logger.info("question is related with the graph")
                            query_response = response[len("related:"):].strip()
                            # creating users history
                            self.history.create_history(user_id, query, query_response)
                            logger.info(f"user query is {query} response is {query_response}")
                            return {"text":query_response}

                        elif response.strip() == "not":
                            logger.info(f"question not related with the graph so sending the query {query} to agent")
                            response = asyncio.run(self.assistant(query, user_id, token, user_context=summary))
                            logger.info(f"user query is {query} response is {response}")
                            return response
                        else:
                            logger.warning(f"Unexpected classifier response: {response}. Defaulting to not related.")
                            return {"text":response}

                    elif resource == "hypothesis":
                        """
                        TODO
                        save hypothesis graphs ids along summary if same graph is asked again we won't send an api call instead we will just refer from the db by the id
                        """
                        summary = self.hypothesis_generation.get_by_hypothesis_id(token,query,graph_id)
                        prompt = classifier_prompt.format(query=query,graph_summary=summary)
                        response = self.advanced_llm.generate(prompt)

                        if response.startswith("related:"):
                            logger.info("question is related with the graph")
                            query_response = response[len("related:"):].strip()
                            self.history.create_history(user_id, query, query_response)
                            logger.info(f"user query is {query} response is {query_response}")
                            return {"text":query_response}
                            
                        elif response.strip() == "not":
                            logger.info(f"question not related with the graph so sending the query {query} to agent")
                            response = asyncio.run(self.assistant(query, user_id, token, user_context=summary))
                            logger.info(f"user query is {query} response is {response}")
                            return response

                        else:
                            logger.warning(f"Unexpected classifier response: {response}. Defaulting to not related.")
                            return {"text":response}
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
                        return {"text": "Only Graph id is passed Explanation for hypothesis resource without query is invalid."}
                    else:
                        logger.error(f"Unsupported resource type: '{resource}'")
                        return {"text": f"Unsupported resource type: '{resource}'"}
 
            if query:
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


