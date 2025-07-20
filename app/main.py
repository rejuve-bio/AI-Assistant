import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from .llm_handle.llm_models import (
    LLMInterface,
)
from .annotation_graph.annotated_graph import Graph
from app.annotation_graph.schema_handler import SchemaHandler
from app.rag.rag import RAG
from app.prompts.conversation_handler import conversation_prompt
from app.prompts.classifier_prompt import classifier_prompt
from app.summarizer import Graph_Summarizer
from app.hypothesis_generation.hypothesis import HypothesisGeneration
from app.storage.history import History
from app.storage.sql_redis_storage import DatabaseManager
import logging.handlers as loghandlers
from dotenv import load_dotenv
from typing import Annotated
import traceback
import logging
import autogen
import os

logger = logging.getLogger(__name__)

log_dir = "/AI-Assistant/logfiles"
log_file = os.path.join(log_dir, "Assistant.log")

os.makedirs(log_dir, exist_ok=True)

logger.setLevel(logging.DEBUG)
loghandle = loghandlers.TimedRotatingFileHandler(
    filename="logfiles/Assistant.log",
    when="D",
    interval=1,
    backupCount=7,
    encoding="utf-8",
)
loghandle.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(loghandle)
load_dotenv()


class AiAssistance:

    def __init__(
        self,
        advanced_llm: LLMInterface,
        basic_llm: LLMInterface,
        schema_handler: SchemaHandler,
        embedding_model=None,
        qdrant_client=None,
    ) -> None:
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.annotation_graph = Graph(advanced_llm, schema_handler)
        self.graph_summarizer = Graph_Summarizer(self.advanced_llm)

        self.rag = RAG(
            llm=advanced_llm,
            qdrant_client=qdrant_client,
        )
        self.history = History()
        self.store = DatabaseManager()
        self.hypothesis_generation = HypothesisGeneration(advanced_llm)

        if self.advanced_llm.model_provider == "gemini":
            self.llm_config = [
                {"model": "gemini-1.5-flash", "api_key": self.advanced_llm.api_key}
            ]
        else:
            self.llm_config = [
                {
                    "model": self.advanced_llm.model_name,
                    "api_key": self.advanced_llm.api_key,
                }
            ]

    def agent(self, message, user_id, token):
        # message = self.preprocess_message(message)

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
            llm_config={"config_list": self.llm_config},
            system_message=(
                """
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
                """
            ),
        )

        hypothesis_generation_agent = AssistantAgent(
            name="hypothesis generations",
            llm_config={"config_list": self.llm_config},
            system_message=(
                """
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
                """
            ),
        )

        rag_agent = AssistantAgent(
            name="rag_retrival",
            llm_config={"config_list": self.llm_config},
            system_message=(
                """
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
            ),
        )

        user_agent = UserProxyAgent(
            name="user",
            llm_config=False,
            code_execution_config=False,
            human_input_mode="NEVER",
            is_termination_msg=lambda x: x.get("content", "")
            and x.get("content", "").rstrip().endswith("TERMINATE"),
        )

        @user_agent.register_for_execution()
        @annotation_validate_agent.register_for_llm(
            description="retrieve the json format provided from the tool"
        )
        def get_json_format() -> str:
            try:
                logger.info(
                    f"Generating graph with arguments: {message}"
                )  # Add this line to log the arguments
                response = self.annotation_graph.validated_json(message)
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
        @rag_agent.register_for_llm(
            description="Retrieve information for general knowledge queries."
        )
        def get_general_response() -> str:
            try:
                response = self.rag.get_result_from_rag(message, user_id)
                return response
            except Exception as e:
                logger.error("Error in retrieving response", exc_info=True)
                return "Error in retrieving response."

        @user_agent.register_for_execution()
        @hypothesis_generation_agent.register_for_llm(
            description="generation of hypothesis"
        )
        def hypothesis_generation() -> str:
            try:
                logger.info(f"Here is the user query passed to the agent {message}")
                response = self.hypothesis_generation.generate_hypothesis(
                    token=token, user_query=message
                )
                return response
            except:
                traceback.print_exc()

        group_chat = GroupChat(
            agents=[
                user_agent,
                rag_agent,
                annotation_validate_agent,
                hypothesis_generation_agent,
            ],
            messages=[],
            max_round=3,
        )
        group_manager = GroupChatManager(
            groupchat=group_chat,
            llm_config={"config_list": self.llm_config},
            human_input_mode="NEVER",
        )

        user_agent.initiate_chat(group_manager, message=message, clear_history=True)

        response = group_chat.messages[2]["content"]
        if response:
            return response
        return group_chat.messages[1]["content"]

    async def assistant(self, query, user_id, token, user_context=None, context=None):
        try:
            user_information = self.store.get_context_and_memory(user_id)
            context = None
            memory = user_information["memories"]
            history = user_information["questions"]
            logger.info(f"here is the memory and history {memory} {history}")
        except:
            context = {""}
            history = {""}
            memory = {""}

        # Get conversation history for better context
        conversation_history = self.history.retrieve_user_history(user_id)
        user_history = conversation_history.get(str(user_id), [])

        # Format history for the prompt
        history_context = ""
        if user_history:
            history_context = "\n\nPrevious conversation:\n"
            for entry in user_history[-2:]:  # Last 2 conversations for context
                history_context += f"User: {entry['user']}\n"
                history_context += f"Assistant: {entry['assistant answer']}\n"

        prompt = conversation_prompt.format(
            memory=memory,
            query=query,
            history=history,
            user_context=user_context,
            conversation_history=history_context,
        )
        response = self.advanced_llm.generate(prompt)

        if response:
            if "response:" in response:
                result = response.split("response:")[1].strip()
                final_response = result.strip('"')
                await self.store.save_user_information(
                    self.advanced_llm, query, user_id, context
                )
                query_id = self.history.create_history(user_id, query, final_response)
                return {"text": final_response, "query_id": query_id}

            elif "question:" in response:
                refactored_question = response.split("question:")[1].strip()
                await self.store.save_user_information(
                    self.advanced_llm, query, user_id, context
                )
                agent_response = self.agent(refactored_question, user_id, token)
                # Also save agent responses to history
                query_id = self.history.create_history(
                    user_id, refactored_question, agent_response
                )
                return {"text": agent_response, "query_id": query_id}
            else:
                logger.warning(f"Unexpected response format: {response}")
                await self.store.save_user_information(
                    self.advanced_llm, query, user_id, context
                )
                query_id = self.history.create_history(user_id, query, response)
                return {
                    "text": response
                    or "I'm sorry, I couldn't process your request properly.",
                    "query_id": query_id,
                }
        else:
            logger.error("No response generated from LLM")
            await self.store.save_user_information(
                self.advanced_llm, query, user_id, context
            )
            return {"text": "I'm sorry, I couldn't generate a response at this time."}

    def assistant_response(
        self,
        query,
        user_id,
        token,
        graph=None,
        graph_id=None,
        file=None,
        resource="annotation",
        json_query=None,
        pdf_ids=None,
    ):
        try:
            logger.info(
                f"passes parameters are query = {query}, user_id= {user_id}, graphid={graph_id}, graph = {graph}, resource = {resource}, pdf_ids = {pdf_ids}"
            )
            # Always use the RAG agent for all queries
            response = self.rag.get_result_from_rag(query, user_id, pdf_ids=pdf_ids)

            # Save RAG responses to history for conversation continuity
            if response and isinstance(response, dict) and "text" in response:
                query_id = self.history.create_history(user_id, query, response["text"])
                response["query_id"] = query_id

            return response
        except Exception as e:
            logger.error(f"Exception in assistant_response: {e}")
            traceback.print_exc()
            return {"text": f"Error processing query: {str(e)}"}
