
from .llm_handle.llm_models import LLMInterface,OpenAIModel,get_llm_model,openai_embedding_model
from .annotation_graph.annotated_graph import Graph
from app.annotation_graph.schema_handler import SchemaHandler
from app.rag.rag import RAG
from app.prompts.conversation_handler import conversation_prompt
from app.prompts.classifier_prompt import classifier_prompt
from app.summarizer import Graph_Summarizer
from app.hypothesis_generation.hypothesis import HypothesisGeneration
from app.storage.history import History
from app.storage.sql_redis_storage import DatabaseManager
from socket_manager import emit_to_user
import asyncio
import logging.handlers as loghandlers
from dotenv import load_dotenv
import traceback
import json
import os
from flask_socketio import emit
from typing import TypedDict, List, Annotated, Any,Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import operator
import logging

logger = logging.getLogger(__name__)
log_dir = "/AI-Assistant/logfiles"
log_file = os.path.join(log_dir, "Assistant.log")
# os.makedirs(log_dir, exist_ok=True)
logger.setLevel(logging.DEBUG)
loghandle = loghandlers.TimedRotatingFileHandler(
                filename="logfiles/Assistant.log",
                when='D', interval=1, backupCount=7,
                encoding="utf-8")
loghandle.setFormatter(
    logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(loghandle)
logger = logging.getLogger(__name__)
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    user_id: str
    token: str
    query_type: str
    response: str
    error: str

class AiAssistance:
    
    def __init__(self, advanced_llm, basic_llm, schema_handler) -> None:
        self.advanced_llm = advanced_llm     
        self.basic_llm = basic_llm
        self.annotation_graph = Graph(advanced_llm, schema_handler)
        self.graph_summarizer = Graph_Summarizer(self.advanced_llm)
        self.rag = RAG(llm=advanced_llm)
        self.history = History()
        self.store = DatabaseManager()
        self.hypothesis_generation = HypothesisGeneration(advanced_llm)
        
        # Initialize the LangGraph workflow
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Define tools
        @tool
        def get_json_format(query: str, token: str) -> str:
            """Retrieve the json format provided from the annotation graph tool"""
            try:
                logger.info(f"Generating graph with arguments: {query}")
                response = self.annotation_graph.validated_json(query)
                return response
            except Exception as e:
                logger.error("Error in generating graph", exc_info=True)
                return f"I couldn't generate a graph for the given question {query} please try again."
        
        @tool
        def get_general_response(query: str, user_id: str) -> str:
            """Retrieve information for general knowledge queries."""
            try:
                response = self.rag.get_result_from_rag(query, user_id)
                return response
            except Exception as e:
                logger.error("Error in retrieving response", exc_info=True)
                return "Error in retrieving response."
        
        @tool
        def hypothesis_generation(query: str, token: str) -> str:
            """Generation of hypothesis for biological mechanisms"""
            try:
                logger.info(f"Here is the user query passed to the agent {query}")
                response = self.hypothesis_generation.generate_hypothesis(token=token, user_query=query)
                return response
            except Exception as e:
                logger.error("Error in hypothesis generation", exc_info=True)
                traceback.print_exc()
                return "Error in generating hypothesis."
        
        # Store tools for later use
        self.tools = [get_json_format, get_general_response, hypothesis_generation]
        
        # Create workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("annotation_agent", self._annotation_agent)
        workflow.add_node("hypothesis_agent", self._hypothesis_agent)
        workflow.add_node("rag_agent", self._rag_agent)
        workflow.add_node("finalizer", self._finalize_response)
        
        # Define edges
        workflow.set_entry_point("classifier")
        
        workflow.add_conditional_edges(
            "classifier",
            self._route_query,
            {
                "annotation": "annotation_agent",
                "hypothesis": "hypothesis_agent",
                "rag": "rag_agent",
                "error": "finalizer"
            }
        )
        
        workflow.add_edge("annotation_agent", "finalizer")
        workflow.add_edge("hypothesis_agent", "finalizer")
        workflow.add_edge("rag_agent", "finalizer")
        workflow.add_edge("finalizer", END)
        
        return workflow
    
    def _classify_query(self, state: AgentState) -> Dict[str, Any]:
        query = state["user_query"]
        
        classifier_prompt = f"""Classify this query into one of these categories:
        - annotation: Requests for factual information about genes, proteins, variants
        - hypothesis: Requests for explanations, mechanisms, or predictions
        - rag: General information requests about Rejuve
        Query: {query}
        Respond ONLY with the category name."""
        
        response = self.advanced_llm.generate(classifier_prompt).lower()
        query_type = response.split()[0]  # Take first word in case LLM adds explanation
        
        return {
            "query_type": query_type,
            "messages": [HumanMessage(content=f"Query classified as: {query_type}")]
        }
        
    def _route_query(self, state: AgentState) -> str:
        """Route query based on classification"""
        return state.get("query_type", "rag")
    
    def _annotation_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle annotation-related queries"""
        try:
            emit_to_user("Creating Query Builder Format...")
            # Use the annotation graph tool
            response = self.annotation_graph.validated_json(state["user_query"], user_id=state["user_id"])
            
            return {
                "response": response,
                "messages": [AIMessage(content=f"Annotation query processed: {response}")]
            }
        except Exception as e:
            logger.error("Error in annotation agent", exc_info=True)
            return {
                "response": f"Error processing annotation query: {str(e)}",
                "error": str(e),
                "messages": [AIMessage(content=f"Error in annotation processing: {str(e)}")]
            }
    
    def _hypothesis_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle hypothesis generation queries"""
        try:
            emit_to_user("Generating hypothesis...")
            response = self.hypothesis_generation.generate_hypothesis(
                token=state["token"], 
                user_query=state["user_query"]
            )
            
            return {
                "response": response,
                "messages": [AIMessage(content=f"Hypothesis generated: {response}")]
            }
        except Exception as e:
            logger.error("Error in hypothesis agent", exc_info=True)
            return {
                "response": f"Error generating hypothesis: {str(e)}",
                "error": str(e),
                "messages": [AIMessage(content=f"Error in hypothesis generation: {str(e)}")]
            }
    
    def _rag_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle general information queries"""
        try:
            emit_to_user("Retrieving information...")
            response = self.rag.get_result_from_rag(state["user_query"], state["user_id"])
            
            return {
                "response": response,
                "messages": [AIMessage(content=f"RAG query processed: {response}")]
            }
        except Exception as e:
            logger.error("Error in RAG agent", exc_info=True)
            return {
                "response": f"Error retrieving information: {str(e)}",
                "error": str(e),
                "messages": [AIMessage(content=f"Error in RAG processing: {str(e)}")]
            }
    
    def _finalize_response(self, state: AgentState) -> Dict[str, Any]:
        """Finalize and return the response"""
        response = state.get("response", "No response generated")
        
        return {
            "messages": [AIMessage(content=f"Final response: {response}")]
        }
    
    def agent(self, message: str, user_id: str, token: str) -> str:
        """Main entry point for processing queries"""
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "user_query": message,
                "user_id": user_id,
                "token": token,
                "query_type": "",
                "response": "",
                "error": ""
            }
            
            # Run the workflow
            result = self.app.invoke(initial_state)
            
            # Extract response
            response = result.get("response", "")
            if response:
                return response
            
            # Fallback to last message content
            if result.get("messages"):
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
            
            return "No response generated"
            
        except Exception as e:
            logger.error("Error in agent processing", exc_info=True)
            return f"Error processing query: {str(e)}"

    async def assistant(self, query, user_id: str, token: str, user_context=None, context=None):

        try:
            user_information = self.store.get_context_and_memory(user_id)
            context = None
            memory = user_information['memories']
            history = user_information['questions']
            logger.info(f"Memory and history: {memory} {history}")
        except:
            context = ""
            history = ""
            memory = ""
        
        # Extract the actual question for the conversation prompt
        if isinstance(query, dict):
            question_text = query.get('question', '')
        else:
            question_text = query
        
        # Use your conversation_prompt here
        prompt = conversation_prompt.format(
            memory=memory,
            query=question_text,
            history=history,
            user_context=user_context
        )
        response = self.advanced_llm.generate(prompt)
        emit_to_user("Analyzing...")
        if response:
            if "response:" in response:
                result = response.split("response:")[1].strip()
                final_response = result.strip('"')
                await self.store.save_user_information(self.advanced_llm, question_text, user_id, context)
                self.history.create_history(user_id, question_text, final_response)
                emit_to_user(final_response,status="completed")
                return {"text": final_response}
                
            elif "question:" in response:
                refactored_question = response.split("question:")[1].strip()
                await self.store.save_user_information(self.advanced_llm, question_text, user_id, context)
                agent_response = self.agent(refactored_question, user_id, token)
                emit_to_user(agent_response,status="completed")
                return agent_response
            else:
                logger.warning(f"Unexpected response format: {response}")
                await self.store.save_user_information(self.advanced_llm, question_text, user_id, context)
                emit_to_user({"text": response or "I'm sorry, I couldn't process your request properly."},status="completed")
                return {"text": response or "I'm sorry, I couldn't process your request properly."}
        else:
            logger.error("No response generated from LLM")
            await self.store.save_user_information(self.advanced_llm, question_text, user_id, context)
            emit_to_user({"text": "I'm sorry, I couldn't generate a response at this time."},status="completed")
            return {"text": "I'm sorry, I couldn't generate a response at this time."}
    
    def assistant_response(self,query,user_id,token,graph=None,graph_id=None,file=None,resource="annotation",json_query=None):  
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
                            emit_to_user("Analyzing User Query...")
                            prompt = classifier_prompt.format(query=query, graph_summary=summary)
                            response = self.advanced_llm.generate(prompt)
                            
                            if response.startswith("related:"):
                                logger.info("question is related with the graph")
                                emit_to_user("Generating Response...")
                                query_response = response[len("related:"):].strip()
                                # creating users history
                                self.history.create_history(user_id, query, query_response)
                                logger.info(f"user query is {query} response is {query_response}")
                                emit_to_user({"text":query_response},status="completed")
                                return {"text":query_response}

                            elif "not" in response:
                                logger.info("question not related with the graph so sending the query {query} to agent")
                                response = asyncio.run(self.assistant(query, user_id, token, user_context=summary,context=resource))
                                logger.info(f"user query is {query} response is {response}")  
                                emit_to_user({"text":query_response},status="completed")
                                return response           
                            else:
                                logger.warning(f"Unexpected classifier response: {response}. Defaulting to not related.")
                                return response

                    elif resource == "hypothesis":
                        """
                        TODO
                        save hypothesis graphs ids along summary if same graph is asked again we won't send an api call instead we will just refer from the db by the id
                        """
                        summary = self.hypothesis_generation.get_by_hypothesis_id(token,graph_id,query)
                        emit_to_user("Analyzing User Query...")
                        logger.info(f"Summaries of the graph id {graph_id} is {summary}")
                        if summary is None:
                            logger.info(f"question not related with the graph so sending the query {query} to agent")
                            try:
                                response = asyncio.run(self.assistant(query, user_id, token, user_context=summary))
                                logger.info(f"user query is {query} response is {response}")
                                emit_to_user(response,status="completed")
                                return response
                            except:
                                emit_to_user({"text":"Sorry I coudnt understand your question"},status="completed")
                                return {"text":"Sorry I coudnt understand your question"}
                            
                        prompt = classifier_prompt.format(query=query,graph_summary=summary)
                        response = self.advanced_llm.generate(prompt)

                        if response.startswith("related:"):
                            logger.info("question is related with the graph")
                            emit_to_user("Generating Response...")
                            query_response = response[len("related:"):].strip()
                            self.history.create_history(user_id, query, query_response)
                            logger.info(f"user query is {query} response is {query_response}")
                            emit_to_user({"text":query_response},status="completed")
                            return {"text":query_response}
                            
                        elif response.strip() == "not":
                            logger.info(f"question not related with the graph so sending the query {query} to agent")
                            response = asyncio.run(self.assistant(query, user_id, token, user_context=summary))
                            logger.info(f"user query is {query} response is {response}")
                            emit_to_user({"text":response},status="completed")
                            return response

                        else:
                            logger.warning(f"Unexpected classifier response: {response}. Defaulting to not related.")
                            emit_to_user({"text": f"Unsupported resource type: '{resource}'"},status="completed")
                            return {"text":response}
                    else:
                        logger.error(f"Unsupported resource type: '{resource}'")
                        emit_to_user({"text": f"Unsupported resource type: '{resource}'"},status="completed")
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
                        summary = self.hypothesis_generation.get_by_hypothesis_id(token,graph_id,query)
                        return {"text": summary}
                    else:
                        logger.error(f"Unsupported resource type: '{resource}'")
                        return {"text": f"Unsupported resource type: '{resource}'"}
 
            if query:
                logger.info(f"agent being called for a given query {query} from resource {resource}")
                response = asyncio.run(self.assistant(query=query, user_id=user_id, token=token,context=resource))
                return response 

            # if query and graph:
            #     summary = self.graph_summarizer.summary(user_query=query,graph=graph)
            #     self.history.create_history(user_id, query, response)             
            #     return summary

            # if graph:
            #     summary = self.graph_summarizer.summary(user_query=query,graph=graph)
            #     self.history.create_history(user_id, query, response)     
            #     return summary

            # if json_query:
            #     logger.info(f"Executing a json query {json_query} to the annotation service")
            #     try:
            #         logger.info(f"Generating graph with arguments: {json_query}")  # Add this line to log the arguments
            #         response = self.annotation_graph.generate_graph(f"",json_query,token)
            #         return response
            #     except Exception as e:
            #         logger.error("Error in generating graph", exc_info=True)
            #         return f"I couldn't generate a graph for the given format would you please try again."


        except:
            traceback.print_exc()


