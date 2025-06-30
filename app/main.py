
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
import asyncio
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

# os.makedirs(log_dir, exist_ok=True)

logger.setLevel(logging.DEBUG)
loghandle = loghandlers.TimedRotatingFileHandler(
                filename="logfiles/Assistant.log",
                when='D', interval=1, backupCount=7,
                encoding="utf-8")
loghandle.setFormatter(
    logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(loghandle)
load_dotenv()
from typing import TypedDict, Literal, List, Optional, Any
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
import re
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    message: str
    user_id: str
    token: str
    user_context: Optional[str]
    context: Optional[str]
    current_agent: Optional[str]
    annotation_response: Optional[str]
    hypothesis_response: Optional[str]
    rag_response: Optional[str]
    graph_id_response: Optional[str]
    graph_id: Optional[str]
    final_response: str
    needs_graph_id_agent: bool
    agents_completed: List[str]
    input_dict: dict

class LangGraphAiAssistance:
    def __init__(self, advanced_llm, basic_llm, schema_handler):
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.annotation_graph = Graph(advanced_llm, schema_handler)
        self.graph_summarizer = Graph_Summarizer(self.advanced_llm)
        self.rag = RAG(llm=advanced_llm)
        self.history = History()
        self.store = DatabaseManager()
        self.hypothesis_generation = HypothesisGeneration(advanced_llm)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=MemorySaver())

    def _build_workflow(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._route_query)
        workflow.add_node("annotation_agent", self._annotation_agent)
        workflow.add_node("hypothesis_agent", self._hypothesis_agent)
        workflow.add_node("rag_agent", self._rag_agent)
        workflow.add_node("graph_id_agent", self._graph_id_agent)
        workflow.add_node("combiner", self._combine_responses)
        
        # Add edges
        workflow.add_edge(START, "router")
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "annotation": "annotation_agent",
                "hypothesis": "hypothesis_agent", 
                "rag": "rag_agent"
            }
        )
        
        # From each agent, check if graph_id agent is needed
        workflow.add_conditional_edges(
            "annotation_agent",
            self._check_graph_id_needed,
            {
                "graph_id": "graph_id_agent",
                "combiner": "combiner"
            }
        )
        
        workflow.add_conditional_edges(
            "hypothesis_agent", 
            self._check_graph_id_needed,
            {
                "graph_id": "graph_id_agent",
                "combiner": "combiner"
            }
        )
        
        workflow.add_conditional_edges(
            "rag_agent",
            self._check_graph_id_needed, 
            {
                "graph_id": "graph_id_agent",
                "combiner": "combiner"
            }
        )
        
        workflow.add_edge("graph_id_agent", "combiner")
        workflow.add_edge("combiner", END)
        
        return workflow

    def _parse_input_message(self, message_input) -> tuple[str, Optional[str]]:
        """Parse input message which can be string or dictionary"""
        if isinstance(message_input, dict):
            question = message_input.get('question', '')
            graph_id = message_input.get('graph_id', None)
            return question, graph_id
        else:
            # Fallback to string parsing for backward compatibility
            return message_input, self._extract_graph_id_from_string(message_input)
    
    def _extract_graph_id_from_string(self, message: str) -> Optional[str]:
        """Extract graph ID from string message if present (fallback method)"""
        patterns = [
            r'graph[_\s]*id[:\s]*([a-zA-Z0-9_-]+)',
            r'graph[:\s]*([a-zA-Z0-9_-]+)',
            r'id[:\s]*([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1)
        return None

    def _route_query(self, state: AgentState) -> AgentState:
        """Route the query to appropriate agent and extract graph_id if present"""
        # Parse the input message (dictionary or string)
        message, graph_id = self._parse_input_message(state["input_dict"])
        
        # Update state with parsed values
        state["message"] = message
        state["graph_id"] = graph_id
        state["needs_graph_id_agent"] = graph_id is not None
        state["agents_completed"] = []
        
        # Determine primary agent based on message content
        message_lower = message.lower()
        if self._is_annotation_query(message_lower):
            state["current_agent"] = "annotation"
        elif self._is_hypothesis_query(message_lower):
            state["current_agent"] = "hypothesis"
        else:
            state["current_agent"] = "rag"
            
        return state

    def _is_annotation_query(self, message: str) -> bool:
        """Detect annotation-related queries"""
        annotation_keywords = [
            "gene id", "ensg", "protein information", "tp53", "brca1", "brca2",
            "gene-gene interactions", "documented relationships", "established facts"
        ]
        return any(keyword in message for keyword in annotation_keywords)

    def _is_hypothesis_query(self, message: str) -> bool:
        """Detect hypothesis generation queries"""
        hypothesis_keywords = [
            "how might", "could", "possibly", "potential mechanisms",
            "causal relationships", "explain variants", "rs numbers",
            "phenotypes", "biological processes"
        ]
        return any(keyword in message for keyword in hypothesis_keywords)

    def _route_decision(self, state: AgentState) -> Literal["annotation", "hypothesis", "rag"]:
        """Return the routing decision"""
        return state["current_agent"]

    def _annotation_agent(self, state: AgentState) -> AgentState:
        """Handle annotation-related queries"""
        try:
            logger.info(f"Annotation agent processing: {state['message']}")
            response = self.annotation_graph.validated_json(state["message"])
            state["annotation_response"] = response
            state["agents_completed"].append("annotation")
        except Exception as e:
            logger.error("Error in annotation agent", exc_info=True)
            state["annotation_response"] = f"I couldn't generate a graph for the given question {state['message']} please try again."
            state["agents_completed"].append("annotation")
        
        return state

    def _hypothesis_agent(self, state: AgentState) -> AgentState:
        """Handle hypothesis generation queries"""
        try:
            logger.info(f"Hypothesis agent processing: {state['message']}")
            response = self.hypothesis_generation.generate_hypothesis(
                token=state["token"],
                user_query=state["message"]
            )
            state["hypothesis_response"] = response
            state["agents_completed"].append("hypothesis")
        except Exception as e:
            logger.error("Error in hypothesis agent", exc_info=True)
            state["hypothesis_response"] = "Error in generating hypothesis."
            state["agents_completed"].append("hypothesis")
        
        return state

    def _rag_agent(self, state: AgentState) -> AgentState:
        """Handle general information queries"""
        try:
            logger.info(f"RAG agent processing: {state['message']}")
            response = self.rag.get_result_from_rag(state["message"], state["user_id"])
            state["rag_response"] = response
            state["agents_completed"].append("rag")
        except Exception as e:
            logger.error("Error in RAG agent", exc_info=True)
            state["rag_response"] = "Error in retrieving response."
            state["agents_completed"].append("rag")
        
        return state

    def _graph_id_agent(self, state: AgentState) -> AgentState:
        """Handle graph ID specific queries"""
        try:
            logger.info(f"Graph ID agent processing graph_id: {state['graph_id']}")
            # Replace this with your actual graph ID processing function
            response = self._process_graph_id(state["graph_id"], state["token"])
            state["graph_id_response"] = response
            state["agents_completed"].append("graph_id")
        except Exception as e:
            logger.error("Error in graph ID agent", exc_info=True)
            state["graph_id_response"] = f"Error processing graph ID: {state['graph_id']}"
            state["agents_completed"].append("graph_id")
        
        return state

    def _process_graph_id(self, graph_id: str, token: str) -> str:
        """
        Process the graph ID and return relevant information
        Replace this with your actual graph ID processing logic
        """
        # This is a placeholder - implement your actual graph ID processing function
        # For example: return self.graph_processor.process_graph_id(graph_id, token)
        return f"Processed graph ID: {graph_id} with additional context"

    def _check_graph_id_needed(self, state: AgentState) -> Literal["graph_id", "combiner"]:
        """Check if graph ID agent is needed"""
        if state["needs_graph_id_agent"] and "graph_id" not in state["agents_completed"]:
            return "graph_id"
        return "combiner"

    def _combine_responses(self, state: AgentState) -> AgentState:
        """Combine responses from multiple agents"""
        responses = []
        
        # Add primary agent response
        if state.get("annotation_response"):
            responses.append(f"Annotation Analysis: {state['annotation_response']}")
        elif state.get("hypothesis_response"):
            responses.append(f"Hypothesis Generation: {state['hypothesis_response']}")
        elif state.get("rag_response"):
            responses.append(f"General Information: {state['rag_response']}")
        
        # Add graph ID response if available
        if state.get("graph_id_response"):
            responses.append(f"Graph ID Analysis: {state['graph_id_response']}")
        
        # Combine all responses
        if len(responses) > 1:
            state["final_response"] = "\n\n".join(responses)
        elif len(responses) == 1:
            state["final_response"] = responses[0]
        else:
            state["final_response"] = "I'm sorry, I couldn't process your request properly."
        
        return state

    def agent(self, message_input, user_id: str, token: str) -> str:
        """
        Main agent method - accepts dictionary or string input
        
        Args:
            message_input: Dictionary with 'question' and optional 'graph_id' keys, 
                          or string message (for backward compatibility)
            user_id: User identifier
            token: Authentication token
        
        Returns:
            Final response string
        
        Example:
            # Dictionary input (preferred)
            response = agent({'question': 'What is ENSG00000140718?', 'graph_id': 'abc123'}, user_id, token)
            
            # String input (backward compatibility)
            response = agent('What is ENSG00000140718?', user_id, token)
        """
        initial_state = {
            "message": "",  # Will be set by router
            "user_id": user_id,
            "token": token,
            "user_context": None,
            "context": None,
            "current_agent": None,
            "annotation_response": None,
            "hypothesis_response": None,
            "rag_response": None,
            "graph_id_response": None,
            "graph_id": None,  # Will be set by router
            "final_response": "",
            "needs_graph_id_agent": False,
            "agents_completed": [],
            "input_dict": message_input
        }
        
        # Run the workflow
        result = self.app.invoke(initial_state)
        return result["final_response"]

    async def assistant(self, query, user_id: str, token: str, user_context=None, context=None):
        """
        Main assistant method - supports both dictionary and string input
        
        Args:
            query: Dictionary with 'question' and optional 'graph_id' keys, or string
            user_id: User identifier
            token: Authentication token
            user_context: Optional user context
            context: Optional context
        """
        response = self.agent("What is ENSG00000140718? graph_id:abc123", user_id, token)
        # try:
        #     user_information = self.store.get_context_and_memory(user_id)
        #     context = None
        #     memory = user_information['memories']
        #     history = user_information['questions']
        #     logger.info(f"Memory and history: {memory} {history}")
        # except:
        #     context = ""
        #     history = ""
        #     memory = ""
        
        # # Extract the actual question for the conversation prompt
        # if isinstance(query, dict):
        #     question_text = query.get('question', '')
        # else:
        #     question_text = query
        
        # # Use your conversation_prompt here
        # prompt = conversation_prompt.format(
        #     memory=memory,
        #     query=question_text,
        #     history=history,
        #     user_context=user_context
        # )
        # response = self.advanced_llm.generate(prompt)

        # if response:
        #     if "response:" in response:
        #         result = response.split("response:")[1].strip()
        #         final_response = result.strip('"')
        #         await self.store.save_user_information(self.advanced_llm, question_text, user_id, context)
        #         self.history.create_history(user_id, question_text, final_response)
        #         return {"text": final_response}
                
        #     elif "question:" in response:
        #         refactored_question = response.split("question:")[1].strip()
        #         await self.store.save_user_information(self.advanced_llm, question_text, user_id, context)
                
        #         # For refactored questions, we need to handle the format properly
        #         if isinstance(query, dict) and query.get('graph_id'):
        #             # Preserve graph_id in refactored query
        #             agent_input = {'question': refactored_question, 'graph_id': query['graph_id']}
        #         else:
        #             agent_input = refactored_question
                    
        #         agent_response = self.agent(agent_input, user_id, token)
        #         return {"text": agent_response}
        #     else:
        #         logger.warning(f"Unexpected response format: {response}")
        #         await self.store.save_user_information(self.advanced_llm, question_text, user_id, context)
        #         return {"text": response or "I'm sorry, I couldn't process your request properly."}
        # else:
        #     logger.error("No response generated from LLM")
        #     await self.store.save_user_information(self.advanced_llm, question_text, user_id, context)
        #     return {"text": "I'm sorry, I couldn't generate a response at this time."}
    
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
                            prompt = classifier_prompt.format(query=query, graph_summary=summary)
                            response = self.advanced_llm.generate(prompt)
                            
                            if response.startswith("related:"):
                                logger.info("question is related with the graph")
                                query_response = response[len("related:"):].strip()
                                # creating users history
                                self.history.create_history(user_id, query, query_response)
                                logger.info(f"user query is {query} response is {query_response}")
                                return {"text":query_response}

                            elif "not" in response:
                                logger.info("question not related with the graph so sending the query {query} to agent")
                                response = asyncio.run(self.assistant(query, user_id, token, user_context=summary,context=resource))
                                logger.info(f"user query is {query} response is {response}")  
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
                        logger.info(f"Summaries of the graph id {graph_id} is {summary}")
                        if summary is None:
                            logger.info(f"question not related with the graph so sending the query {query} to agent")
                            try:
                                response = asyncio.run(self.assistant(query, user_id, token, user_context=summary))
                                logger.info(f"user query is {query} response is {response}")
                                return response
                            except:
                                return {"text":"Sorry I coudnt understand your question"}
                            
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
                        summary = self.hypothesis_generation.get_by_hypothesis_id(token,graph_id,query)
                        return {"text": summary}
                    else:
                        logger.error(f"Unsupported resource type: '{resource}'")
                        return {"text": f"Unsupported resource type: '{resource}'"}
 
            if query:
                logger.info(f"agent being called for a given query {query} from resource {resource}")
                response = asyncio.run(self.assistant(query=query, user_id=user_id, token=token,context=resource))
                return response 

            if query and graph:
                summary = self.graph_summarizer.summary(user_query=query,graph=graph)
                self.history.create_history(user_id, query, response)             
                return summary

            if graph:
                summary = self.graph_summarizer.summary(user_query=query,graph=graph)
                self.history.create_history(user_id, query, response)     
                return summary

            if json_query:
                logger.info(f"Executing a json query {json_query} to the annotation service")
                try:
                    logger.info(f"Generating graph with arguments: {json_query}")  # Add this line to log the arguments
                    response = self.annotation_graph.generate_graph(f"json format accepted from the user is {json}",json,token)
                    return response
                except Exception as e:
                    logger.error("Error in generating graph", exc_info=True)
                    return f"I couldn't generate a graph for the given format would you please try again."


        except:
            traceback.print_exc()


