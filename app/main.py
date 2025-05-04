
import logging
import logging.handlers as loghandlers
from dotenv import load_dotenv
from app.annotation_graph.schema_handler import SchemaHandler
from app.rag.rag import RAG
from .annotation_graph.annotated_graph import Graph
from .llm_handle.llm_models import LLMInterface,OpenAIModel,get_llm_model,openai_embedding_model
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent
from typing import Annotated
from app.storage.qdrant import Qdrant
from app.prompts.conversation_handler import conversation_prompt, conversation_prompt_answer
from app.prompts.classifier_prompt import classifier_prompt, classifier_agent_prompt
from app.memory_layer import MemoryManager
from app.summarizer import Graph_Summarizer
from app.history import History
from app.galaxy.Informer import GalaxyInformer
import asyncio
import traceback
import json
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
loghandle = loghandlers.TimedRotatingFileHandler(
                filename="logfiles/Assistant.log",
                when='D', interval=1, backupCount=7,
                encoding="utf-8",
                delay = True,
                utc= True)
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

    def _extract_json_from_codeblock(self, content: str) -> str:
        start = content.find("```json")
        end = content.rfind("```")
        if start != -1 and end != -1:
            json_content = content[start + 7:end].strip()
            return json_content
        else:
            return content


    def agent(self,message,user_id, token):
        message = self.preprocess_message(message)

        
        # classify the users question nased on users explicit intent
        classifier_agent = AssistantAgent(
            name="classifier",
            llm_config={"config_list": self.llm_config},
            system_message= classifier_agent_prompt,
            description="Analyzes and decomposes user queries into sequential, classifiable sub-tasks (bio_annotation or general_knowledge) with dependencies, outputting a JSON plan"
        )
        graph_agent = AssistantAgent(
            name="gragh_generate",
            llm_config = {"config_list" : self.llm_config},
            system_message=(
                "You are a knowledgeable assistant specializing in answering questions related to biological annotations, such as identifying genes, proteins, terms, SNPs, transcripts, and interactions."
                " You have access to a bio knowledge graph to retrieve relevant data."
                "use the function provided to you even if it doesn't have parameters or you deem it unessesary."
                " You can only use the functions provided to you. When your task is complete, reply 'TERMINATE' when the task is done."
               )
        )

        galaxy_agent = AssistantAgent(
            name="galaxy_agent",
            llm_config = {"config_list" : self.llm_config},
            system_message=(
                "You are a senior Galaxy platform specialist and automation information expert in Galaxy tools, workflows, and datasets."
                "You can only use the functions provided to you. Reply 'TERMINATE' when the task is done."
                ),
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

        @user_agent.register_for_execution()
        @galaxy_agent.register_for_llm(description="Get any (general and specific) information about Galaxy tools, workflows, workflow invocations, and datasets based on the users query. Use this function/tool at all times no matter the query.")
        def get_galaxy_information():
            """
            Fetch detailed information on Galaxy tools, workflows, datasets,
            and invocations—including their usage, current state, 
            and any related queries—handling all information requests about Galaxy entities.
            """

            prompt_galaxy=f"""
            **Role**:
            You are a senior bioinformatics specialist and automation expert in Galaxy tools, workflows, and datasets.
            Classify the inputed query into one of the following catagories: 'dataset', 'tool', 'workflow'.
            Classification should be based on the content and the intent of the query and not on the specific words used.
            Respond with the classification only and nothiing else.
            **input**: {message}
            """
            response= self.advanced_llm.generate(prompt=prompt_galaxy)
            print(f"response from galaxy {response}")
            response = response.strip().lower()
            informer= GalaxyInformer(entity_type=response)
            galaxy_response=informer.get_entity_info(search_query = message, user_id = user_id)
            return galaxy_response

        # classify the users question
        classifiction_message=[{"role": "User","content":message}]
        classification= classifier_agent.generate_reply(classifiction_message)
        logger.info(f"classified tasks type {type(classification)}")
        classification_json=self._extract_json_from_codeblock(classification['content'])
        try:
            classification = json.loads(classification_json) if isinstance(classification_json, str) else classification_json
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "Sorry, I couldn’t classify your request."
        logger.info(f"Number of classified tasks {len(classification['tasks'])}")
        logger.info(classification_json)

        # collect the agent responses here.
        agent_responses={}
        agent_knowledge=[]
        for task in classification['tasks']:
            logger.info(f"responding for {task['type']} task: {task['task_id']} with query {task['subquery']}")
            query= task['subquery']
            if task['depends_on']:
                for dependency in task['depends_on']:
                    agent_knowledge.append(agent_responses[dependency])
                logger.info(f"collecting previous knowledge")
                query= "knowledge so far: \n" + str(agent_knowledge) + "\n  question: \n " +task['subquery']

            if task['type'] == 'bio_annotation':
                # response using imitiate chat because they are just two agents interacting
                response = user_agent.initiate_chat(graph_agent,message=query, max_turns=3)
                response= response.chat_history[-1]['content']
                
            elif task['type'] == 'general_knowledge':  
                response= user_agent.initiate_chat(rag_agent,message=query, max_turns=3)
                response= response.chat_history[-1]['content']
            elif task['type'] == 'galaxy' :
                response= user_agent.initiate_chat(galaxy_agent,message=query, max_turns=3)
                response= response.chat_history[-1]['content']

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
            agent_responses[task['task_id']]={"query": task['subquery'], "response": response}

        logger.info("collecting responses and generating general response")
        response= self.advanced_llm.generate(prompt= f"message:{message}\n agent_response: {agent_responses} \n\n Structure the agent_response as a response to the message.")
        return response


    async def save_memory(self,query,user_id):
        # saving the new query of the user to a memorymanager

        logger.info(f"Conversation record:\n {query}")
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
                logging.info("Question refactored based on history")
                refactored_question = response.split("question:")[1].strip()
        # Save both the both the users query and the Ai-assistants response
        response = self.agent(refactored_question, user_id, token)

        ### Adding rich context into the answers as well for an interactive commuication
        prompt_ans= conversation_prompt_answer.format(memory=context,history=history,query=query,raw_answer=response,user_context=user_context)
        response= self.advanced_llm.generate(prompt_ans)
        logger.info(f"file type: {response} \n {type(response)}")
        logging.info("Answer refactored based on context to be more interactive")

        query_response= f"User: {query} \n Assistant: {response}" # conversation to be saved
        await self.save_memory(query_response,user_id)
        self.history.create_history(user_id, query, response)     
        return response 


    def process_file(self, user_id, query, file):
        if file.filename.lower().endswith('.pdf'):
                    response = self.rag.save_retrievable_docs(file,user_id,filter=True)   
                    self.history.create_history(user_id, query, json.dumps(response))
                    return response
        else:
            response = {
                'text': "Only PDF files are supported."
                }
            return response, 400
        
    def process_query(self, resource, token, query, user_id, graph_id ):
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


    def assistant_response(self,query,user_id,token,graph=None,graph_id=None,file=None,resource="annotation"):
      
        try:
            logger.info(f"passes parameters are query = {query}, user_id= {user_id}, graphid={graph_id}, graph = {graph}, resource = {resource}")
            if (file and graph):
                return {"text":"please pass a file to be uploaded or a query with/without graph ids not both"}

            # Needs to be checked
            if (file and query):
                self.process_file(user_id, query, file)
                # process the fle first and then the query
                self.process_query(resource, token, query, user_id, graph_id)
            elif file:
                self.process_file(user_id,query,file)
                
            if graph_id:  
                logger.info("Explaining nodes")

                # Case 1: Both graph_id and query are provided
                if query:
                    self.process_query(resource, token, query, user_id, graph_id )
                    
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
            traceback.print_exc()#