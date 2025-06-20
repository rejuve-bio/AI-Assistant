import uuid
import json
import openai
from app.prompts.memory_prompt import FACT_RETRIEVAL_PROMPT, get_update_memory_messages
from .llm_handle.llm_models import LLMInterface, OpenAIModel, get_llm_model, gemini_embedding_model
import traceback

class MemoryManager:
    def __init__(self, llm, client):
        """
        Initializes the MemoryManager with the necessary components.
        :param llm: The language model instance.
        :param client: The Qdrant client instance.
        """
        self.llm = llm
        self.embedding_model = gemini_embedding_model  # Gemini embedding model
        self.client = client

    def get_fact_retrieval_message(self, messages):
        """
        Constructs the fact retrieval prompt.
        :param messages: The input messages.
        :return: A tuple containing the system and user prompts.
        """
        return FACT_RETRIEVAL_PROMPT, f"Input: {messages}"

    def qdrant_client_retrieved_user_similar_preferences(self, user_id, embedding):
        """
        Retrieves similar user preferences from Qdrant.
        :param user_id: The user ID.
        :param embedding: The embedding vector.
        :return: Retrieved contents from Qdrant.
        """
        return self.client._retrieve_memory(user_id, embedding)

    def add_memory(self, messages, user_id):
        try:
            """
            Adds memory for a user.
            :param messages: Messages from the user.
            :param user_id: The user ID.
            :return: A list of returned memories with their details.
            """
            if not user_id:
                return "userid is an obligatory to save memory"
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            else:
                messages = []

            metadata = {}
            system_prompt, user_prompt = self.get_fact_retrieval_message(messages)

            response = self.llm.generate(prompt = f"{system_prompt} \n {user_prompt}")

            new_retrieved_facts = []
            try:
                if isinstance(response, dict):
                    new_retrieved_facts = response.get("facts", [])
                elif isinstance(response, str):
                    # Convert the substring to a dictionary
                    try:
                        start_idx = response.find("{")
                        end_idx = response.rfind("}") + 1
                        response_dict_str = response[start_idx:end_idx]
                        response_dict= response_dict_str.replace("'", "\"")
                        response_dict = json.loads(response_dict)  # Replace single quotes with double quotes for valid JSON
                        # Extract the 'content' field
                        content = response_dict.get("content", "")
                        print(f"Extracted content: {content}")
                        new_retrieved_facts.append(content)
                    except Exception as e:
                        print(f"Error extracting content: {e}")
                        new_retrieved_facts.append(response)
            except Exception as e:
                print(f"Error processing LLM response: {str(e)}")
                new_retrieved_facts = []

            retrieved_old_memory = []
            new_message_embeddings = {}

            # Process each fact with Gemini embeddings
            for fact in new_retrieved_facts:
                try:
                    # Generate embeddings for single fact (batch-compatible)
                    embedded_message = self.embedding_model([fact])  # Wrap in list                   
                    # Extract single embedding from batch response
                    if embedded_message and len(embedded_message) > 0:
                        embedding_vector = embedded_message[0]  # Get first embedding
                        new_message_embeddings[fact] = embedding_vector
                        # Retrieve existing memories
                        existing_memory = self.qdrant_client_retrieved_user_similar_preferences(
                            user_id, embedding_vector
                        )
                                               
                        if existing_memory:
                            retrieved_old_memory.extend([{
                                "id": mem["id"],
                                "text": mem["content"]
                            } for mem in existing_memory])

                except Exception as e:
                    print(f"Error processing fact '{fact}': {str(e)}")
                    continue

            # Prepare UUID mapping
            temp_uuid_mapping = {
                str(idx): item["id"]
                for idx, item in enumerate(retrieved_old_memory)
            }
            for idx, item in enumerate(retrieved_old_memory):
                item["id"] = str(idx)

            # Process memory updates
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts
            )
            system_prompt = (
                                "You are an AI assistant tasked with managing memory updates. "
                                "Always return your response as a valid JSON object."
                                "Make sure ther response is in the format {'memory': [{'id': 'uuid', 'text': 'memory_text', 'event': 'ADD/UPDATE/DELETE/NONE'}]}"
                            )
            new_memories_with_actions = self.llm.generate(prompt=function_calling_prompt, system_prompt=system_prompt)
            returned_memories = []

            # Handle memory operations
            if "memory" in new_memories_with_actions:
                if isinstance(new_memories_with_actions, str):
                    new_memories_with_actions = json.loads(new_memories_with_actions)
                for resp in new_memories_with_actions.get("memory", []):
                    try:
                        data = resp.get("text", "")
                        event_type = resp.get("event", "NONE")
                        if event_type == "ADD":
                            # Verify embedding exists before creating
                            if data in new_message_embeddings and fact == data:
                                memory_id = self.client._create_memory_update_memory(
                                    user_id=user_id,
                                    data=data,
                                    embedding=new_message_embeddings[data],  # Direct vector
                                    metadata=metadata
                                )
                                if memory_id:
                                    returned_memories.append({
                                        "id": memory_id,
                                        "memory": data,
                                        "event": event_type
                                    })

                        elif event_type == "UPDATE":
                            if data in new_message_embeddings:
                                self.client._create_memory_update_memory(
                                    user_id=user_id,
                                    memory_id=temp_uuid_mapping[resp["id"]],
                                    data=data,
                                    embedding=new_message_embeddings[data],  # Direct vector
                                    metadata=metadata,
                                )
                                returned_memories.append({
                                    "id": temp_uuid_mapping[resp["id"]],
                                    "memory": data,
                                    "event": event_type,
                                    "previous_memory": resp.get("old_memory", "")
                                })

                        elif event_type == "DELETE":
                            self.client._delete_memory(temp_uuid_mapping[resp["id"]])
                            returned_memories.append({
                                "id": temp_uuid_mapping[resp["id"]],
                                "memory": data,
                                "event": event_type
                            })

                        elif event_type == "NONE":
                            print(f"No operation for memory: {data}")

                    except Exception as e:
                        print(f"Error processing memory operation: {str(e)}")
                        continue

            print(f"Final returned memories: {returned_memories}")
            return returned_memories

        except Exception as e:
            traceback.print_exc()
            return []