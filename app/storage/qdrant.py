
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import traceback
import pandas as pd
from typing import List
from qdrant_client.models import PointStruct, PointIdsList
from dotenv import load_dotenv
import uuid
from datetime import datetime
import random

MAX_MEMORY_LIMIT = 10
MAX_PDF_LIMIT = 2
USER_COLLECTION = os.getenv("USER_COLLECTION","USER_COLLECTIONS")
USER_MEMORY_NAME = "user memories"

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
class Qdrant:

    def __init__(self):

        try:
            self.client = QdrantClient(os.environ.get('QDRANT_CLIENT','http://localhost:6333'))
            print(f"qdrant connected")
        except:
            print('qdrant connection is failed')


    def get_create_collection(self,collection_name):

        try:
            self.client.get_collection(collection_name)
            print(f"Collection '{collection_name}' EXISTS.")
        except:
            print("no such collection exists")
            try:
                logger.info(f"creating collection {collection_name}")
                # Get vector size based on model type
                #####################################
                vector_size = 768 # changed to 768 from 1536
                ###########################
                self.client.create_collection(
                    collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.DOT) )
                    # why DOT product is used as distance metric?
                print(f"Collection '{collection_name}' CREATED.")
            except:
                traceback.print_exc()
                logger.info("error creating a collection")


    def upsert_data(self,collection_name,df,user_id=None):
                try:
                    excluded_columns = {"dense"}
                    payload_columns = [col for col in df.columns if col not in excluded_columns]
                    payloads_list = [
                        {col: getattr(item, col) for col in payload_columns}
                        for item in df.itertuples(index=False)
                    ]

                    if user_id:
                        # Noting the time of uploading the file
                        upsert_time= datetime.now().isoformat()
                        filename = df["filename"].to_list()[0]
                        for payload in payloads_list:
                            payload["user_id"] = user_id
                            payload["id"] = f"{user_id}_{filename}"
                            payload["time"] =  upsert_time     # Adding to track the time of the upserted data
                        
                    
                    if 'id' not in df.columns:
                        df['id'] = [random.randint(100000, 999999) for _ in range(len(df))]
                    
                    self.get_create_collection(collection_name)
                    self.client.upsert(
                        collection_name=collection_name,
                        points=models.Batch(
                            ids=df["id"].tolist(),
                            vectors=df["dense"].tolist(),
                            payloads=payloads_list,
                        ),
                    )
                    print("Embedding saved")
                    return "Data Successfully Uploaded"
                
                except Exception as e:
                    traceback.print_exc()
                    print("Error saving:", e)
            
    def retrieve_data(self,collection, query,user_id, filter=None, galaxy=None):
        try:
            if filter:
                result = self.client.search(
                        collection_name=collection,
                        query_vector=query,
                        with_payload=True,
                        score_threshold=0.3,
                        query_filter= models.Filter(
                                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id),),]),
                        limit=10)
                response = {}
                for i, point in enumerate(result):
                    response[i] = {
                        "score": point.score,
                        "content": point.payload.get('content', 'No content available')
                    }
                return response
        
            result = self.client.search(
                    collection_name=collection,
                    query_vector=query,
                    with_payload=True,
                    score_threshold=0.3,
                    limit=10)
            response = {}

            if galaxy is None:
                for i, point in enumerate(result):
                    response[i] = {
                        "id": point.id,
                        "score": point.score,
                        "authors": point.payload.get('authors', 'Unknown'),
                        "content": point.payload.get('content', 'No content available'),
                        "filename": point.payload.get('filename')
                    }

            # Handle and retrive galaxy information
            elif galaxy is not None:
                if galaxy == "tool":

                    for i,point in enumerate(result):
                        response[i]={
                            "id": point.id,
                            "score": point.score,
                            "description": point.payload.get('description', 'description not available'),
                            "tool_id": point.payload.get('tool_id'),
                            "name": point.payload.get('name')

                     }
                elif galaxy == "workflow":
                    for i,point in enumerate(result):
                        response[i]={
                            "id": point.id,
                            "score": point.score,
                            "model_class": point.payload.get('model_class', 'unknown'),
                            'description': point.payload.get('description', 'unkown'),
                            "owner": point.payload.get('owner', 'unknown'),
                            "workflow_id": point.payload.get('workflow_id'),
                            "name": point.payload.get('name')
                     }
                elif galaxy == 'dataset':
                    for i, point in enumerate(result):
                        response[i]={
                            'id': point.id,
                            'score': point.score,
                            "dataset_id": point.payload.get('dataset_id'),
                            "name": point.payload.get('name'),
                            "full_path": point.payload.get('full_path', 'unknown'),
                            "type": point.payload.get('type', 'unknown'),
                            "source": point.payload.get('source')
                        }
            return response
        except:
            return {"error":"not found"}
    # Metadata as a parameter? What does metadata represent?
    def _create_memory_update_memory(self, user_id, data, embedding, metadata=None, memory_id=None):
        self.get_create_collection(USER_COLLECTION)
        current_time = datetime.utcnow().isoformat()
        payload_data = [{
            "content": data,
            "user_id": user_id,
            "created_at_updated_at": current_time,
            "status": USER_MEMORY_NAME
        }]

        try:
            # Convert embedding to proper format
            if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
                raise ValueError("Invalid embedding format")

            # Qdrant requires vectors to be list[list[float]] even for single points
            vectors = [embedding]  # Wrap single embedding in a list

            if memory_id:
                # Update existing memory
                self.client.upsert(
                    collection_name=USER_COLLECTION,
                    points=models.Batch(
                        ids=[memory_id],     # Must be list[str]
                        vectors=vectors,     # Now [[float, float,...]]
                        payloads=payload_data,
                    )
                )
                return memory_id

            # Handle new memory creation
            # Check and enforce memory limit
            memories = self.client.scroll(USER_COLLECTION, with_payload=True)
            if len(memories[0]) >= MAX_MEMORY_LIMIT:
                sorted_memories = sorted(
                    memories[0],
                    key=lambda m: m.payload["created_at_updated_at"]
                )
                oldest_id = sorted_memories[0].id
                self._delete_memory(oldest_id)
                logger.info(f"Deleted oldest memory {oldest_id} due to limit {MAX_MEMORY_LIMIT}")

            # Create new memory with proper ID format
            memory_id = str(uuid.uuid4())  # Single string ID
            self.client.upsert(
                collection_name=USER_COLLECTION,
                points=models.Batch(
                    ids=[memory_id],      # Wrap in list
                    vectors=vectors,       # Proper [[float,...]] format
                    payloads=payload_data,
                )
            )
            logger.info(f"Created new memory {memory_id}")
            return memory_id  # Return string ID

        except Exception as e:
            logger.error(f"Memory operation failed: {str(e)}")
            traceback.print_exc()
            return None


    def _delete_memory(self, memory_id):

        self.client.delete(
            collection_name=USER_COLLECTION,
            points_selector=models.PointIdsList(
                points=[memory_id],
            ),
        )
        return None

    def _retrieve_memory(self,user_id,embedding=None):
        try:
            #added the get_create_collection function to create a collection if it does not exist
            self.get_create_collection(USER_COLLECTION)
            if embedding:
                result = self.client.search(
                        collection_name=USER_COLLECTION,
                        query_vector=embedding,
                        with_payload=True,
                        # score threshold of 0.5 will return a similiar memories with similiarity score of more than 0.5
                        score_threshold=0.5,
                        query_filter= models.Filter(
                                                must=[
                                                    models.FieldCondition(
                                                    key="user_id", match=models.MatchValue(value=user_id),),
                                                    models.FieldCondition(
                                                    key="status", match=models.MatchValue(value=USER_MEMORY_NAME),)
                                                    ],
                                                ),
                        limit=1000)

                if result:
                    response = {}
                    for i, point in enumerate(result):
                        response[i] = {
                            "id": point.id,
                            "content": point.payload.get('content'),
                            "date": point.payload.get('created_at_updated_at')
                            }

                    return [response[0]]
            else:
                data = self.client.scroll(
                    collection_name=USER_COLLECTION,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                        ]
                    ),
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )

                data = [record.payload['content'] for record in data[0][::-1]]
                return data
        except:
            traceback.print_exc()
            return None
        
    # Updating the time of the recently used document.
    def update_payload_info(self, collection_name, file_name):
        self.client.set_payload(
            collection_name=f"{collection_name}",
            payload={
                "time": datetime.now().isoformat()
            },
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=file_name),  # Filter using file_name
                    ),
                ],
            ),
        )
    # deleting function that deletes all the least recently used file points
    def delete_pdf(self, collection_name, file_name=None):
        if file_name:
            self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="content",
                                match=models.MatchValue(value=file_name)
                            ),
                        ],
                    ),
                )
            
        else:
            # Retrieve all points (files) from the collection
            print(f"Collection name:  {collection_name}")
            response = self.client.scroll(
                            collection_name=collection_name,
                            limit=10000  # Adjust limit based on how many files you expect
                        )
            # response is of type tuple thus
            response= response[0]

            print(type(response))
            # print(f' responses: {response}')
            # If the response is empty, return early
            if not response:
                print("could find any points")
                return

            # Extract the file names and timestamps
            
            selected_file = None
            min_timestamp = None

            # Find the file with the least timestamp
            for file in response:
                timestamp = file.payload['time']
                if timestamp:
                    if min_timestamp is None or timestamp < min_timestamp:
                        min_timestamp = timestamp
                        selected_file = file.payload['filename']
                        file_id = file.payload['id']

            # If a file was found, delete it
            if selected_file:
                print(f"deleting the file with name  {selected_file} and id {file_id}")
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="filename",
                                match=models.MatchValue(value=selected_file)
                            ),
                        ],
                    ),
                )
            return selected_file, file_id
        
    def delete_collection(self, collection_name):
        """ Adding colelction deleting function to delete a collection if it exists"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            if collection_name in collection_names:
                self.client.delete_collection(collection_name)
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")