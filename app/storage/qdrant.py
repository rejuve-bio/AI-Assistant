
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

OPEN_AI_VECTOR_SIZE=1536
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
            self.client = QdrantClient(os.environ.get('QDRANT_CLIENT','http://qdrant:6333'))
            print(f"qdrant connected")
        except:
            print('qdrant connection is failed')


    def get_create_collection(self,collection_name):

        try:
            self.client.get_collection(collection_name)
        except:
            print("no such collection exists")
            try:
                logger.info(f"creating collection {collection_name}")
                self.client.create_collection(
                    collection_name,
                    vectors_config=models.VectorParams(size=OPEN_AI_VECTOR_SIZE, distance=models.Distance.DOT) )
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
                        for payload in payloads_list:
                            payload["user_id"] = user_id
                        
                    import random
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
            
    def retrieve_data(self,collection, query,user_id,filter=None):
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
        for i, point in enumerate(result):
            response[i] = {
                "id": point.id,
                "score": point.score,
                "authors": point.payload.get('authors', 'Unknown'),
                "content": point.payload.get('content', 'No content available')
            }
        return response

    def _create_memory_update_memory(self,user_id,data, embedding, metadata,memory_id=None):

        self.get_create_collection(USER_COLLECTION)

        current_time = datetime.utcnow().isoformat()
        data = [{"content": data, "user_id": user_id, "created_at_updated_at": current_time, "status":USER_MEMORY_NAME}]
        if memory_id:
                self.client.upsert(
                    collection_name=USER_COLLECTION,
                    points=models.Batch(
                    ids=[memory_id],
                    vectors=embedding,
                    payloads=data,),)
                return memory_id
        # check if a collection have top 10 collections
        try:
            memories = self.client.scroll(USER_COLLECTION, with_payload=True)
            if len(memories[0]) >= MAX_MEMORY_LIMIT:
                sorted_memories = sorted(
                    memories[0],
                    key=lambda memory: memory.payload["created_at_updated_at"]
                )
                # Delete the oldest memory
                oldest_memory_id = sorted_memories[0].id
                self._delete_memory(oldest_memory_id)

                logger.info(f"older memory is being deleted since you have reached the limit {MAX_MEMORY_LIMIT}")

            logger.info("uploading new memory")
            memory_id = [str(uuid.uuid4())]
            self.client.upsert(
                    collection_name=USER_COLLECTION,
                    points=models.Batch(
                        ids=memory_id,
                        vectors=embedding,
                        payloads=data,),)
            logger.info("collection updated")
            return memory_id
        except:
            traceback.print_exc()


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
