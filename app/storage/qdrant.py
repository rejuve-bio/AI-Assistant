
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
COLBERT_VECTOR_SIZE=128
USER_COLLECTION = "user_memory_store"

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
class Qdrant:

    def __init__(self):

        try:
            self.client = QdrantClient(os.environ.get('QDRANT_CLIENT'))
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


    def upsert_data(self,collection_name,df):
        try:
            excluded_columns = {"dense"}
            payload_columns = [col for col in df.columns if col not in excluded_columns]
            payloads_list = [
                        {col: getattr(item, col) for col in payload_columns}
                        for item in df.itertuples(index=False)]

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
                ),)
            print("embedding saved")
        except:
                traceback.print_exc()
                print("error saving")


    def retrieve_data(self,collection,query):

        result = self.client.search(
                collection_name=collection,
                query_vector=query["dense"],
                with_payload=True,
                score_threshold=0.5,
                limit=1000)
        response = {}

        # Extracting and formatting the relevant points
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

        if memory_id:
                data = [{"content":data,"user_id":user_id}]
                self.client.upsert(
                    collection_name=USER_COLLECTION,
                    points=models.Batch(
                    ids=[memory_id],
                    vectors=embedding,
                    payloads=data,),)
                return memory_id

        data = [{"content":data,"user_id":user_id}]
        memory_id = [str(uuid.uuid4())]
        self.client.upsert(
                collection_name=USER_COLLECTION,
                points=models.Batch(
                    ids=memory_id,
                    vectors=embedding,
                    payloads=data,),)
        return memory_id

    def _delete_memory(self, memory_id):

        self.client.delete(
            collection_name=USER_COLLECTION,
            points_selector=models.PointIdsList(
                points=[memory_id],
            ),
        )
        return None

    def _retrieve_memory(self,user_id,embedding=None):
        print("user_id", user_id)
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
                                                key="user_id", match=models.MatchValue(value=user_id),)
                                                ]
                                            ),
                    limit=1000)

            if result:
                response = {}
                for i, point in enumerate(result):
                    response[i] = {
                        "id": point.id,
                        "content": point.payload.get('content'),
                        }

                return [response[0]]
            return response
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

