
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import traceback
import pandas as pd
from typing import List
from qdrant_client.models import PointStruct, PointIdsList
from dotenv import load_dotenv


OPEN_AI_VECTOR_SIZE=1536
COLBERT_VECTOR_SIZE=128
GEMINI_VECTOR_SIZE=768

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

    def retrieve_user_data(self,collection_name, query,user_id):
        result = self.client.search(
                collection_name=collection_name,
                query_vector=query["dense"],
                with_payload=True,
                score_threshold=0.5,
                limit=1000)
        response = {}
        
        # add user payload filter

        for i, point in enumerate(result):
            response[i] = {
                "score": point.score,
                "content": point.payload.get('content', 'No content available')
            }
        return response

