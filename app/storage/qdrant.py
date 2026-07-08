from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import traceback
from dotenv import load_dotenv
import uuid
import logging


logger = logging.getLogger(__name__)

load_dotenv()


class Qdrant:

    def __init__(self, embedding_model, vector_size=1536, batch_size=20):
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.vector_size = vector_size

        try:
            qdrant_url = os.environ.get(
                "QDRANT_CLIENT"
            )
            self.client = QdrantClient(
                url=qdrant_url,
                port=6333,
                grpc_port=6334,
                prefer_grpc=False,
            )
            logger.info(f"qdrant connected to {qdrant_url} (REST API)")
        except Exception:
            logger.error("qdrant connection is failed")

    def _get_embeddings(self, texts):
        return self.embedding_model(texts)

    def ensure_collection_exists(
        self, collection_name, vector_size=None, distance=None
    ):
        # Checks if a collection exists in Qdrant, and creates it if it does not.
        try:
            logger.info(f"Checking if collection '{collection_name}' exists...")
            self.client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' exists.")
        except Exception as e:
            logger.info(
                f"Collection '{collection_name}' does not exist. Creating it... Error: {e}"
            )
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size or self.vector_size,
                        distance=distance or models.Distance.COSINE,
                    ),
                )
                logger.info(f"Collection '{collection_name}' CREATED successfully.")
            except Exception as create_error:
                logger.error(
                    f"Failed to create collection '{collection_name}': {create_error}"
                )
                raise create_error

    def delete_content_by_id(self, collection_name, content_id):
        """
        Delete all points in the given collection with the specified content_id.
        Works for both PDF and web content.
        """
        try:
            self.ensure_collection_exists(collection_name)

            self.client.delete(
                collection_name=collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content_id",
                            match=models.MatchValue(value=content_id),
                        )
                    ]
                ),
            )
            logger.info(
                f"Deleted all points for content_id {content_id} in collection {collection_name}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Error deleting content_id {content_id} from collection {collection_name}: {e}"
            )
            return False

    def _upsert_content_data(self, collection_name, chunks, metadata):
        if chunks is None or metadata is None:
            raise ValueError("chunks and metadata are required for content data")

        self.ensure_collection_exists(collection_name)
        meta = metadata or {}

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            embeddings = self._get_embeddings(batch_chunks)

            points = []
            for text, emb in zip(batch_chunks, embeddings):
                point_id = str(uuid.uuid4())
                points.append(
                    models.PointStruct(
                        id=point_id, vector=emb, payload={**meta, "text": text}
                    )
                )
            self.client.upsert(collection_name=collection_name, points=points)

        logger.info("Content chunks saved")
        return "Content Data Successfully Uploaded"

    def _upsert_general_data(self, collection_name, data):
        if data is None or not isinstance(data, list):
            raise ValueError("data must be a list of dictionaries for non-content data")

        self.ensure_collection_exists(collection_name)
        total = len(data)
        logger.info("Embedding and uploading %d items to '%s'...", total, collection_name)

        for i, item in enumerate(data):
            if "content" in item:
                text = item["content"]
            elif "text" in item:
                text = item["text"]
            elif "description" in item:
                text = item["description"]
            else:
                text = str(item)

            if i % 10 == 0:
                logger.info("Uploading item %d/%d...", i + 1, total)

            embedding = self._get_embeddings([text])[0]

            payload = {"source": "sample_data", **item}
            doc_id = item.get("id", str(uuid.uuid4()))
            chunk_num = item.get("Chunk_Number")
            if chunk_num is not None:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}-{chunk_num}"))
            else:
                point_id = doc_id
            self.client.upsert(
                collection_name=collection_name,
                points=[models.PointStruct(id=point_id, vector=embedding, payload=payload)],
            )

        logger.info("Sample data saved — %d items uploaded to '%s'", total, collection_name)
        return "Sample Data Successfully Uploaded"

    def upsert_data(
        self,
        collection_name,
        data,
        is_content=False,
        chunks=None,
        metadata=None,
    ):
        """
        Unified method to upsert data to Qdrant collection.
        Handles both content chunks and general data (list of dicts).

        :param collection_name: The collection name
        :param data: List of dictionaries for general data, or None for content data
        :param is_content: Boolean indicating if this is content data (PDF/web)
        :param chunks: List of text chunks (only for content data)
        :param metadata: Metadata dictionary (only for content data)
        """
        try:
            if is_content:
                return self._upsert_content_data(collection_name, chunks, metadata)
            else:
                return self._upsert_general_data(collection_name, data)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error saving: {e}")

    def retrieve_similar_content(
        self,
        collection_name,
        query,
        user_id=None,
        content_ids=None,
        top_k=10,
    ):
        """
        Unified retrieval method for Qdrant. Supports filtering by user_id, content_ids, or combinations.
        :param collection_name: The Qdrant collection to search.
        :param query: The query string or vector.
        :param user_id: Optional user ID to filter results.
        :param content_ids: Optional list of content IDs to filter results (for any content type).
        :param top_k: Number of results to return.
        :param filter: If True, applies user/content filtering; if False, general search.
        :return: List of relevant results.
        """
        try:
            logger.info(
                f"Starting retrieve_similar_content for collection: {collection_name}"
            )

            # Check existence without creating — avoids empty collections for every querying user
            try:
                collection_info = self.client.get_collection(collection_name)
                if collection_info.points_count == 0:
                    logger.info(
                        f"Collection '{collection_name}' is empty, returning empty results"
                    )
                    return []
            except Exception:
                logger.info(
                    f"Collection '{collection_name}' does not exist, returning empty results"
                )
                return []

            # If query is a string, embed it
            if isinstance(query, str):
                query = self._get_embeddings([query])[0]
            elif isinstance(query, list) and isinstance(query[0], str):
                query = self._get_embeddings(query)[0]

            filters = []
            if user_id:
                filters.append(
                    models.FieldCondition(
                        key="user_id", match=models.MatchValue(value=user_id)
                    )
                )
            if content_ids:
                filters.append(
                    models.FieldCondition(
                        key="content_id", match=models.MatchAny(any=content_ids)
                    )
                )
            query_filter = models.Filter(must=filters) if filters else None

            logger.info(
                f"Searching collection '{collection_name}' with {len(filters)} filters"
            )
            hits = self.client.query_points(
                collection_name=collection_name,
                query=query,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                score_threshold=0.35
                if query_filter is None else None
            )
            logger.info(f"Found {len(hits.points)} hits in collection '{collection_name}'")
            return [h.payload for h in hits.points]

        except Exception as e:
            logger.error(
                f"Error in retrieve_similar_content for collection '{collection_name}': {e}"
            )
            traceback.print_exc()
            return []

