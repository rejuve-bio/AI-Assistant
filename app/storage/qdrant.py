from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import traceback
from dotenv import load_dotenv
import uuid
import logging

MAX_MEMORY_LIMIT = 10
MAX_PDF_LIMIT = 2
USER_COLLECTION = os.getenv("USER_COLLECTION", "USER_COLLECTIONS")
USER_MEMORY_NAME = "user memories"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class Qdrant:

    def __init__(self, embedding_model, vector_size=1536, batch_size=20):
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.vector_size = vector_size

        try:
            qdrant_url = os.environ.get(
                "QDRANT_CLIENT", "http://host.docker.internal:6333"
            )
            self.client = QdrantClient(
                url=qdrant_url,
                port=6333,
                grpc_port=6334,
                prefer_grpc=False,
            )
            logger.info(f"qdrant connected to {qdrant_url} (REST API)")
        except:
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

    def delete_pdf_by_id(self, collection_name, pdf_id):
        # Delete all points in the given collection (user_id) with the specified pdf_id.
        try:
            self.ensure_collection_exists(collection_name)
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="pdf_id",
                            match=models.MatchValue(value=pdf_id),
                        )
                    ]
                ),
            )
            logger.info(
                f"Deleted all points for pdf_id {pdf_id} in collection {collection_name}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Error deleting pdf_id {pdf_id} from collection {collection_name}: {e}"
            )
            return False

    def upsert_data(
        self,
        collection_name,
        data,
        is_pdf=False,
        chunks=None,
        metadata=None,
    ):
        """
        Unified method to upsert data to Qdrant collection.
        Handles both PDF chunks and general data (list of dicts).

        :param collection_name: The collection name
        :param data: List of dictionaries for general data, or None for PDF data
        :param is_pdf: Boolean indicating if this is PDF data
        :param chunks: List of text chunks (only for PDF data)
        :param metadata: Metadata dictionary (only for PDF data)
        """
        try:
            if is_pdf:
                # Handle PDF data using chunks and metadata
                if chunks is None or metadata is None:
                    raise ValueError("chunks and metadata are required for PDF data")

                self.ensure_collection_exists(collection_name)
                meta = metadata or {}

                # embed in batches to avoid big payloads
                for i in range(0, len(chunks), self.batch_size):
                    batch_chunks = chunks[i : i + self.batch_size]
                    embeddings = self._get_embeddings(batch_chunks)

                    points = []
                    for text, emb in zip(batch_chunks, embeddings):
                        # unique ID per chunk
                        point_id = str(uuid.uuid4())
                        points.append(
                            models.PointStruct(
                                id=point_id, vector=emb, payload={**meta, "text": text}
                            )
                        )
                    self.client.upsert(collection_name=collection_name, points=points)

                logger.info("PDF chunks saved")
                return "PDF Data Successfully Uploaded"
            else:
                # Handle general data (list of dicts)
                if data is None or not isinstance(data, list):
                    raise ValueError(
                        "data must be a list of dictionaries for non-PDF data"
                    )

                self.ensure_collection_exists(collection_name)

                # Process each item in the data list
                for item in data:
                    # Extract text content from the item - prioritize 'content' field for sample data
                    if "content" in item:
                        text = item["content"]
                    elif "text" in item:
                        text = item["text"]
                    elif "description" in item:
                        text = item["description"]
                    else:
                        # If no standard field, convert the whole item to string
                        text = str(item)

                    # Generate embedding for this text
                    embedding = self._get_embeddings([text])[0]

                    # Create payload with the item data and source info
                    payload = {
                        "text": text,
                        "source": "sample_data",
                        **item,
                    }

                    # Use the item's existing ID if available, otherwise generate a new one
                    point_id = item.get("id", str(uuid.uuid4()))

                    # Upsert this single point
                    self.client.upsert(
                        collection_name=collection_name,
                        points=[
                            models.PointStruct(
                                id=point_id, vector=embedding, payload=payload
                            )
                        ],
                    )

                logger.info("Sample data saved")
                return "Sample Data Successfully Uploaded"

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error saving: {e}")

    def retrieve_similar_content(
        self, collection_name, query, user_id=None, pdf_ids=None, top_k=10, filter=None
    ):
        """
        Unified retrieval method for Qdrant. Supports filtering by user_id, pdf_ids, or both.
        :param collection_name: The Qdrant collection to search.
        :param query: The query string or vector.
        :param user_id: Optional user ID to filter results.
        :param pdf_ids: Optional list of PDF IDs to filter results.
        :param top_k: Number of results to return.
        :param filter: If True, applies user/pdf filtering; if False, general search.
        :return: List of relevant results.
        """
        try:
            logger.info(
                f"Starting retrieve_similar_content for collection: {collection_name}"
            )

            # Ensure collection exists before searching
            self.ensure_collection_exists(collection_name)

            # Check if collection has any points
            try:
                collection_info = self.client.get_collection(collection_name)
                if collection_info.points_count == 0:
                    logger.info(
                        f"Collection '{collection_name}' is empty, returning empty results"
                    )
                    return []
            except Exception as e:
                logger.warning(
                    f"Could not check collection info for '{collection_name}': {e}"
                )

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
            if pdf_ids:
                filters.append(
                    models.FieldCondition(
                        key="pdf_id", match=models.MatchAny(any=pdf_ids)
                    )
                )
            query_filter = models.Filter(must=filters) if filters else None

            logger.info(
                f"Searching collection '{collection_name}' with {len(filters)} filters"
            )
            hits = self.client.search(
                collection_name=collection_name,
                query_vector=query,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            logger.info(f"Found {len(hits)} hits in collection '{collection_name}'")

            # Return a consistent format (list of payloads)
            return [h.payload for h in hits]
        except Exception as e:
            logger.error(
                f"Error in retrieve_similar_content for collection '{collection_name}': {e}"
            )
            traceback.print_exc()
            return []

    def _create_memory_update_memory(
        self, user_id, data, embedding, metadata, memory_id=None
    ):

        self.ensure_collection_exists(USER_COLLECTION)

        current_time = datetime.utcnow().isoformat()
        data = [
            {
                "content": data,
                "user_id": user_id,
                "created_at_updated_at": current_time,
                "status": USER_MEMORY_NAME,
            }
        ]
        if memory_id:
            self.client.upsert(
                collection_name=USER_COLLECTION,
                points=models.Batch(
                    ids=[memory_id],
                    vectors=embedding,
                    payloads=data,
                ),
            )
            return memory_id
        # check if a collection have top 10 collections
        try:
            memories = self.client.scroll(USER_COLLECTION, with_payload=True)
            if len(memories[0]) >= MAX_MEMORY_LIMIT:
                sorted_memories = sorted(
                    memories[0],
                    key=lambda memory: memory.payload["created_at_updated_at"],
                )
                # Delete the oldest memory
                oldest_memory_id = sorted_memories[0].id
                self._delete_memory(oldest_memory_id)

                logger.info(
                    f"older memory is being deleted since you have reached the limit {MAX_MEMORY_LIMIT}"
                )

            logger.info("uploading new memory")
            memory_id = [str(uuid.uuid4())]
            self.client.upsert(
                collection_name=USER_COLLECTION,
                points=models.Batch(
                    ids=memory_id,
                    vectors=embedding,
                    payloads=data,
                ),
            )
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

    def _retrieve_memory(self, user_id, embedding=None):
        try:
            if embedding:
                result = self.client.search(
                    collection_name=USER_COLLECTION,
                    query_vector=embedding,
                    with_payload=True,
                    # score threshold of 0.5 will return a similiar memories with similiarity score of more than 0.5
                    score_threshold=0.5,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id),
                            ),
                            models.FieldCondition(
                                key="status",
                                match=models.MatchValue(value=USER_MEMORY_NAME),
                            ),
                        ],
                    ),
                    limit=1000,
                )

                if result:
                    response = {}
                    for i, point in enumerate(result):
                        response[i] = {
                            "id": point.id,
                            "content": point.payload.get("content"),
                            "date": point.payload.get("created_at_updated_at"),
                        }

                    return [response[0]]
            else:
                data = self.client.scroll(
                    collection_name=USER_COLLECTION,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id", match=models.MatchValue(value=user_id)
                            ),
                        ]
                    ),
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )

                data = [record.payload["content"] for record in data[0][::-1]]
                return data
        except:
            traceback.print_exc()
            return None
