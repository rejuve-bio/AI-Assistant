from app.prompts.rag_prompts import RETRIEVE_PROMPT, RAG_REFLECTION_PROMPT
from app.storage.memory_layer import MemoryManager
import traceback
import json
import os
import logging
import uuid
from datetime import datetime
import fitz
from app.rag.utils.content_processor import ContentProcessor
from app.rag.utils.content_analyzer import ContentAnalyzer
from app.storage.mongo_storage import mongo_db_manager
from app.rag.utils.web_search import SimpleWebSearch


logger = logging.getLogger(__name__)


VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION")
USER_COLLECTION = os.getenv("USER_COLLECTION", "CHAT_MEMORY")
CONTENT_LIMIT = 10  # Total content limit (PDFs + web content)


class RAG:
    def __init__(self, llm, qdrant_client):
        """
        Initializes the RAG (Retrieval Augmented Generation) class.
        Uses the provided Qdrant client
        :param llm: An instance of the LLMInterface for generating responses.
        :param qdrant_client: The shared Qdrant client.
        """
        self.llm = llm
        self.client = qdrant_client
        self.content_processor = ContentProcessor()
        self.content_analyzer = ContentAnalyzer(self.llm)
        logger.info(
            "RAG initialized with LLM and shared Qdrant client/embedding model."
        )

    def _chunk_and_store(self, collection_name, chunks, metadata):
        return self.client.upsert_data(
            collection_name=collection_name,
            data=None,
            is_content=True,
            chunks=chunks,
            metadata=metadata,
        )

    def save_doc_to_rag(
        self,
        data,
        collection_name=None,
        is_content=False,
        pdf_path=None,
        file_name=None,
        user_id=None,
        content_id=None,
        is_web=False,
        web_content=None,
    ):
        """
        Unified method to save documents to RAG storage using the unified upsert_data method.

        :param data: The data to store (list of dicts for sample data, or None for content)
        :param collection_name: The collection name to store in
        :param is_content: Boolean indicating if this is content data (PDF/web)
        :param pdf_path: Path to PDF file (only for PDF data)
        :param file_name: Name of the file (only for PDF data)
        :param user_id: User ID (only for content data)
        :param is_web: Boolean indicating if this is web content
        :param web_content: Web content data (only for web data)
        :param content_id: Content ID (for content data)
        """
        if is_content and not is_web:
            if not (pdf_path and file_name and user_id and content_id):
                logger.error("Missing required parameters for PDF processing")
                return None
            chunks = self.content_processor.process_pdf(pdf_path)
            metadata = {
                "content_id": content_id,
                "filename": file_name,
                "user_id": user_id,
                "content_type": "pdf",
            }
            return self._chunk_and_store(collection_name, chunks, metadata)
        elif is_web:
            if not (web_content and user_id and content_id):
                logger.error("Missing required parameters for web content processing")
                return None
            result = self.content_processor.process_web_content(
                web_content.get("metadata", {}).get("url", "")
            )
            if not result:
                logger.error("Failed to process web content")
                return None
            metadata = {
                "content_id": content_id,
                "url": web_content.get("metadata", {}).get("url", ""),
                "title": web_content.get("metadata", {}).get("title", ""),
                "user_id": user_id,
                "content_type": "web",
            }
            return self._chunk_and_store(collection_name, result["chunks"], metadata)
        else:
            # Handle sample/general data using unified upsert_data method
            if isinstance(data, list) and all(isinstance(d, dict) for d in data):
                # Pass the list of dicts directly to upsert_data
                return self.client.upsert_data(
                    collection_name=collection_name,
                    data=data,
                    is_content=False,
                )
            else:
                logger.error("Invalid data format for sample data storage")
                return None

    def save_retrievable_docs(self, file, user_id):
        try:
            return_response = {"text": None, "resource": {}}

            # Check for duplicate files
            logger.info("checking if user files is already saved")
            pdf_files = mongo_db_manager.get_user_content_files(user_id, "pdf")
            if any(f.get("filename") == file.filename for f in pdf_files):
                return_response["text"] = "PDF already exists."
                return_response["resource"]["filename"] = file.filename
                logger.info("file is found from the mongodb data")
                return return_response

            # Check quota
            if mongo_db_manager.get_content_count(user_id) >= CONTENT_LIMIT:
                return_response["text"] = (
                    "Your quota is full. Maximum 10 content items allowed."
                )
                return_response["resource"]["count"] = (
                    mongo_db_manager.get_content_count(user_id)
                )
                return return_response

            content_id = str(uuid.uuid4())
            upload_folder = "pdfs_uploaded/pdfs"
            os.makedirs(upload_folder, exist_ok=True)
            pdf_path = os.path.join(upload_folder, f"{content_id}.pdf")
            file.save(pdf_path)

            # Get number of pages
            try:
                with fitz.open(pdf_path) as doc:
                    num_pages = doc.page_count
            except Exception:
                num_pages = None

            # Get upload time
            upload_time = datetime.now()

            # Get file size in MB
            try:
                file_size_bytes = os.path.getsize(pdf_path)
                file_size = round(file_size_bytes / (1024 * 1024), 2)
            except Exception:
                file_size = None

            full_text = self.content_processor.extract_text_from_pdf(pdf_path)
            analysis = self.content_analyzer.analyze_content(full_text, "pdf")

            logger.info("Analyzing content for keywords, summary adn suggested questions")
            file_analysis = {
                "content_id": content_id,
                "filename": file.filename,
                "keywords": analysis.get("keywords", ""),
                "topics": analysis.get("topics", ""),
                "summary": analysis.get("summary", ""),
                "suggested_questions": analysis.get("suggested_questions", ""),
                "num_pages": num_pages,
                "file_size": f"{file_size} MB",
                "upload_time": upload_time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Store in Qdrant using custom chunking logic
            saved = self.save_doc_to_rag(
                data=None,
                collection_name=user_id,
                is_content=True,
                pdf_path=pdf_path,
                file_name=file.filename,
                user_id=user_id,
                content_id=content_id,
            )
            
            if saved:
                logger.info("content saved to Qdrant")
                # Add PDF metadata to the database using unified table
                mongo_db_manager.add_content_file(
                    user_id=user_id,
                    content_id=content_id,
                    content_type="pdf",
                    filename=file.filename,
                    num_pages=num_pages,
                    upload_time=upload_time,
                    summary=analysis.get("summary"),
                    keywords=str(analysis.get("keywords", [])),
                    topics=str(analysis.get("topics", [])),
                    metadata={
                        "file_size": file_size,
                        "suggested_questions": str(analysis.get("suggested_questions", [])),
                    },
                )

            # Add memory for the upload
            MemoryManager(self.llm).add_memory(f"pdf file : {file.filename}", user_id)

            # Add a history entry for the PDF upload
            mongo_db_manager.create_history(
                user_id=user_id,
                user_message=f"Uploaded PDF: {file.filename}",
                assistant_answer=f"PDF '{file.filename}' uploaded successfully.",
            )

            return_response["text"] = "PDF uploaded successfully."
            return_response["resource"] = file_analysis
            return return_response
        except Exception as e:
            logger.error(f"Error in save_retrievable_docs: {e}")
            import traceback

            traceback.print_exc()
            return {"text": f"Error uploading PDF: {str(e)}"}

    def save_web_content(self, url, user_id):
        try:
            return_response = {"text": None, "resource": {}}

            # Validate URL
            is_valid, error_msg = self.content_processor.validate_url(url)
            if not is_valid:
                return_response["text"] = f"Invalid URL: {error_msg}"
                return return_response

            # Check for duplicate URLs
            content_files = mongo_db_manager.get_user_content_files(user_id, "web")
            if any(f.get("url") == url for f in content_files):
                return_response["text"] = "URL already exists."
                return_response["resource"]["url"] = url
                return return_response

            # Check quota
            if mongo_db_manager.get_content_count(user_id) >= CONTENT_LIMIT:
                return_response["text"] = (
                    "Your quota is full. Maximum 10 content items allowed."
                )
                return_response["resource"]["count"] = (
                    mongo_db_manager.get_content_count(user_id)
                )
                return return_response

            content_id = str(uuid.uuid4())
            upload_time = datetime.now()

            web_content = self.content_processor.extract_text_from_url(url)
            if not web_content:
                return_response["text"] = "Failed to extract content from URL."
                return return_response

            cleaned_text = self.content_processor.clean_text_content(
                web_content["text"]
            )
            analysis = self.content_analyzer.analyze_content(cleaned_text, "web")

            web_analysis = {
                "content_id": content_id,
                "url": url,
                "title": web_content["metadata"].get("title", "No title") or "No title",
                "author": web_content["metadata"].get("author", "Unknown") or "Unknown",
                "keywords": analysis.get("keywords", ""),
                "topics": analysis.get("topics", ""),
                "summary": analysis.get("summary", ""),
                "suggested_questions": analysis.get("suggested_questions", ""),
                "upload_time": upload_time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Store in Qdrant using custom chunking logic
            saved = self.save_doc_to_rag(
                data=None,
                collection_name=user_id,
                is_web=True,
                web_content=web_content,
                user_id=user_id,
                content_id=content_id,
            )

            if saved:
                logger.info("content saved to Qdrant")
                # Add web content metadata to the database
                mongo_db_manager.add_content_file(
                    user_id=user_id,
                    content_id=content_id,
                    content_type="web",
                    url=url,
                    title=web_content["metadata"].get("title") or None,
                    author=web_content["metadata"].get("author") or None,
                    upload_time=upload_time,
                    summary=analysis.get("summary"),
                    keywords=str(analysis.get("keywords", [])),
                    topics=str(analysis.get("topics", [])),
                    metadata={
                        "suggested_questions": str(analysis.get("suggested_questions", [])),
                    },
                )

            # Add memory for the upload
            MemoryManager(self.llm).add_memory(f"web content : {url}", user_id)

            # Add a history entry for the web content upload
            mongo_db_manager.create_history(
                user_id=user_id,
                user_message=f"Added web content: {url}",
                assistant_answer=f"Web content from '{url}' added successfully.",
            )

            return_response["text"] = "Web content added successfully."
            return_response["resource"] = web_analysis
            return return_response
        except Exception as e:
            logger.error(f"Error in save_web_content: {e}")
            import traceback

            traceback.print_exc()
            return {"text": f"Error adding web content: {str(e)}"}

    def query(
        self,
        query_str: str,
        user_id=None,
        filter=False,
        content_ids=None,
    ):
        """
        Unified query method for retrieving similar content from Qdrant.
        :param query_str: The query string to process.
        :param user_id: The ID of the user making the query.
        :param content_ids: Optional list of content IDs to filter user content.
        :return: List of relevant results.
        """
        try:
            if filter:
                # User content collection, optionally filtered by content_ids
                return self.client.retrieve_similar_content(
                    collection_name=user_id,
                    query=query_str,
                    user_id=user_id,
                    content_ids=content_ids,
                    top_k=10,
                )
            else:
                # General collection
                return self.client.retrieve_similar_content(
                    collection_name=VECTOR_COLLECTION,
                    query=query_str,
                    top_k=10,
                )
        except Exception as e:
            logger.error(f"An error occurred during query processing: {e}")
            traceback.print_exc()
            return []

    def _reflect_and_revise(self, query_str: str, retrieved_content: list, initial_answer: str) -> tuple:
        """
        Critic step: evaluate the initial RAG answer for grounding, completeness,
        and accuracy against the source chunks.

        Returns a tuple of (final_answer: str, confidence: float) — either the
        original (approved) or a revised version generated from the critic's
        specific feedback, along with a confidence score (0.0–1.0).
        """
        try:
            reflection_prompt = RAG_REFLECTION_PROMPT.format(
                query=query_str,
                retrieved_content=retrieved_content,
                generated_answer=initial_answer,
            )
            verdict_raw = self.llm.generate(reflection_prompt)

            # --- Parse the structured verdict ---
            # The LLM wrapper may have already parsed JSON into a dict,
            # or it may still be a raw string.
            verdict_dict = None
            if isinstance(verdict_raw, dict):
                verdict_dict = verdict_raw
            elif isinstance(verdict_raw, str):
                cleaned = verdict_raw.strip()
                try:
                    verdict_dict = json.loads(cleaned)
                except (json.JSONDecodeError, ValueError):
                    pass

            if isinstance(verdict_dict, dict) and "verdict" in verdict_dict:
                verdict_label = str(verdict_dict.get("verdict", "")).upper()
                confidence = float(verdict_dict.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]

                logger.info(
                    f"RAG reflection verdict: {verdict_label} (confidence={confidence:.2f})"
                )

                if verdict_label == "GOOD":
                    logger.info("RAG reflection: initial answer approved.")
                    return initial_answer, confidence

                if verdict_label == "REVISE":
                    feedback = str(verdict_dict.get("feedback", "Improve grounding."))
                    logger.info(f"RAG reflection: revising answer. Feedback: {feedback}")

                    revision_prompt = (
                        RETRIEVE_PROMPT.format(
                            query=query_str,
                            retrieved_content=retrieved_content,
                        )
                        + f"\n\nAdditional instruction: {feedback}"
                    )
                    revised = self.llm.generate(revision_prompt)
                    if isinstance(revised, dict):
                        revised = str(revised)
                    # Revised answer gets a small confidence boost over the original
                    revised_confidence = min(1.0, confidence + 0.15)
                    return (revised.strip() if revised else initial_answer), revised_confidence

            # --- Fallback: handle old plain-text format for backward compat ---
            verdict_str = str(verdict_raw).strip() if verdict_raw else ""
            logger.info(f"RAG reflection verdict (text fallback): {verdict_str[:120]}")

            if verdict_str.upper().startswith("GOOD"):
                logger.info("RAG reflection: initial answer approved.")
                return initial_answer, 0.75  # default confidence for unscored approval

            if verdict_str.upper().startswith("REVISE:"):
                feedback = verdict_str[len("REVISE:"):].strip()
                logger.info(f"RAG reflection: revising answer. Feedback: {feedback}")
                revision_prompt = (
                    RETRIEVE_PROMPT.format(
                        query=query_str,
                        retrieved_content=retrieved_content,
                    )
                    + f"\n\nAdditional instruction: {feedback}"
                )
                revised = self.llm.generate(revision_prompt)
                if isinstance(revised, dict):
                    revised = str(revised)
                return (revised.strip() if revised else initial_answer), 0.6

            # Unexpected format
            logger.warning(
                f"RAG reflection returned unexpected format: '{verdict_str[:80]}' — using initial answer."
            )
            return initial_answer, 0.5

        except Exception as e:
            logger.error(f"RAG reflection step failed, using initial answer: {e}", exc_info=True)
            return initial_answer, 0.5

    def get_result_from_rag(self, query_str: str, user_id: str, content_ids=None):
        """
        Retrieves the result for a query by calling the query method
        and generating a response based on the retrieved content.
        :param query_str: The query string to process.
        :param user_id: The ID of the user making the request.
        :param content_ids: Optional list of content IDs to filter user content.
        :return: The result from the LLM generated based on the query and retrieved content.
        """
        try:
            logger.info("Generating result for the query.")
                
            result1 = []  # Initialize as empty
            result2 = []
            content_sources = []
            
            if content_ids:
                # Only query user collection with content_ids
                logger.info(f"Generating result for the query from the specified content {content_ids}.")
                result2 = self.query(
                    query_str=query_str,
                    user_id=user_id,
                    filter=True,
                    content_ids=content_ids,
                )
                
                if not isinstance(content_ids, list):
                    content_ids = [content_ids]

                for content_id in content_ids:
                    doc = mongo_db_manager.get_content_file_by_id(
                        user_id=user_id,
                        content_id=content_id
                    )

                    if not doc:
                        continue

                    content_sources.append({
                        "content_id": content_id,
                        "filename": doc.get("filename"),
                        "summary": doc.get("summary"),
                        "topics": doc.get("topics", ""),
                        "suggested_questions": doc.get("suggested_questions", "")
                    })
            else:
                # No content_ids - query general
                result1 = self.query(query_str=query_str, user_id=user_id)

                  
            logger.info(f"Query executed successfully. result1 and result2 obtained. {result1} {result2}")
            # Combine both results (general + user content)
            combined_results = []
            if isinstance(result1, list):
                combined_results.extend(result1)
            if isinstance(result2, list):
                combined_results.extend(result2)
            if not combined_results:
                logger.error("No query result to process.")
                return None          

            prompt = RETRIEVE_PROMPT.format(
                query=query_str, retrieved_content=combined_results
            )
            result = self.llm.generate(prompt)

            logger.info(f"Initial RAG answer generated.")

            # --- Reflection loop ---
            # Validate and potentially revise the initial answer before returning it.
            confidence = 0.5  # default confidence
            if isinstance(result, str) and result.strip():
                result, confidence = self._reflect_and_revise(query_str, combined_results, result)
            # ----------------------

            logger.info(f"Result generated successfully (confidence={confidence:.2f}). {result}")
            response = {
                "text": result,
                "confidence": confidence,
                "resource": {
                    "type": "RAG",
                    "content_sources": content_sources
                }
            }
            return response
        except Exception as e:
            logger.error(f"An error occurred while generating the result: {e}")
            traceback.print_exc()
            return None
