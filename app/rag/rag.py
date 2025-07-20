from app.prompts.rag_prompts import SYSTEM_PROMPT, RETRIEVE_PROMPT
from app.storage.memory_layer import MemoryManager
import traceback
import os
import logging
import re
import json
import uuid
from datetime import datetime
import fitz
from app.rag.utils.pdf_processor import extract_text_from_pdf, chunk_text
from app.rag.utils.pdf_analyzer import PDFAnalyzer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "SITE_INFORMATION")
USER_COLLECTION = os.getenv("USER_COLLECTION", "CHAT_MEMORY")
USERS_PDF_COLLECTION = os.getenv("PDF_COLLECTION", "PDF_COLLECTION")
PDF_LIMIT = 5
USER_PDF_FILE = "rag_user_pdf.json"


def load_user_pdf():
    if os.path.exists(USER_PDF_FILE):
        with open(USER_PDF_FILE, "r") as f:
            return json.load(f)
    return {}


def save_user_pdf(user_pdf_data):
    with open(USER_PDF_FILE, "w") as f:
        json.dump(user_pdf_data, f, indent=4)


def update_user_data(user_id, new_user_data):
    try:
        current_data = load_user_pdf()
        current_data[user_id] = new_user_data
        save_user_pdf(current_data)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Error updating user data: {e}")
        return False


def get_user_data(user_id):
    current_data = load_user_pdf()
    return current_data.get(user_id, {"count": 0, "files": []})


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
        logger.info(
            "RAG initialized with LLM and shared Qdrant client/embedding model."
        )

    def save_doc_to_rag(
        self,
        data,
        collection_name=None,
        is_pdf=False,
        pdf_path=None,
        file_name=None,
        user_id=None,
        pdf_id=None,
    ):
        """
        Unified method to save documents to RAG storage using the unified upsert_data method.

        :param data: The data to store (list of dicts for sample data, or None for PDF)
        :param collection_name: The collection name to store in
        :param is_pdf: Boolean indicating if this is PDF data
        :param pdf_path: Path to PDF file (only for PDF data)
        :param file_name: Name of the file (only for PDF data)
        :param user_id: User ID (only for PDF data)
        :param pdf_id: PDF ID (only for PDF data)
        """
        if is_pdf:
            # Handle PDF data using custom extraction and chunking
            if pdf_path and file_name and user_id and pdf_id:
                text = extract_text_from_pdf(pdf_path)
                chunks = chunk_text(text)
                metadata = {
                    "pdf_id": pdf_id,
                    "filename": file_name,
                    "user_id": user_id,
                }
                # Use unified upsert_data method for PDF
                return self.client.upsert_data(
                    collection_name=collection_name,
                    data=None,
                    is_pdf=True,
                    chunks=chunks,
                    metadata=metadata,
                )
            else:
                logger.error("Missing required parameters for PDF processing")
                return None
        else:
            # Handle sample/general data using unified upsert_data method
            if isinstance(data, list) and all(isinstance(d, dict) for d in data):
                # Pass the list of dicts directly to upsert_data
                return self.client.upsert_data(
                    collection_name=collection_name,
                    data=data,
                    is_pdf=False,
                )
            else:
                logger.error("Invalid data format for sample data storage")
                return None

    def save_retrievable_docs(self, file, user_id):
        try:
            logger = logging.getLogger(__name__)
            return_response = {"text": None, "resource": {}}

            # Check for duplicate files
            if any(
                f["filename"] == file.filename for f in get_user_data(user_id)["files"]
            ):
                return_response["text"] = "PDF already exists."
                return_response["resource"]["filename"] = file.filename
                return return_response

            # Check quota
            if get_user_data(user_id)["count"] >= PDF_LIMIT:
                return_response["text"] = "Your quota is full. Maximum 5 PDFs allowed."
                return_response["resource"]["count"] = get_user_data(user_id)["count"]
                return return_response

            pdf_id = str(uuid.uuid4())
            upload_folder = "storage/pdfs"
            os.makedirs(upload_folder, exist_ok=True)
            pdf_path = os.path.join(upload_folder, f"{pdf_id}.pdf")
            file.save(pdf_path)

            # Get number of pages
            try:
                with fitz.open(pdf_path) as doc:
                    num_pages = doc.page_count
            except Exception:
                num_pages = None

            # Get upload time
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Get file size in MB
            try:
                file_size_bytes = os.path.getsize(pdf_path)
                file_size = round(file_size_bytes / (1024 * 1024), 2)
            except Exception:
                file_size = None

            # Use PDFAnalyzer for summary and analysis
            pdf_analyzer = PDFAnalyzer()
            full_text = extract_text_from_pdf(pdf_path)
            summary = pdf_analyzer.generate_summary(
                full_text, user_id=user_id, pdf_id=pdf_id
            )
            analysis = pdf_analyzer.analyze_pdf_content(full_text)
            analysis["summary"] = summary

            file_analysis = {
                "pdf_id": pdf_id,
                "filename": file.filename,
                "keywords": analysis.get("keywords", ""),
                "topics": analysis.get("topics", ""),
                "summary": analysis.get("summary", ""),
                "suggested_questions": analysis.get("suggested_questions", ""),
                "num_pages": num_pages,
                "file_size": f"{file_size} MB",
                "upload_time": upload_time,
            }

            # Store in Qdrant using custom chunking logic
            self.save_doc_to_rag(
                data=None,
                collection_name=user_id,
                is_pdf=True,
                pdf_path=pdf_path,
                file_name=file.filename,
                user_id=user_id,
                pdf_id=pdf_id,
            )

            # update user tracking for each file
            user_data = get_user_data(user_id)
            update_user_data(
                user_id,
                {
                    "count": user_data["count"] + 1,
                    "files": user_data["files"]
                    + [
                        {
                            "filename": file.filename,
                            "pdf_id": pdf_id,
                            "num_pages": num_pages,
                            "file_size": file_size,
                            "upload_time": upload_time,
                            "summary": analysis.get("summary", ""),
                        }
                    ],
                },
            )

            # Add memory for the upload
            MemoryManager(self.llm).add_memory(f"pdf file : {file.filename}", user_id)

            return_response["text"] = "PDF uploaded successfully."
            return_response["resource"] = file_analysis
            return return_response
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in save_retrievable_docs: {e}")
            import traceback

            traceback.print_exc()
            return {"text": f"Error uploading PDF: {str(e)}"}

    def query(
        self,
        query_str: str,
        user_id=None,
        filter=None,
        pdf_ids=None,
    ):
        """
        Unified query method for retrieving similar content from Qdrant.
        :param query_str: The query string to process.
        :param user_id: The ID of the user making the query.
        :param pdf_ids: Optional list of PDF IDs to filter user PDFs.
        :return: List of relevant results.
        """
        try:
            if filter:
                # User PDF collection, optionally filtered by pdf_ids
                return self.client.retrieve_similar_content(
                    collection_name=user_id,
                    query=query_str,
                    user_id=user_id,
                    pdf_ids=pdf_ids,
                    top_k=10,
                    filter=True,
                )
            else:
                # General collection
                return self.client.retrieve_similar_content(
                    collection_name=VECTOR_COLLECTION,
                    query=query_str,
                    top_k=10,
                    filter=False,
                )
        except Exception as e:
            logger.error(f"An error occurred during query processing: {e}")
            traceback.print_exc()
            return []

    def get_result_from_rag(self, query_str: str, user_id: str, pdf_ids=None):
        """
        Retrieves the result for a query by calling the query method
        and generating a response based on the retrieved content.
        :param query_str: The query string to process.
        :param user_id: The ID of the user making the request.
        :param pdf_ids: Optional list of PDF IDs to filter user PDFs.
        :return: The result from the LLM generated based on the query and retrieved content.
        """
        try:
            logger.info("Generating result for the query.")
            result1 = self.query(query_str=query_str, user_id=user_id)
            result2 = self.query(
                query_str=query_str, user_id=user_id, filter=True, pdf_ids=pdf_ids
            )
            # Combine both results (general + user PDFs)
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
            logger.info("Result generated successfully.")
            response = {"text": result}
            return response
        except Exception as e:
            logger.error(f"An error occurred while generating the result: {e}")
            traceback.print_exc()
            return None
