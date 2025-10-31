from app.lib.auth import token_required
from flask import Blueprint, request, current_app, jsonify, Response
from dotenv import load_dotenv
import traceback
import json
import os
from app.rag.utils.tts_utils import tts_manager
from app.storage.mongo_storage import mongo_db_manager
from app.storage.redis import redis_manager
from app.storage.history_manager import HistoryManager

load_dotenv()
main_bp = Blueprint("main", __name__)


@main_bp.route("/query", methods=["POST"])
@token_required
def process_query(current_user_id, auth_token):
    """
    Unified question answering endpoint for the Rejuve platform.

    - Accepts form data.
    - Required fields:
        - user_id: The user's identifier (string).
        - question: The user's question or prompt (string).
        - context: JSON string with keys:
            - id: For content queries, a list of content IDs; for other resources, a single ID.
            - resource: The type of resource (e.g., 'content', 'annotation', 'hypothesis').
        - graph, json_query: Optional, for advanced queries.
    - For content queries (resource == 'content'), content_ids are extracted from context['id'].
    - If content_ids are provided, answers are retrieved only from those content items; otherwise, answers are retrieved from all collections (user and general).
    - Handles both user-uploaded content question answering and general knowledge queries.
    """
    try:
        ai_assistant = current_app.config["ai_assistant"]
        data = request.form

        user_id = current_user_id
        question = data.get("question") or data.get("query")
        context_raw = data.get("context", "{}")
        try:
            context = json.loads(context_raw)
        except Exception:
            context = {}
        context_id = context.get("id", None)
        resource = context.get("resource", "annotation")
        # Code-exec specific optional fields
        urls = request.form.getlist("urls") if resource == "code_exec" else []
        options_raw = data.get("options", "{}") if resource == "code_exec" else "{}"
        try:
            options = json.loads(options_raw)
        except Exception:
            options = {}
        graph = data.get("graph", None)
        json_query = data.get("json_query", None)

        # Determine content_ids if resource is content and id is a list or string
        content_ids = None
        if resource == "content":
            if isinstance(context_id, list):
                content_ids = context_id
            elif isinstance(context_id, str):
                # If it's a comma-separated string, split it
                if context_id.strip().startswith("["):
                    try:
                        content_ids = json.loads(context_id)
                    except Exception:
                        content_ids = [context_id]
                else:
                    content_ids = [
                        cid.strip() for cid in context_id.split(",") if cid.strip()
                    ]

        # Ensure query exists before processing
        if not question and not json_query:
            return jsonify({"error": "No query provided."}), 400

        # Pass all relevant arguments to ai_assistant
        response = ai_assistant.assistant_response(
            query=question,
            user_id=user_id,
            token=auth_token,
            graph_id=context_id if resource != "content" else None,
            graph=graph,
            resource=resource,
            json_query=json_query,
            content_ids=content_ids,
            files=None,
            urls=urls or None,
            options=options or None,
        )

        return jsonify(response)
    except Exception as e:
        current_app.logger.error(f"Exception: {e}")
        traceback.print_exc()
        return f"Bad Response: {e}", 400


def save_data_file(uploaded_file, user_id, file_type):
    """Helper function to save CSV/HTML/XML files for code execution"""
    try:
        return_response = {"text": None, "resource": {}}
        
        # Check for duplicate files
        existing_files = mongo_db_manager.get_user_content_files(user_id, file_type)
        if any(f.get("filename") == uploaded_file.filename for f in existing_files):
            return_response["text"] = f"{file_type.upper()} file already exists."
            return_response["resource"]["filename"] = uploaded_file.filename
            return return_response
        
        # Check quota
        if mongo_db_manager.get_content_count(user_id) >= 10:
            return_response["text"] = "Your quota is full. Maximum 10 content items allowed."
            return_response["resource"]["count"] = mongo_db_manager.get_content_count(user_id)
            return return_response
        
        import uuid
        from datetime import datetime
        
        content_id = str(uuid.uuid4())
        upload_folder = f"storage/data_files/{file_type}"
        os.makedirs(upload_folder, exist_ok=True)
        
        # Get file extension
        ext = os.path.splitext(uploaded_file.filename)[1].lower()
        file_path = os.path.join(upload_folder, f"{content_id}{ext}")
        uploaded_file.save(file_path)
        
        upload_time = datetime.now()
        
        # Get file size in MB
        try:
            file_size_bytes = os.path.getsize(file_path)
            file_size = round(file_size_bytes / (1024 * 1024), 2)
        except Exception:
            file_size = None
        
        file_analysis = {
            "content_id": content_id,
            "filename": uploaded_file.filename,
            "file_path": file_path,
            "file_size": f"{file_size} MB" if file_size else None,
            "upload_time": upload_time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Store metadata in MongoDB
        mongo_db_manager.add_content_file(
            user_id=user_id,
            content_id=content_id,
            content_type=file_type,
            filename=uploaded_file.filename,
            file_size=file_size,
            upload_time=upload_time,
        )
        
        # Add history entry
        from app.storage.history_manager import HistoryManager
        HistoryManager().create_history(
            user_id=user_id,
            user_message=f"Uploaded {file_type.upper()}: {uploaded_file.filename}",
            assistant_answer=f"{file_type.upper()} file '{uploaded_file.filename}' uploaded successfully.",
        )
        
        return_response["text"] = f"{file_type.upper()} file uploaded successfully."
        return_response["resource"] = file_analysis
        return return_response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"text": f"Error uploading {file_type} file: {str(e)}"}


@main_bp.route("/upload_content", methods=["POST"])
@token_required
def upload_content(current_user_id, auth_token):
    """
    Unified endpoint for uploading PDF, CSV, HTML, XML files and web content
    Accepts either files (PDFs, CSV, HTML, XML) or URLs (web content)
    accepts a question to answer after upload (optional)
    """
    try:
        user_id = current_user_id
        question = request.form.get(
            "question"
        )  # Optional question to answer after upload
        if not user_id:
            return jsonify(error="Missing user_id"), 400

        ai_assistant = current_app.config["ai_assistant"]
        results = []

        # Handle file uploads (PDF, CSV, HTML, XML)
        if "files" in request.files:
            files = request.files.getlist("files")
            for uploaded in files:
                if uploaded.filename:
                    filename_lower = uploaded.filename.lower()
                    
                    # Handle PDF files (with RAG indexing)
                    if filename_lower.endswith(".pdf"):
                        response = ai_assistant.rag.save_retrievable_docs(uploaded, user_id)
                        item = {"filename": uploaded.filename, "response": response}

                        # Handle question answering for both new uploads and duplicates
                        if question and isinstance(response, dict):
                            content_id = None
                            is_duplicate = response.get("text") == "PDF already exists."

                            if is_duplicate:
                                # Get the existing content ID for this filename
                                pdf_files = mongo_db_manager.get_user_content_files(
                                    user_id, "pdf"
                                )
                                existing_content = next(
                                    (
                                        f
                                        for f in pdf_files
                                        if f.get("filename") == uploaded.filename
                                    ),
                                    None,
                                )
                                if existing_content:
                                    content_id = existing_content.get("content_id")
                                    # Update the response text to indicate question was answered
                                    item["response"][
                                        "text"
                                    ] = "PDF already exists, but question was answered using existing content."
                            else:
                                # New upload - get content_id from response
                                content_id = response.get("resource", {}).get("content_id")

                            # Answer the question if we have a content_id
                            if content_id:
                                try:
                                    answer = ai_assistant.assistant_response(
                                        query=question,
                                        user_id=user_id,
                                        token=auth_token,
                                        resource="content",
                                        graph_id=None,
                                        content_ids=[content_id],
                                    )
                                    item["answer"] = answer or {
                                        "text": "No answer generated"
                                    }
                                except Exception:
                                    traceback.print_exc()
                                    item["answer"] = {"text": "Error answering question"}

                        results.append(item)
                    
                    # Handle CSV files
                    elif filename_lower.endswith(".csv"):
                        response = save_data_file(uploaded, user_id, "csv")
                        item = {"filename": uploaded.filename, "response": response}
                        
                        # Handle question answering (if question provided and it's for code_exec)
                        if question and isinstance(response, dict) and question.lower().startswith(("compute", "analyze", "plot", "calculate", "generate", "create")):
                            content_id = response.get("resource", {}).get("content_id")
                            file_path = response.get("resource", {}).get("file_path")
                            
                            if content_id and file_path:
                                try:
                                    answer = ai_assistant.assistant_response(
                                        query=question,
                                        user_id=user_id,
                                        token=auth_token,
                                        resource="code_exec",
                                        graph_id=None,
                                        files=[file_path] if file_path else None,
                                    )
                                    item["answer"] = answer or {"text": "No answer generated"}
                                except Exception:
                                    traceback.print_exc()
                                    item["answer"] = {"text": "Error answering question"}
                        
                        results.append(item)
                    
                    # Handle HTML files
                    elif filename_lower.endswith((".html", ".htm")):
                        response = save_data_file(uploaded, user_id, "html")
                        item = {"filename": uploaded.filename, "response": response}
                        
                        # Handle question answering for code_exec
                        if question and isinstance(response, dict) and question.lower().startswith(("compute", "analyze", "plot", "calculate", "generate", "extract", "create")):
                            content_id = response.get("resource", {}).get("content_id")
                            file_path = response.get("resource", {}).get("file_path")
                            
                            if content_id and file_path:
                                try:
                                    answer = ai_assistant.assistant_response(
                                        query=question,
                                        user_id=user_id,
                                        token=auth_token,
                                        resource="code_exec",
                                        graph_id=None,
                                        files=[file_path] if file_path else None,
                                    )
                                    item["answer"] = answer or {"text": "No answer generated"}
                                except Exception:
                                    traceback.print_exc()
                                    item["answer"] = {"text": "Error answering question"}
                        
                        results.append(item)
                    
                    # Handle XML files
                    elif filename_lower.endswith(".xml"):
                        response = save_data_file(uploaded, user_id, "xml")
                        item = {"filename": uploaded.filename, "response": response}
                        
                        # Handle question answering for code_exec
                        if question and isinstance(response, dict) and question.lower().startswith(("compute", "analyze", "plot", "calculate", "generate", "extract", "create", "parse")):
                            content_id = response.get("resource", {}).get("content_id")
                            file_path = response.get("resource", {}).get("file_path")
                            
                            if content_id and file_path:
                                try:
                                    answer = ai_assistant.assistant_response(
                                        query=question,
                                        user_id=user_id,
                                        token=auth_token,
                                        resource="code_exec",
                                        graph_id=None,
                                        files=[file_path] if file_path else None,
                                    )
                                    item["answer"] = answer or {"text": "No answer generated"}
                                except Exception:
                                    traceback.print_exc()
                                    item["answer"] = {"text": "Error answering question"}
                        
                        results.append(item)
                    
                    # Reject unsupported file types
                    else:
                        results.append(
                            {
                                "filename": uploaded.filename,
                                "error": "Only PDF, CSV, HTML, and XML files are allowed.",
                            }
                        )

        # Handle web URLs
        urls = request.form.getlist("urls")
        for url in urls:
            if url and url.strip():
                clean_url = url.strip()
                response = ai_assistant.rag.save_web_content(clean_url, user_id)
                item = {"url": clean_url, "response": response}

                # Handle question answering for both new uploads and duplicates
                if question and isinstance(response, dict):
                    content_id = None
                    is_duplicate = response.get("text") == "URL already exists."

                    if is_duplicate:
                        # Get the existing content ID for this URL
                        web_files = mongo_db_manager.get_user_content_files(
                            user_id, "web"
                        )
                        existing_content = next(
                            (f for f in web_files if f.get("url") == clean_url), None
                        )
                        if existing_content:
                            content_id = existing_content.get("content_id")
                            # Update the response text to indicate question was answered
                            item["response"][
                                "text"
                            ] = "URL already exists, but question was answered using existing content."
                    else:
                        # New upload - get content_id from response
                        content_id = response.get("resource", {}).get("content_id")

                    # Answer the question if we have a content_id
                    if content_id:
                        try:
                            answer = ai_assistant.assistant_response(
                                query=question,
                                user_id=user_id,
                                token=auth_token,
                                resource="content",
                                graph_id=None,
                                content_ids=[content_id],
                            )
                            item["answer"] = answer or {"text": "No answer generated"}
                        except Exception:
                            traceback.print_exc()
                            item["answer"] = {"text": "Error answering question"}

                results.append(item)

        return jsonify(results=results), 200
    except Exception as e:
        current_app.logger.error(f"Content upload error: {e}")
        traceback.print_exc()
        return jsonify(error=f"Error uploading content: {str(e)}"), 500


@main_bp.route("/user_status/documents/", methods=["GET"])
@token_required
def user_status(current_user_id, auth_token):
    # Get user's content status and limits (PDFs + web content)
    try:
        data = request.form
        user_id = data.get("user_id") or current_user_id
        if not user_id:
            return jsonify(error="Missing user_id"), 400

        # Get all content files using unified method
        all_content_files = mongo_db_manager.get_user_content_files(user_id)

        # Separate PDF and web content
        pdf_files_data = []
        web_files_data = []

        for content in all_content_files:
            if content.get("content_type") == "pdf":
                pdf_files_data.append(
                    {
                        "filename": content.get("filename"),
                        "content_id": content.get("content_id"),
                        "content_type": "pdf",
                        "num_pages": content.get("num_pages"),
                        "file_size": content.get("file_size"),
                        "upload_time": (
                            content.get("upload_time").strftime("%Y-%m-%d %H:%M:%S")
                            if content.get("upload_time")
                            else None
                        ),
                        "summary": content.get("summary"),
                    }
                )
            elif content.get("content_type") == "web":
                web_files_data.append(
                    {
                        "url": content.get("url"),
                        "title": content.get("title"),
                        "author": content.get("author"),
                        "content_id": content.get("content_id"),
                        "content_type": "web",
                        "file_size": content.get("file_size"),
                        "upload_time": (
                            content.get("upload_time").strftime("%Y-%m-%d %H:%M:%S")
                            if content.get("upload_time")
                            else None
                        ),
                        "summary": content.get("summary"),
                    }
                )

        # Get counts using unified methods
        total_count = mongo_db_manager.get_content_count(user_id)
        pdf_count = mongo_db_manager.get_content_count(user_id, "pdf")
        web_count = mongo_db_manager.get_content_count(user_id, "web")

        # Combine all content
        all_files = pdf_files_data + web_files_data

        return (
            jsonify(
                user_id=user_id,
                total_count=total_count,
                pdf_count=pdf_count,
                web_count=web_count,
                files=all_files,
            ),
            200,
        )

    except Exception as e:
        current_app.logger.error(f"User status error: {e}")
        traceback.print_exc()
        return jsonify(error=f"Error getting user status: {str(e)}"), 500


@main_bp.route("/clear_user_data", methods=["DELETE"])
@token_required
def clear_user_data(current_user_id, auth_token):
    # Clear all content data and conversation history for a specific user
    try:
        data = request.form
        user_id = data.get("user_id") or current_user_id
        if not user_id:
            return jsonify(error="Missing user_id"), 400

        # Get all content files using unified method
        all_content_files = mongo_db_manager.get_user_content_files(user_id)

        for content in all_content_files:
            if content.get("content_type") == "pdf":
                # Remove PDF file from storage
                pdf_path = os.path.join(
                    "storage/pdfs", f"{content.get('content_id')}.pdf"
                )
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

            # Remove from database using unified method
            mongo_db_manager.delete_content_file(user_id, content.get("content_id"))

        # Clear conversation history
        HistoryManager().clear_user_history(user_id)

        # Clear Qdrant collection for this user
        try:
            qdrant_client = current_app.config["qdrant_client"]
            qdrant_client.client.delete_collection(collection_name=user_id)
            print(f"Qdrant collection '{user_id}' deleted")
        except Exception as qdrant_error:
            print(f"Qdrant collection deletion error (may not exist): {qdrant_error}")

        return (
            jsonify(message=f"User data and Qdrant collection cleared for {user_id}"),
            200,
        )

    except Exception as e:
        current_app.logger.error(f"Clear user data error: {e}")
        traceback.print_exc()
        return jsonify(error=f"Error clearing user data: {str(e)}"), 500


@main_bp.route("/delete_content", methods=["DELETE"])
@token_required
def delete_content(current_user_id, auth_token):
    # Unified endpoint for deleting content (PDF or web)
    try:
        data = request.form
        user_id = data.get("user_id")
        content_id = data.get("content_id")
        content_type = data.get("content_type", "pdf")
        if not user_id or not content_id:
            return jsonify(error="Missing user_id or content_id"), 400

        # Get content details from database
        content_file = mongo_db_manager.get_content_file_by_id(user_id, content_id)
        if not content_file:
            return jsonify(error="Content not found for this user"), 404

        # Handle PDF-specific deletion
        if content_type == "pdf" or content_file.get("content_type") == "pdf":
            # Remove PDF file from storage
            pdf_path = os.path.join("storage/pdfs", f"{content_id}.pdf")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            else:
                current_app.logger.warning(
                    f"PDF file {pdf_path} not found for deletion."
                )

        # Remove from database
        mongo_db_manager.delete_content_file(user_id, content_id)

        # Remove from Qdrant
        qdrant_client = current_app.config["qdrant_client"]
        qdrant_client.delete_content_by_id(user_id, content_id)

        return (
            jsonify(
                message=f"{content_file.get('content_type', 'unknown').upper()} {content_id} deleted for user {user_id}"
            ),
            200,
        )
    except Exception as e:
        current_app.logger.error(f"Delete content error: {e}")
        traceback.print_exc()
        return jsonify(error=f"Error deleting content: {str(e)}"), 500


@main_bp.route("/audio/summary", methods=["GET"])
@token_required
def get_summary_audio(current_user_id, auth_token):
    # Generate and serve summary audio on-demand, with Redis caching
    try:
        data = request.form
        user_id = data.get("user_id") if data else None
        content_id = data.get("content_id") if data else None

        if not user_id or not content_id:
            return jsonify(error="Missing user_id or content_id"), 400

        # Redis cache key
        cache_key = f"audio:summary:{user_id}:{content_id}"
        audio_data = redis_manager.get_audio_cache(cache_key)
        if audio_data:
            current_app.logger.info(
                f"[AUDIO CACHE] Served summary audio for user_id={user_id}, content_id={content_id} from Redis cache."
            )
            return Response(audio_data, mimetype="audio/mpeg")

        # Get content file using unified method
        content_file = mongo_db_manager.get_content_file_by_id(user_id, content_id)

        if not content_file:
            return jsonify(error="Content not found for this user"), 404

        # Get the summary from the stored user data
        summary_text = content_file.get("summary") or ""

        if not summary_text:
            return jsonify(error="No summary found for this content"), 404

        # Generate audio on-demand
        audio_data = tts_manager.generate_audio_on_demand(summary_text, voice="russell")

        if audio_data is None:
            return jsonify(error="Failed to generate audio"), 500

        # Store in Redis cache for 10 minutes
        redis_manager.set_audio_cache(cache_key, audio_data, expire_seconds=600)

        # Return the audio data directly
        return Response(audio_data, mimetype="audio/mpeg")

    except Exception as e:
        current_app.logger.error(f"Summary audio error: {e}")
        traceback.print_exc()
        return jsonify(error=f"Error generating summary audio: {str(e)}"), 500


@main_bp.route("/audio/query", methods=["GET"])
@token_required
def get_query_audio(current_user_id, auth_token):
    # Generate and serve query audio on-demand using query_id, with Redis caching
    try:
        data = request.form
        user_id = data.get("user_id") if data else None
        query_id = data.get("query_id") if data else None

        if not user_id or not query_id:
            return jsonify(error="Missing user_id or query_id"), 400

        # Redis cache key
        cache_key = f"audio:query:{user_id}:{query_id}"
        audio_data = redis_manager.get_audio_cache(cache_key)
        if audio_data:
            current_app.logger.info(
                f"[AUDIO CACHE] Served query audio for user_id={user_id}, query_id={query_id} from Redis cache."
            )
            return Response(audio_data, mimetype="audio/mpeg")

        # Get the specific conversation entry by query_id using HistoryManager
        entry = HistoryManager().get_entry_by_query_id(user_id, query_id)

        if not entry:
            return jsonify(error="Query not found in history"), 404

        # Get the assistant's answer text
        text_content = entry.get("assistant answer", "")

        if not text_content:
            return jsonify(error="No text found for this query"), 404

        # Generate audio on-demand
        audio_data = tts_manager.generate_audio_on_demand(text_content, voice="russell")

        if audio_data is None:
            return jsonify(error="Failed to generate audio"), 500

        # Store in Redis cache for 10 minutes
        redis_manager.set_audio_cache(cache_key, audio_data, expire_seconds=600)

        # Return the audio data directly
        return Response(audio_data, mimetype="audio/mpeg")

    except Exception as e:
        current_app.logger.error(f"Query audio error: {e}")
        traceback.print_exc()
        return jsonify(error=f"Error generating query audio: {str(e)}"), 500


@main_bp.route("/history", methods=["GET"])
@token_required
def get_user_history(current_user_id, auth_token):
    # Get conversation history for a user
    try:
        data = request.form
        user_id = data.get("user_id") or current_user_id

        history = HistoryManager().retrieve_user_history(user_id)

        return jsonify(history), 200
    except Exception as e:
        current_app.logger.error(f"Error retrieving history: {e}")
        return jsonify(error=f"Error retrieving history: {str(e)}"), 500


@main_bp.route("/history", methods=["DELETE"])
@token_required
def clear_user_history(current_user_id, auth_token):
    # Clear conversation history for a user
    try:
        data = request.form
        user_id = data.get("user_id") or current_user_id

        HistoryManager().clear_user_history(user_id)

        return jsonify(message="History cleared successfully"), 200
    except Exception as e:
        current_app.logger.error(f"Error clearing history: {e}")
        return jsonify(error=f"Error clearing history: {str(e)}"), 500


@main_bp.route("/", methods=["GET"])
def health_check():
    return jsonify("This is health check")
