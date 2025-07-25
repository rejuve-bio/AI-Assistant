from app.lib.auth import token_required
from flask import Blueprint, request, current_app, jsonify, Response
from dotenv import load_dotenv
import traceback
import json
import os
from app.rag.utils.tts_utils import tts_manager
from app.storage.sql_redis_storage import set_audio_cache, get_audio_cache, db_manager
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
            - id: For PDF queries, a list of PDF IDs; for other resources, a single ID.
            - resource: The type of resource (e.g., 'pdf', 'annotation', 'hypothesis').
        - graph, json_query: Optional, for advanced queries.
    - For PDF queries (resource == 'pdf'), pdf_ids are extracted from context['id'].
    - If pdf_ids are provided, answers are retrieved only from those PDFs; otherwise, answers are retrieved from all collections (user and general).
    - Handles both user-uploaded PDF question answering and general knowledge queries.
    """
    try:
        ai_assistant = current_app.config["ai_assistant"]
        data = request.form

        user_id = data.get("user_id") or current_user_id
        question = data.get("question") or data.get("query")
        context_raw = data.get("context", "{}")
        try:
            context = json.loads(context_raw)
        except Exception:
            context = {}
        context_id = context.get("id", None)
        resource = context.get("resource", "annotation")
        graph = data.get("graph", None)
        json_query = data.get("json_query", None)

        # Determine pdf_ids if resource is pdf and id is a list or string
        pdf_ids = None
        if resource == "pdf":
            if isinstance(context_id, list):
                pdf_ids = context_id
            elif isinstance(context_id, str):
                # If it's a comma-separated string, split it
                if context_id.strip().startswith("["):
                    try:
                        pdf_ids = json.loads(context_id)
                    except Exception:
                        pdf_ids = [context_id]
                else:
                    pdf_ids = [
                        pid.strip() for pid in context_id.split(",") if pid.strip()
                    ]

        # Ensure query exists before processing
        if not question and not json_query:
            return jsonify({"error": "No query provided."}), 400

        # Pass all relevant arguments to ai_assistant
        response = ai_assistant.assistant_response(
            query=question,
            user_id=user_id,
            token=auth_token,
            graph_id=context_id if resource != "pdf" else None,
            graph=graph,
            resource=resource,
            json_query=json_query,
            pdf_ids=pdf_ids,
        )

        return jsonify(response)
    except Exception as e:
        current_app.logger.error(f"Exception: {e}")
        traceback.print_exc()
        return f"Bad Response: {e}", 400


@main_bp.route("/rag/upload_pdf", methods=["POST"])
@token_required
def upload_pdf(current_user_id, auth_token):
    # Upload and process PDF documents using the RAG module
    try:
        user_id = request.form.get("user_id")
        if not user_id:
            return jsonify(error="Missing user_id"), 400

        if "files" not in request.files:
            return jsonify(error="No files uploaded"), 400

        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            return jsonify(error="No files selected"), 400

        ai_assistant = current_app.config["ai_assistant"]
        results = []
        for uploaded in files:
            # Only allow PDF files
            if not uploaded.filename.lower().endswith(".pdf"):
                results.append(
                    {
                        "filename": uploaded.filename,
                        "error": "Only PDF files are allowed.",
                    }
                )
                continue
            # Delegate all processing to the RAG module
            response = ai_assistant.rag.save_retrievable_docs(uploaded, user_id)
            results.append({"filename": uploaded.filename, "response": response})
        return jsonify(results=results), 200
    except Exception as e:
        current_app.logger.error(f"PDF upload error: {e}")
        traceback.print_exc()
        return jsonify(error=f"Error uploading PDF: {str(e)}"), 500


@main_bp.route("/rag/user_status", methods=["GET"])
@token_required
def user_status(current_user_id, auth_token):
    # Get user's PDF status and limits
    try:
        data = request.form
        user_id = data.get("user_id") or current_user_id
        if not user_id:
            return jsonify(error="Missing user_id"), 400

        pdf_files = db_manager.get_user_pdfs(user_id)
        count = db_manager.get_pdf_count(user_id)
        files = [
            {
                "filename": pdf.filename,
                "pdf_id": pdf.pdf_id,
                "num_pages": pdf.num_pages,
                "file_size": pdf.file_size,
                "upload_time": (
                    pdf.upload_time.strftime("%Y-%m-%d %H:%M:%S")
                    if pdf.upload_time
                    else None
                ),
                "summary": pdf.summary,
            }
            for pdf in pdf_files
        ]
        PDF_LIMIT = 5
        return (
            jsonify(
                user_id=user_id,
                count=count,
                limit=PDF_LIMIT,
                files=files,
            ),
            200,
        )

    except Exception as e:
        current_app.logger.error(f"User status error: {e}")
        traceback.print_exc()
        return jsonify(error=f"Error getting user status: {str(e)}"), 500


@main_bp.route("/rag/clear_user_data", methods=["DELETE"])
@token_required
def clear_user_data(current_user_id, auth_token):
    # Clear all PDF data and conversation history for a specific user
    try:
        data = request.form
        user_id = data.get("user_id") or current_user_id
        if not user_id:
            return jsonify(error="Missing user_id"), 400

        # 1. Delete all PDF files for this user from DB and disk
        pdf_files = db_manager.get_user_pdfs(user_id)
        for pdf in pdf_files:
            # Remove PDF file from storage
            pdf_path = os.path.join("storage/pdfs", f"{pdf.pdf_id}.pdf")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            db_manager.delete_pdf_file(user_id, pdf.pdf_id)

        # 2. Clear conversation history (and related memory/context)
        HistoryManager().clear_user_history(user_id)

        # 3. Clear Qdrant collection for this user
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


@main_bp.route("/rag/delete_pdf", methods=["DELETE"])
@token_required
def delete_pdf(current_user_id, auth_token):
    try:
        data = request.form
        user_id = data.get("user_id")
        pdf_id = data.get("pdf_id")
        if not user_id or not pdf_id:
            return jsonify(error="Missing user_id or pdf_id"), 400

        # Remove PDF file from storage
        pdf_path = os.path.join("storage/pdfs", f"{pdf_id}.pdf")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        else:
            current_app.logger.warning(f"PDF file {pdf_path} not found for deletion.")

        # Remove from database
        db_manager.delete_pdf_file(user_id, pdf_id)

        # Remove from Qdrant
        qdrant_client = current_app.config["qdrant_client"]
        qdrant_client.delete_pdf_by_id(user_id, pdf_id)

        return jsonify(message=f"PDF {pdf_id} deleted for user {user_id}"), 200
    except Exception as e:
        current_app.logger.error(f"Delete PDF error: {e}")
        traceback.print_exc()
        return jsonify(error=f"Error deleting PDF: {str(e)}"), 500


@main_bp.route("/rag/audio/summary", methods=["GET"])
@token_required
def get_summary_audio(current_user_id, auth_token):
    # Generate and serve summary audio on-demand, with Redis caching
    try:
        data = request.form
        user_id = data.get("user_id") if data else None
        pdf_id = data.get("pdf_id") if data else None

        if not user_id or not pdf_id:
            return jsonify(error="Missing user_id or pdf_id"), 400

        # Redis cache key
        cache_key = f"audio:summary:{user_id}:{pdf_id}"
        audio_data = get_audio_cache(cache_key)
        if audio_data:
            current_app.logger.info(
                f"[AUDIO CACHE] Served summary audio for user_id={user_id}, pdf_id={pdf_id} from Redis cache."
            )
            return Response(audio_data, mimetype="audio/mpeg")

        # Get user's PDF data to find the summary
        pdf_files = db_manager.get_user_pdfs(user_id)
        target_file = None
        for pdf in pdf_files:
            if pdf.pdf_id == pdf_id:
                target_file = pdf
                break

        if not target_file:
            return jsonify(error="PDF not found for this user"), 404

        # Get the summary from the stored user data
        summary_text = target_file.summary or ""

        if not summary_text:
            return jsonify(error="No summary found for this PDF"), 404

        # Generate audio on-demand
        audio_data = tts_manager.generate_audio_on_demand(summary_text, voice="russell")

        if audio_data is None:
            return jsonify(error="Failed to generate audio"), 500

        # Store in Redis cache for 10 minutes
        set_audio_cache(cache_key, audio_data, expire_seconds=600)

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
        audio_data = get_audio_cache(cache_key)
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
        set_audio_cache(cache_key, audio_data, expire_seconds=600)

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
