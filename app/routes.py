from app.lib.auth import token_required
from flask import Blueprint, request, current_app, jsonify, Response
from dotenv import load_dotenv
import traceback
import json
import os
from app.rag.utils.tts_utils import tts_manager
from app.storage.sql_redis_storage import set_audio_cache, get_audio_cache

load_dotenv()
main_bp = Blueprint("main", __name__)

# Initialize PDF processing components
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
    # Update only a specific user's data without affecting other users
    try:
        current_data = load_user_pdf()
        current_data[user_id] = new_user_data
        save_user_pdf(current_data)
        return True
    except Exception as e:
        current_app.logger.error(f"Error updating user data: {e}")
        return False


def get_user_data(user_id):
    # Get data for a specific user
    current_data = load_user_pdf()
    return current_data.get(user_id, {"count": 0, "files": []})


@main_bp.route("/query", methods=["POST"])
@token_required
def process_query(current_user_id, auth_token):
    """
    Notes:
    - `query`: Contains the user's question or prompt.
    - `file`: Used when a file (e.g., a PDF) is uploaded for processing.
    - `pdf_ids`: List of PDF IDs to target specific user-uploaded PDFs for question answering using the integrated PDF explainer and RAG system.
    - `id`: Represents a graph ID and should be included if relevant to the query (e.g., when explaining a node from a given graph).
    - `resource`: Identifies the type of resource associated with the `id`. Currently not in use but may support other types (e.g., "Hypothesis") in the future.

    The endpoint now supports unified question answering from both user-uploaded PDFs and general knowledge, leveraging the integrated PDF explainer logic.
    """
    try:
        ai_assistant = current_app.config["ai_assistant"]

        # Accept both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        user_id = data.get("user_id") or current_user_id
        question = data.get("question") or data.get("query")
        pdf_ids = data.get("pdf_ids")
        if isinstance(pdf_ids, str):
            # If pdf_ids is a comma-separated string, convert to list
            pdf_ids = [pid.strip() for pid in pdf_ids.split(",") if pid.strip()]

        context = json.loads(data.get("context", "{}"))
        context_id = context.get("id", None)
        resource = context.get("resource", "annotation")
        graph = data.get("graph", None)
        json_query = data.get("json_query", None)

        # Ensure query exists before processing
        if not question and not json_query:
            return jsonify({"error": "No query provided."}), 400

        # Pass all relevant arguments to ai_assistant
        response = ai_assistant.assistant_response(
            query=question,
            user_id=user_id,
            token=auth_token,
            graph_id=context_id,
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
        user_id = request.form.get("user_id") or request.json.get("user_id")
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
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify(error="Missing user_id"), 400

        user_data = get_user_data(user_id)

        return (
            jsonify(
                user_id=user_id,
                count=user_data["count"],
                limit=PDF_LIMIT,
                files=user_data["files"],
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
    # Clear all PDF data for a specific user
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify(error="Missing user_id"), 400

        # Load current data and remove user
        current_data = load_user_pdf()
        if user_id in current_data:
            del current_data[user_id]
            save_user_pdf(current_data)

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


@main_bp.route("/rag/delete_pdf", methods=["DELETE"])
@token_required
def delete_pdf(current_user_id, auth_token):
    try:
        data = request.get_json() or {}
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

        # Remove from user tracking
        user_data = get_user_data(user_id)
        new_files = [f for f in user_data["files"] if f["pdf_id"] != pdf_id]
        new_count = max(0, user_data["count"] - 1)
        update_user_data(user_id, {"count": new_count, "files": new_files})

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
        data = request.get_json()
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
        user_data = get_user_data(user_id)
        files = user_data.get("files", [])

        # Find the PDF file with matching pdf_id
        target_file = None
        for file_info in files:
            if file_info.get("pdf_id") == pdf_id:
                target_file = file_info
                break

        if not target_file:
            return jsonify(error="PDF not found for this user"), 404

        # Get the summary from the stored user data
        summary_text = target_file.get("summary", "")

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
        data = request.get_json()
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

        # Get the AI assistant to access history
        ai_assistant = current_app.config["ai_assistant"]

        # Get the specific conversation entry by query_id
        entry = ai_assistant.history.get_entry_by_query_id(user_id, query_id)

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
        user_id = request.args.get("user_id") or current_user_id

        ai_assistant = current_app.config["ai_assistant"]
        history = ai_assistant.history.retrieve_user_history(user_id)

        return jsonify(history), 200
    except Exception as e:
        current_app.logger.error(f"Error retrieving history: {e}")
        return jsonify(error=f"Error retrieving history: {str(e)}"), 500


@main_bp.route("/history", methods=["DELETE"])
@token_required
def clear_user_history(current_user_id, auth_token):
    # Clear conversation history for a user
    try:
        user_id = request.args.get("user_id") or current_user_id

        ai_assistant = current_app.config["ai_assistant"]
        # Clear history by setting empty list
        ai_assistant.history.history[str(user_id)] = []
        ai_assistant.history._save_history()

        return jsonify(message="History cleared successfully"), 200
    except Exception as e:
        current_app.logger.error(f"Error clearing history: {e}")
        return jsonify(error=f"Error clearing history: {str(e)}"), 500


