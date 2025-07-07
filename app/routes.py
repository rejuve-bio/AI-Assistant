from app.lib.auth import token_required
from flask import Blueprint, request, current_app, jsonify
from dotenv import load_dotenv
import traceback
import json
import uuid
import os
from app.rag.utils.pdf_processor import extract_text_from_pdf, chunk_text
from app.storage.qdrant import Qdrant
from app.rag.utils.llm_wrapper import LLMWrapper
from app.prompts.rag_prompts import PDF_PROCESSOR_PROMPT

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
        json.dump(user_pdf_data, f)


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


"""
Initialize Qdrant and LLM for PDF processing

for embedding:
- use_openai_embeddings=True: Uses OpenAI for vector embeddings
- use_openai_embeddings=False: Uses SentenceTransformers for vector embeddings

for llm usage:
- use_openai=True: Uses OpenAI for text generation
- use_openai=False: Uses Google gemini for text generation
"""
qdrant_client = Qdrant(use_openai_embeddings=True)
llm_wrapper = LLMWrapper(use_openai=True)


@main_bp.route("/query", methods=["POST"])
@token_required
def process_query(current_user_id, auth_token):
    """
    Notes:
    - `query`: Contains the user's question or prompt.
    - `file`: Used when a file (e.g., a PDF) is uploaded for processing.
    - `id`: Represents a graph ID and should be included if relevant to the query.(when Explaining a node is asked from a given graph)
    - `resource`: Identifies the type of resource associated with the `id`. Currently not in use but it may support other types (e.g., "Hypothesis") in the future.
    """

    try:
        ai_assistant = current_app.config["ai_assistant"]

        if not request.form and "file" not in request.files:
            return jsonify({"error": "Null request is invalid format."}), 400
        if request.form.get("query") and request.form.get("json_query"):
            return jsonify({"error": "Invalid format."}), 400

        data = request.form
        query = data.get("query", None)
        context = json.loads(data.get("context", "{}"))
        context_id = context.get("id", None)
        resource = context.get("resource", "annotation")
        graph = data.get("graph", None)
        json_query = data.get("json_query", None)

        # Handle file upload
        file = None
        if "file" in request.files:
            file = request.files["file"]

        # Ensure query exists before processing
        if query:
            response = ai_assistant.assistant_response(
                query=query,
                file=file,
                user_id=current_user_id,
                token=auth_token,
                graph_id=context_id,
                graph=graph,
                resource=resource,
            )
        else:
            # Handle case when only context is provided
            print("no query provided")
            response = ai_assistant.assistant_response(
                query=None,
                file=file,
                user_id=current_user_id,
                token=auth_token,
                graph_id=context_id,
                graph=graph,
                resource=resource,
                json_query=json_query,
            )

        return jsonify(response)  # Always return a valid JSON response

    except Exception as e:
        current_app.logger.error(f"Exception: {e}")
        traceback.print_exc()
        return f"Bad Response: {e}", 400


@main_bp.route("/rag/upload_pdf", methods=["POST"])
@token_required
def upload_pdf(current_user_id, auth_token):
    # Upload and process PDF documents
    try:
        user_id = request.form.get("user_id") or request.json.get("user_id")
        if not user_id:
            return jsonify(error="Missing user_id"), 400

        if "files" not in request.files:
            return jsonify(error="No files uploaded"), 400

        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            return jsonify(error="No files selected"), 400

        # Check all files for duplicates, limits, and file type before processing
        for uploaded in files:
            # Check file extension
            if not uploaded.filename.lower().endswith(".pdf"):
                return (
                    jsonify(
                        error="Only PDF files are allowed.",
                        resource={
                            "filename": uploaded.filename,
                            "allowed_types": ["pdf"],
                        },
                    ),
                    400,
                )

            # Check for duplicate files by filename
            if any(
                f["filename"] == uploaded.filename
                for f in get_user_data(user_id)["files"]
            ):
                return (
                    jsonify(
                        error="PDF already exists.",
                        resource={"filename": uploaded.filename},
                    ),
                    400,
                )

            if get_user_data(user_id)["count"] >= PDF_LIMIT:
                return (
                    jsonify(
                        error="Your quota is full. Maximum 5 PDFs allowed.",
                        resource={"count": get_user_data(user_id)["count"]},
                    ),
                    400,
                )

        pdf_ids = []
        for uploaded in files:
            pdf_id = str(uuid.uuid4())
            pdf_ids.append(pdf_id)

            upload_folder = "storage/pdfs"
            os.makedirs(upload_folder, exist_ok=True)

            # save file
            pdf_path = os.path.join(upload_folder, f"{pdf_id}.pdf")
            uploaded.save(pdf_path)

            # extract & chunk
            full_text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(full_text)

            # metadata to store alongside each chunk
            metadata = {
                "pdf_id": pdf_id,
                "filename": uploaded.filename,
                "user_id": user_id,
            }

            # store in Qdrant under this user
            qdrant_client.add_pdf_document(
                collection_name=user_id, chunks=chunks, metadata=metadata
            )

            # update user tracking for each file
            update_user_data(
                user_id,
                {
                    "count": get_user_data(user_id)["count"] + 1,
                    "files": get_user_data(user_id)["files"]
                    + [{"filename": uploaded.filename, "pdf_id": pdf_id}],
                },
            )

        # Get final count for response
        final_user_data = get_user_data(user_id)
        final_count = final_user_data["count"]

        return (
            jsonify(
                user_id=user_id,
                pdf_ids=pdf_ids,
                message=f"PDF uploaded successfully. {final_count}/{PDF_LIMIT} PDFs used.",
            ),
            200,
        )

    except Exception as e:
        current_app.logger.error(f"PDF upload error: {e}")
        traceback.print_exc()
        return jsonify(error=f"Error uploading PDF: {str(e)}"), 500


@main_bp.route("/rag/ask_question", methods=["POST"])
@token_required
def ask_question(current_user_id, auth_token):
    # Ask questions about uploaded PDF documents
    try:
        payload = request.get_json()
        if not payload:
            return jsonify(error="Invalid JSON payload"), 400

        user_id = payload.get("user_id")
        question = payload.get("question")
        pdf_ids = payload.get("pdf_ids")

        if not user_id or not question:
            return jsonify(error="Missing user_id or question"), 400

        context_chunks = qdrant_client.query_pdf_documents(
            collection_name=user_id, query=question, top_k=10, pdf_ids=pdf_ids
        )
        context = "\n\n".join(context_chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        answer = llm_wrapper.chat(PDF_PROCESSOR_PROMPT, user_prompt)

        return jsonify(answer=answer), 200

    except Exception as e:
        current_app.logger.error(f"Question asking error: {e}")
        traceback.print_exc()
        return jsonify(error=f"Error processing question: {str(e)}"), 500


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
