from app.lib.auth import token_required
from flask import Blueprint, request, current_app, jsonify, Response
from dotenv import load_dotenv
import traceback
import json
import os
from app.rag.utils.tts_utils import tts_manager
from app.storage.redis import redis_manager
from app.storage.mongo_storage import mongo_db_manager

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
        user_id = current_user_id
        
        # uploaded files 
        uploaded_files = request.files.getlist("uploaded_files") if "uploaded_files" in request.files else None
        
        # uploaded datas
        data = request.form
        question = data.get("question") or data.get("query")
        context_raw = data.get("context", "{}")
        try:
            context = json.loads(context_raw)
        except Exception:
            context = {}
        context_id = context.get("content_id", None)
        graph_id = context.get("id",None)
        resource = context.get("resource", "annotation")
        url = context.get("url",None)
        graph = data.get("graph", None)
        if isinstance(graph, str) and graph.strip():
            try:
                graph = json.loads(graph)
            except Exception:
                # Keep original value so downstream validation can report a clear issue.
                pass
        json_query = data.get("json_query", None)


        if url:
            if isinstance(url, str):
                url = [url]
            elif isinstance(url, list):
                url = url
            else:
                url = list(url)

        def _flatten_to_strings(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple, set)):
                out = []
                for item in value:
                    out.extend(_flatten_to_strings(item))
                return out
            text = str(value).strip()
            return [text] if text else []

        # Determine content_ids if resource is content and id is a list or string
        content_ids = None
        if context_id is not None:
            if isinstance(context_id, list):
                content_ids = _flatten_to_strings(context_id)
            elif isinstance(context_id, str):
                # If it's a comma-separated string, split it
                if context_id.strip().startswith("["):
                    try:
                        content_ids = _flatten_to_strings(json.loads(context_id))
                    except Exception:
                        content_ids = _flatten_to_strings([context_id])
                else:
                    content_ids = _flatten_to_strings(
                        [cid.strip() for cid in context_id.split(",") if cid.strip()]
                    )

        # Handle file uploads
        upload_results = []
        newly_uploaded_content_ids = []

        if uploaded_files:
            for uploaded in uploaded_files:
                if uploaded.filename and uploaded.filename.lower().endswith(".pdf"):
                    response = ai_assistant.agents.rag.save_retrievable_docs(uploaded, user_id)
                    if isinstance(response, dict):
                        is_duplicate = response.get("text") == "PDF already exists."
                        if is_duplicate:
                            pdf_files = mongo_db_manager.get_user_content_files(user_id, "pdf")
                            existing = next((f for f in pdf_files if f.get("filename") == uploaded.filename), None)
                            if existing:
                                newly_uploaded_content_ids.append(existing.get("content_id"))
                        else:
                            new_id = response.get("resource", {}).get("content_id")
                            if new_id:
                                newly_uploaded_content_ids.extend(_flatten_to_strings(new_id))
                        upload_results.append({"filename": uploaded.filename, "response": response})
            
            # Merge content_ids
            if newly_uploaded_content_ids:
                content_ids = (content_ids or []) + newly_uploaded_content_ids

        # Case 1: files uploaded, but no question -> upload-only flow
        if uploaded_files and not question and not json_query:
            suggested_questions = []

            for r in upload_results:
                sq = r.get("response", {}).get("resource", {}).get("suggested_questions")
                if sq:
                    if isinstance(sq, list):
                        suggested_questions.extend(sq)
                    else:
                        suggested_questions.append(sq)

            return jsonify({
                "text": "Files uploaded successfully.",
                "content_ids": content_ids,
                "suggested_questions": suggested_questions,
            }), 200

        # Case 2: no files and no query -> invalid request
        if not uploaded_files and not question and not json_query:
            return jsonify({
                "error": "No input provided. Please upload files or submit a question."
            }), 400

        # Pass all relevant arguments to ai_assistant
        response = ai_assistant.assistant_response(
            query=question,
            user_id=user_id,
            token=auth_token,
            graph_id=graph_id,
            graph=graph,
            resource=resource,
            content_ids=content_ids,
            urls=url
        )

        return jsonify(response)
    except Exception as e:
        current_app.logger.error(f"Exception: {e}")
        traceback.print_exc()
        return f"Bad Response: {e}", 400


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
        mongo_db_manager.clear_user_history(user_id)

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
        user_id = current_user_id
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
            pdf_path = os.path.join("pdfs_uploaded/pdfs", f"{content_id}.pdf")
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
        user_id = current_user_id
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
        user_id = current_user_id
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

        # Get the specific conversation entry by query_id
        entry = mongo_db_manager.get_entry_by_query_id(user_id, query_id)

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

        history = mongo_db_manager.retrieve_user_history(user_id)

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

        mongo_db_manager.clear_user_history(user_id)

        return jsonify(message="History cleared successfully"), 200
    except Exception as e:
        current_app.logger.error(f"Error clearing history: {e}")
        return jsonify(error=f"Error clearing history: {str(e)}"), 500


def handle_hypothesis_faq(auth_token):
    import os
    import requests
    projects_api_url = os.getenv("HYPOTHESIS_DATA_API")
    headers = {"Authorization": auth_token}

    r = requests.get(projects_api_url, headers=headers, timeout=15)
    if r.status_code != 200:
        print("Projects API failed:", r.text)
        return None

    projects = r.json().get("projects", [])
    if not projects:
        print("No projects found")
        return None

    project_map = []
    all_genes = set()
    all_tissues = set()
    all_phenotypes = set()

    for p in projects:
        pid = p.get("id")
        name = p.get("name")
        phenotype = p.get("phenotype")

        if not pid:
            continue

        d = requests.get(
            projects_api_url,
            headers=headers,
            params={"id": pid},
            timeout=15
        )

        if d.status_code != 200:
            print(f"Project {pid} detail failed")
            continue

        data = d.json()

        genes = set()
        tissues = set()

        for h in data.get("hypotheses", []):
            if h.get("causal_gene"):
                genes.add(h["causal_gene"])
                all_genes.add(h["causal_gene"])

        for t in data.get("ldsc", {}).get("tissues", []):
            if t.get("name"):
                tissues.add(t["name"])
                all_tissues.add(t["name"])

        if phenotype:
            all_phenotypes.add(phenotype)

        project_map.append({
            "project_id": pid,
            "project_name": name,
            "phenotype": phenotype,
            "causal_genes": list(genes),
            "tissues": list(tissues)
        })

    if not project_map:
        print("No usable hypothesis data")
        return None

    # LLM once with all info
    llm_prompt = f"""
                Here are hypothesis results grouped by project:

                {json.dumps(project_map, indent=2)}

                Generate 3 example research questions based on:
                - project phenotypes
                - causal genes
                - tissues

                Return JSON list of strings only.
                """
    ai_assistant = current_app.config["ai_assistant"]
    llm_response = ai_assistant.advanced_llm.generate(llm_prompt)
    return jsonify({
        "text": "Here’s your hypothesis-based AI-generated questions:",
        "projects": project_map,
        "sample_questions": llm_response
    }), 200

   

@main_bp.route("/faq", methods=["GET"])
@token_required
def get_faq_intro(current_user_id,auth_token):
    """
    Get welcome message and list of FAQ questions.
    No authentication required - public endpoint for discovery.
    """
    try:
        context = request.args.get("context",None)
        if context == "hypothesis":
            return handle_hypothesis_faq(auth_token)

        questions = mongo_db_manager.get_all_faq_questions(context)
        question_list = [
            {"id": q["question_id"], "text": q["question_text"], "link": f"/faq/{q['question_id']}"}
            for q in questions
        ]

        return jsonify({
            "text": "Hello! I’m MOZI, your AI assistant for exploring and annotating "
                "biomedical entities in the BioAtomspace. "
                f"To help you get started, here are some example questions you can try on {context} "
                "Just click one to begin:",
            "questions": question_list
        }), 200
    except Exception as e:
            current_app.logger.error(f"Error in FAQ intro: {e}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@main_bp.route("/faq/<question_id>", methods=["GET"])
@token_required
def get_faq_answer(current_user_id,auth_token,question_id):
    """
    Get answer for a FAQ question from MongoDB.
    No authentication required for demo purposes.
    Returns pre-populated answer instantly.
    """
    try:
        faq = mongo_db_manager.get_faq_by_id(question_id)
        
        if not faq:
            return jsonify({
                "error": f"Question ID '{question_id}' not found in FAQ",
                "text": "Use POST /query for custom questions"
            }), 404
        
        return jsonify({
            "question": faq["question_text"],
            "text": faq["text"],
            "json_format": faq["json_format"]
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in FAQ answer: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@main_bp.route("/", methods=["GET"])
def health_check():
    return jsonify("This is health check")