from typing import List, Optional

from app.lib.auth import token_required, AuthContext
from fastapi import APIRouter, Depends, Form, File, UploadFile, Query, Request
from fastapi.responses import JSONResponse, Response
from dotenv import load_dotenv
import logging
import traceback
import json
import os
from app.rag.utils.tts_utils import tts_manager
from app.storage.redis import redis_manager
from app.storage.mongo_storage import mongo_db_manager

load_dotenv()
main_router = APIRouter()
logger = logging.getLogger(__name__)

AUDIO_MIME = "audio/mpeg"


def get_ai_assistant(request: Request):
    """FastAPI dependency: resolves the AiAssistance instance from app.state."""
    return request.app.state.ai_assistant


def get_qdrant_client(request: Request):
    """FastAPI dependency: resolves the shared Qdrant client from app.state."""
    return request.app.state.qdrant_client


def _handle_uploads(uploaded_files, ai_assistant, user_id, content_ids):
    upload_results = []
    newly_uploaded_content_ids = []

    for uploaded in uploaded_files:
        if uploaded.filename and uploaded.filename.lower().endswith(".pdf"):
            response = ai_assistant.rag.save_retrievable_docs(uploaded, user_id)
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
                        newly_uploaded_content_ids.append(new_id)
                upload_results.append({"filename": uploaded.filename, "response": response})

    if newly_uploaded_content_ids:
        content_ids = content_ids + newly_uploaded_content_ids if content_ids else newly_uploaded_content_ids

    return upload_results, content_ids


def _parse_context(data):
    context_raw = data.get("context", "{}")
    try:
        context = json.loads(context_raw)
    except Exception:
        context = {}

    context_id = context.get("content_id", None)
    graph_id = context.get("id", None)
    resource = context.get("resource", "annotation")
    url = context.get("url", None)
    json_query = data.get("json_query", None)
    question = data.get("question") or data.get("query")

    if url:
        if isinstance(url, str):
            url = [url]
        elif not isinstance(url, list):
            url = list(url)

    content_ids = None
    if context_id is not None:
        if isinstance(context_id, list):
            content_ids = context_id
        elif isinstance(context_id, str):
            if context_id.strip().startswith("["):
                try:
                    content_ids = json.loads(context_id)
                except Exception:
                    content_ids = [context_id]
            else:
                content_ids = [cid.strip() for cid in context_id.split(",") if cid.strip()]

    return question, graph_id, resource, url, json_query, content_ids


def _dispatch_query(ai_assistant, question, json_query, uploaded_files, upload_results, content_ids, user_id, auth_token, graph_id, resource, url, resume=None, thread_id=None):
    if uploaded_files and not question and not json_query:
        suggested_questions = []
        for r in upload_results:
            sq = r.get("response", {}).get("resource", {}).get("suggested_questions")
            if sq:
                if isinstance(sq, list):
                    suggested_questions.extend(sq)
                else:
                    suggested_questions.append(sq)
        return JSONResponse(status_code=200, content={
            "text": "Files uploaded successfully.",
            "content_ids": content_ids,
            "suggested_questions": suggested_questions,
        })

    if not thread_id:
        return JSONResponse(status_code=400, content={
            "error": "Missing thread_id. Every request must name the conversation it belongs to — "
                     "generate one per conversation on the client (e.g. a UUID at 'New Chat') and "
                     "send the same value with every message in that conversation.",
            "field": "thread_id",
        })

    if not uploaded_files and not question and not json_query and not resume:
        return JSONResponse(status_code=400, content={
            "error": "No input provided. Please upload files, submit a question, or resume a pending confirmation."
        })

    response = ai_assistant.assistant_response(
        query=question,
        user_id=user_id,
        token=auth_token,
        graph_id=graph_id,
        resource=resource,
        content_ids=content_ids,
        urls=url,
        resume=resume,
        thread_id=thread_id,
    )
    return JSONResponse(content=response)


@main_router.post("/query")
def process_query(
    auth: AuthContext = Depends(token_required),
    ai_assistant=Depends(get_ai_assistant),
    question: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
    context: str = Form("{}"),
    json_query: Optional[str] = Form(None),
    resume: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
    uploaded_files: Optional[List[UploadFile]] = File(None),
):
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
        - resume: Optional. The exact `value` of a button clicked in response to a
          `needs_confirmation` reply's `confirmation.options` (e.g. "confirm",
          "reject", or a specific candidate name) — a deterministic, validated
          answer to a pending confirmation. Mutually distinct from `question`:
          free text always goes through `question` and is interpreted as before;
          `resume` is only ever one of the exact offered option values.
        - thread_id: REQUIRED. Which conversation this message belongs to — the
          client picks it (e.g. a new UUID generated at "New Chat") and sends the
          same value with every message in that conversation. Requests without it
          are rejected with 400; there is deliberately no implicit fallback, so a
          client can't silently end up with every conversation sharing one thread.
          Scopes both the checkpointer (pause/resume state) and the durable
          conversation thread (messages + tool-call references).
    - For content queries (resource == 'content'), content_ids are extracted from context['id'].
    - If content_ids are provided, answers are retrieved only from those content items; otherwise, answers are retrieved from all collections (user and general).
    - Handles both user-uploaded content question answering and general knowledge queries.
    """
    try:
        user_id = auth.user_id

        question_parsed, graph_id, resource, url, json_query_parsed, content_ids = _parse_context(
            {"context": context, "json_query": json_query, "question": question, "query": query}
        )

        upload_results = []
        if uploaded_files:
            upload_results, content_ids = _handle_uploads(uploaded_files, ai_assistant, user_id, content_ids)

        return _dispatch_query(ai_assistant, question_parsed, json_query_parsed, uploaded_files, upload_results, content_ids, user_id, auth.token, graph_id, resource, url, resume, thread_id)
    except Exception as e:
        logger.error(f"Exception: {e}")
        traceback.print_exc()
        return Response(content=f"Bad Response: {e}", status_code=400, media_type="text/html")


@main_router.get("/user_status/documents/")
def user_status(
    auth: AuthContext = Depends(token_required),
    user_id: Optional[str] = Form(None),
):
    # Get user's content status and limits (PDFs + web content)
    try:
        user_id = user_id or auth.user_id
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "Missing user_id"})

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

        return JSONResponse(status_code=200, content={
            "user_id": user_id,
            "total_count": total_count,
            "pdf_count": pdf_count,
            "web_count": web_count,
            "files": all_files,
        })

    except Exception as e:
        logger.error(f"User status error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Error getting user status: {str(e)}"})


@main_router.delete("/clear_user_data")
def clear_user_data(
    auth: AuthContext = Depends(token_required),
    qdrant_client=Depends(get_qdrant_client),
    user_id: Optional[str] = Form(None),
):
    # Clear all content data and conversation history for a specific user
    try:
        user_id = user_id or auth.user_id
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "Missing user_id"})

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
            qdrant_client.client.delete_collection(collection_name=user_id)
            logger.info(f"Qdrant collection '{user_id}' deleted")
        except Exception as qdrant_error:
            logger.warning(f"Qdrant collection deletion error (may not exist): {qdrant_error}")

        return JSONResponse(status_code=200, content={
            "message": f"User data and Qdrant collection cleared for {user_id}"
        })

    except Exception as e:
        logger.error(f"Clear user data error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Error clearing user data: {str(e)}"})


@main_router.delete("/delete_content")
def delete_content(
    auth: AuthContext = Depends(token_required),
    qdrant_client=Depends(get_qdrant_client),
    content_id: Optional[str] = Form(None),
    content_type: str = Form("pdf"),
):
    # Unified endpoint for deleting content (PDF or web)
    try:
        user_id = auth.user_id
        if not user_id or not content_id:
            return JSONResponse(status_code=400, content={"error": "Missing user_id or content_id"})

        # Get content details from database
        content_file = mongo_db_manager.get_content_file_by_id(user_id, content_id)
        if not content_file:
            return JSONResponse(status_code=404, content={"error": "Content not found for this user"})

        # Handle PDF-specific deletion
        if content_type == "pdf" or content_file.get("content_type") == "pdf":
            # Remove PDF file from storage
            pdf_path = os.path.join("pdfs_uploaded/pdfs", f"{content_id}.pdf")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            else:
                logger.warning(f"PDF file {pdf_path} not found for deletion.")

        # Remove from database
        mongo_db_manager.delete_content_file(user_id, content_id)

        # Remove from Qdrant
        qdrant_client.delete_content_by_id(user_id, content_id)

        return JSONResponse(status_code=200, content={
            "message": f"{content_file.get('content_type', 'unknown').upper()} {content_id} deleted for user {user_id}"
        })
    except Exception as e:
        logger.error(f"Delete content error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Error deleting content: {str(e)}"})


@main_router.get("/audio/summary")
def get_summary_audio(
    auth: AuthContext = Depends(token_required),
    content_id: Optional[str] = Form(None),
):
    # Generate and serve summary audio on-demand, with Redis caching
    try:
        user_id = auth.user_id

        if not user_id or not content_id:
            return JSONResponse(status_code=400, content={"error": "Missing user_id or content_id"})

        # Redis cache key
        cache_key = f"audio:summary:{user_id}:{content_id}"
        audio_data = redis_manager.get_audio_cache(cache_key)
        if audio_data:
            logger.info(
                f"[AUDIO CACHE] Served summary audio for user_id={user_id}, content_id={content_id} from Redis cache."
            )
            return Response(content=audio_data, media_type=AUDIO_MIME)

        # Get content file using unified method
        content_file = mongo_db_manager.get_content_file_by_id(user_id, content_id)

        if not content_file:
            return JSONResponse(status_code=404, content={"error": "Content not found for this user"})

        # Get the summary from the stored user data
        summary_text = content_file.get("summary") or ""

        if not summary_text:
            return JSONResponse(status_code=404, content={"error": "No summary found for this content"})

        # Generate audio on-demand
        audio_data = tts_manager.generate_audio_on_demand(summary_text, voice="russell")

        if audio_data is None:
            return JSONResponse(status_code=500, content={"error": "Failed to generate audio"})

        # Store in Redis cache for 10 minutes
        redis_manager.set_audio_cache(cache_key, audio_data, expire_seconds=600)

        # Return the audio data directly
        return Response(content=audio_data, media_type=AUDIO_MIME)

    except Exception as e:
        logger.error(f"Summary audio error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Error generating summary audio: {str(e)}"})


@main_router.get("/audio/query")
def get_query_audio(
    auth: AuthContext = Depends(token_required),
    query_id: Optional[str] = Form(None),
):
    # Generate and serve query audio on-demand using query_id, with Redis caching
    try:
        user_id = auth.user_id

        if not user_id or not query_id:
            return JSONResponse(status_code=400, content={"error": "Missing user_id or query_id"})

        # Redis cache key
        cache_key = f"audio:query:{user_id}:{query_id}"
        audio_data = redis_manager.get_audio_cache(cache_key)
        if audio_data:
            logger.info(
                f"[AUDIO CACHE] Served query audio for user_id={user_id}, query_id={query_id} from Redis cache."
            )
            return Response(content=audio_data, media_type=AUDIO_MIME)

        # Get the specific conversation entry by query_id
        entry = mongo_db_manager.get_entry_by_query_id(user_id, query_id)

        if not entry:
            return JSONResponse(status_code=404, content={"error": "Query not found in history"})

        # Get the assistant's answer text
        text_content = entry.get("assistant answer", "")

        if not text_content:
            return JSONResponse(status_code=404, content={"error": "No text found for this query"})

        # Generate audio on-demand
        audio_data = tts_manager.generate_audio_on_demand(text_content, voice="russell")

        if audio_data is None:
            return JSONResponse(status_code=500, content={"error": "Failed to generate audio"})

        # Store in Redis cache for 10 minutes
        redis_manager.set_audio_cache(cache_key, audio_data, expire_seconds=600)

        # Return the audio data directly
        return Response(content=audio_data, media_type=AUDIO_MIME)

    except Exception as e:
        logger.error(f"Query audio error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Error generating query audio: {str(e)}"})


@main_router.get("/history")
def get_user_history(
    auth: AuthContext = Depends(token_required),
    user_id: Optional[str] = Form(None),
):
    # Get conversation history for a user
    try:
        user_id = user_id or auth.user_id

        history = mongo_db_manager.retrieve_user_history(user_id)

        return JSONResponse(status_code=200, content=history)
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        return JSONResponse(status_code=500, content={"error": f"Error retrieving history: {str(e)}"})


@main_router.delete("/history")
def clear_user_history(
    auth: AuthContext = Depends(token_required),
    user_id: Optional[str] = Form(None),
):
    # Clear conversation history for a user
    try:
        user_id = user_id or auth.user_id

        mongo_db_manager.clear_user_history(user_id)

        return JSONResponse(status_code=200, content={"message": "History cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return JSONResponse(status_code=500, content={"error": f"Error clearing history: {str(e)}"})


def _process_hypothesis_question(projects_api_url, headers, projects):
    import requests
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
            logger.warning(f"Project {pid} detail failed")
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

    return project_map


def handle_hypothesis_faq(auth_token, ai_assistant):
    import requests
    projects_api_url = os.getenv("HYPOTHESIS_DATA_API")
    headers = {"Authorization": auth_token}

    r = requests.get(projects_api_url, headers=headers, timeout=15)
    if r.status_code != 200:
        logger.error(f"Projects API failed: {r.text}")
        return None

    projects = r.json().get("projects", [])
    if not projects:
        logger.warning("No projects found")
        return None

    project_map = _process_hypothesis_question(projects_api_url, headers, projects)

    if not project_map:
        logger.warning("No usable hypothesis data")
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
    llm_response = ai_assistant.advanced_llm.generate(llm_prompt)
    return JSONResponse(status_code=200, content={
        "text": "Here’s your hypothesis-based AI-generated questions:",
        "projects": project_map,
        "sample_questions": llm_response
    })


@main_router.get("/faq")
def get_faq_intro(
    auth: AuthContext = Depends(token_required),
    ai_assistant=Depends(get_ai_assistant),
    context: Optional[str] = Query(None),
):
    """
    Get welcome message and list of FAQ questions.
    """
    try:
        if context == "hypothesis":
            return handle_hypothesis_faq(auth.token, ai_assistant)

        questions = mongo_db_manager.get_all_faq_questions(context)
        question_list = [
            {"id": q["question_id"], "text": q["question_text"], "link": f"/faq/{q['question_id']}"}
            for q in questions
        ]

        return JSONResponse(status_code=200, content={
            "text": "Hello! I’m MOZI, your AI assistant for exploring and annotating "
                "biomedical entities in the BioAtomspace. "
                f"To help you get started, here are some example questions you can try on {context} "
                "Just click one to begin:",
            "questions": question_list
        })
    except Exception as e:
        logger.error(f"Error in FAQ intro: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@main_router.get("/faq/{question_id}")
def get_faq_answer(
    question_id: str,
    auth: AuthContext = Depends(token_required),
):
    """
    Get answer for a FAQ question from MongoDB.
    Returns pre-populated answer instantly.
    """
    try:
        faq = mongo_db_manager.get_faq_by_id(question_id)

        if not faq:
            return JSONResponse(status_code=404, content={
                "error": f"Question ID '{question_id}' not found in FAQ",
                "text": "Use POST /query for custom questions"
            })

        return JSONResponse(status_code=200, content={
            "question": faq["question_text"],
            "text": faq["text"],
            "json_format": faq["json_format"]
        })

    except Exception as e:
        logger.error(f"Error in FAQ answer: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@main_router.get("/")
def health_check():
    return "This is health check"
