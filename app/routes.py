from typing import List, Optional

from app.lib.auth import token_required, AuthContext
from fastapi import APIRouter, Depends, Form, File, UploadFile, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv
import logging
import re
import traceback
import json
import os
from app.rag.utils.tts_utils import tts_manager
from app.rag.rag import ALLOWED_FILE_EXTENSIONS
from app.storage.redis import redis_manager
from app.storage.mongo_storage import mongo_db_manager
from app.prompts.quick_reply_prompts import QUICK_REPLIES_PROMPT
from app.prompts.hypothesis_prompt import EXAMPLE_QUESTIONS_PROMPT

load_dotenv()
main_router = APIRouter()
logger = logging.getLogger(__name__)

AUDIO_MIME = "audio/mpeg"

_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE
)


def _is_uuid4(value) -> bool:
    return isinstance(value, str) and bool(_UUID4_RE.match(value))


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


def _dispatch_query(ai_assistant, question, json_query, uploaded_files, upload_results, content_ids, user_id, auth_token, graph_id, resource, url, resume=None, thread_id=None, client_message_id=None):
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
        client_message_id=client_message_id,
    )
    if response.get("status") == "busy":
        return JSONResponse(status_code=409, content=response)
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
    client_message_id: Optional[str] = Form(None),
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
    - If content_ids are provided, answers are retrieved only from those content items. If omitted, any content_ids this thread has already
      uploaded or referenced are recalled automatically; only a thread with no such history falls back to the general collection.
    - Handles both user-uploaded content question answering and general knowledge queries.
    """
    try:
        user_id = auth.user_id

        if thread_id and mongo_db_manager.thread_owner_conflict(thread_id, user_id):
            return JSONResponse(status_code=404, content={"text": "No such thread for this user."})

        question_parsed, graph_id, resource, url, json_query_parsed, content_ids = _parse_context(
            {"context": context, "json_query": json_query, "question": question, "query": query}
        )

        upload_results = []
        if uploaded_files:
            upload_results, content_ids = _handle_uploads(uploaded_files, ai_assistant, user_id, content_ids)

        if content_ids and thread_id:
            mongo_db_manager.add_thread_content_ids(thread_id, user_id, content_ids)
        elif thread_id:
            remembered = mongo_db_manager.get_thread_content_ids(thread_id, user_id)
            if remembered:
                content_ids = remembered

        return _dispatch_query(ai_assistant, question_parsed, json_query_parsed, uploaded_files, upload_results, content_ids, user_id, auth.token, graph_id, resource, url, resume, thread_id, client_message_id)
    except Exception as e:
        logger.error(f"Exception: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"text": f"Error processing request: {e}"})


@main_router.get("/user_status/documents/")
def user_status(
    auth: AuthContext = Depends(token_required),
    user_id: Optional[str] = Form(None),
):
    try:
        user_id = user_id or auth.user_id
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "Missing user_id"})

        all_content_files = mongo_db_manager.get_user_content_files(user_id)

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

        total_count = mongo_db_manager.get_content_count(user_id)
        pdf_count = mongo_db_manager.get_content_count(user_id, "pdf")
        web_count = mongo_db_manager.get_content_count(user_id, "web")

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

    llm_prompt = EXAMPLE_QUESTIONS_PROMPT.format(project_map_json=json.dumps(project_map, indent=2))
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


@main_router.get("/v1/threads")
def list_threads(
    auth: AuthContext = Depends(token_required),
):
    """Thread summaries for the authenticated user, latest activity first.
    Read-only -- exposes the existing conversation_threads data, no new
    storage introduced. See the Threads API v1 proposal."""
    try:
        threads = mongo_db_manager.list_threads(auth.user_id)
        return JSONResponse(status_code=200, content={
            "items": threads,
            "next_cursor": None,
        })
    except Exception as e:
        logger.error(f"Error listing threads: {e}")
        return JSONResponse(status_code=502, content={"text": f"Error listing threads: {e}"})


@main_router.get("/v1/threads/{thread_id}")
def get_thread(
    thread_id: str,
    auth: AuthContext = Depends(token_required),
):
    """A single thread's summary. 404 if it doesn't belong to this user --
    another user's thread_id counts as missing, same as the v1 proposal's
    error convention."""
    try:
        summary = mongo_db_manager.get_thread_summary(thread_id, auth.user_id)
        if not summary:
            return JSONResponse(status_code=404, content={"text": "No such thread for this user."})
        return JSONResponse(status_code=200, content=summary)
    except Exception as e:
        logger.error(f"Error getting thread {thread_id}: {e}")
        return JSONResponse(status_code=502, content={"text": f"Error getting thread: {e}"})


_VALID_SOURCES_DIR = {
    "human": "app/annotation_graph/schema/human/data_source_schemas",
    "fly": "app/annotation_graph/schema/fly/data_source_schemas",
}


def _valid_sources_for_species(species: str) -> Optional[set]:
    """The real, recognized data sources for a species, read from the
    actual annotation-query-backend schema files (not invented) --
    used to validate context.sources instead of accepting anything.
    Returns None if the species isn't recognized at all."""
    directory = _VALID_SOURCES_DIR.get((species or "").lower())
    if not directory or not os.path.isdir(directory):
        return None
    return {os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith(".yaml")}


_DEFAULT_SOURCES = {
    "human": ["GENCODE", "UniProt", "GTEx"],
    "fly": ["FLYBASE"],
}

_DEFAULT_SPECIES = "human"


def _resolve_sources(species: str, sources) -> tuple:
    """Returns (resolved_sources, error_message). If sources is empty,
    fills in the species' default. If sources is given, validates every
    entry against the real schema files -- 400 if anything's unrecognized."""
    if not sources:
        return _DEFAULT_SOURCES.get((species or "").lower(), []), None

    valid = _valid_sources_for_species(species)
    if valid is None:
        return sources, None
    unknown = [s for s in sources if s not in valid]
    if unknown:
        return None, f"Unknown source(s) for {species}: {unknown}. Valid sources: {sorted(valid)}"
    return sources, None


def _generate_quick_replies(llm, question: str, summary: str) -> list:
    try:
        prompt = QUICK_REPLIES_PROMPT.format(question=question, summary=summary)
        result = llm.generate(prompt)
        text = result if isinstance(result, str) else str(result)
        lines = [l.strip(" -\t") for l in text.strip().splitlines() if l.strip()]
        return lines[:2] if lines else []
    except Exception as e:
        logger.warning(f"quick_replies generation failed, skipping: {e}")
        return []


def _build_proposal_title(data: dict, kind: Optional[str] = None) -> str:
    if kind == "hypothesis":
        phenotype = (data or {}).get("phenotype") or "a phenotype"
        variant = (data or {}).get("variant")
        if (data or {}).get("credible_set_id") and variant:
            return f"{variant} → {phenotype.lower()}"
        return f"A locus in {phenotype.lower()}"

    if kind == "galaxy":
        target = (data or {}).get("target") or {}
        return target.get("name") or target.get("id") or "A Galaxy run"

    if kind == "hypothesis-project":
        return (data or {}).get("project_name") or (data or {}).get("phenotype") or "A new hypothesis project"

    nodes = {n.get("node_id"): n for n in (data or {}).get("nodes", []) if n.get("node_id")}
    predicates = (data or {}).get("predicates", []) or []
    if not nodes:
        return "Annotation"

    def label(node: dict) -> str:
        for v in (node.get("properties") or {}).values():
            if isinstance(v, str) and v.strip():
                return v.strip()
        node_type = (node.get("type") or "entity").strip() or "entity"
        return f"{node_type}s"

    targets = {p.get("target") for p in predicates}
    roots = [nid for nid in nodes if nid not in targets]
    current = roots[0] if roots else next(iter(nodes))

    chain, seen = [], set()
    while current in nodes and current not in seen:
        seen.add(current)
        chain.append(label(nodes[current]))
        current = next((p.get("target") for p in predicates if p.get("source") == current), None)

    return " → ".join(chain) if chain else "Annotation"


def _format_v1_message(thread_id: str, m: dict, index: int) -> dict:
    """Shared shaping of one stored message entry into the Threads API v1
    {id, thread_id, role, created_at, client_message_id?, parts}
    """
    msg_num = m.get("seq", index)
    if m.get("context_part"):
        parts = [{"type": "context", **m["context_part"]}]
    elif m.get("run_part"):
        parts = [{"type": "run", **m["run_part"]}]
    else:
        parts = [{"type": "text", "text": m.get("text", "")}]
        if m.get("files_part"):
            parts.append({"type": "files", "files": m["files_part"]})
        if m.get("role") == "user" and m.get("graph_id"):
            parts.append({"type": "about", "kind": m.get("resource_type") or "annotation", "id": m["graph_id"]})
        if m.get("proposals"):
            for i, prop in enumerate(m["proposals"]):
                kind = prop.get("kind") or "annotation"
                data = prop.get("data")
                suffix = f"_{i}" if len(m["proposals"]) > 1 else ""
                part = {"type": "proposal", "id": f"{thread_id}_msg_{msg_num}_prop{suffix}", "kind": kind,
                        "title": _build_proposal_title(data, kind=kind), "data": data}
                if prop.get("replaces"):
                    part["replaces"] = prop["replaces"]
                parts.append(part)
        elif m.get("json_format"):
            kind = m.get("resource_type") or "annotation"
            parts.append({"type": "proposal", "id": f"{thread_id}_msg_{msg_num}_prop", "kind": kind,
                          "title": _build_proposal_title(m["json_format"], kind=kind), "data": m["json_format"]})
        if m.get("quick_replies"):
            parts.append({"type": "quick_replies", "replies": m["quick_replies"]})
    created_at = m.get("created_at")
    return {
        "id": f"{thread_id}_msg_{msg_num}",
        "thread_id": thread_id,
        "role": m.get("role"),
        "created_at": mongo_db_manager._iso(created_at) if hasattr(created_at, "isoformat") else created_at,
        "client_message_id": m.get("client_message_id"),
        "parts": parts,
    }


@main_router.get("/v1/threads/{thread_id}/messages")
def get_thread_messages(
    thread_id: str,
    auth: AuthContext = Depends(token_required),
):
    """A thread's messages, oldest first, read from the existing stored
    turns. Each message's id is built from its stable, stored `seq`
    (see _format_v1_message) -- read-only, no schema change."""
    try:
        doc = mongo_db_manager.get_thread(thread_id, auth.user_id)
        if not doc.get("messages") and not mongo_db_manager.get_thread_summary(thread_id, auth.user_id):
            return JSONResponse(status_code=404, content={"text": "No such thread for this user."})

        items = [_format_v1_message(thread_id, m, i) for i, m in enumerate(doc.get("messages", []))]
        return JSONResponse(status_code=200, content={
            "items": items,
            "next_cursor": None,
        })
    except Exception as e:
        logger.error(f"Error getting messages for thread {thread_id}: {e}")
        return JSONResponse(status_code=502, content={"text": f"Error getting messages: {e}"})


@main_router.post("/v1/files")
async def post_v1_files(
    files: List[UploadFile] = File(...),
    auth: AuthContext = Depends(token_required),
    ai_assistant=Depends(get_ai_assistant),
):
    """Accept and store one or more files, per-user, no embedding -- step 1
    only (RAG embedding is a separate, later decision). Returns {id, name}
    per file so the client can reference them in a 'files' part on
    /v1/query right away."""
    items = []
    errors = []
    for f in files:
        result = ai_assistant.rag.save_raw_file(f, auth.user_id)
        if isinstance(result, dict) and result.get("error"):
            errors.append({"name": f.filename, "error": result["error"]})
        elif isinstance(result, dict) and result.get("content_id"):
            items.append({"id": result["content_id"], "name": result.get("name") or f.filename})
        else:
            errors.append({"name": f.filename, "error": "Upload failed."})

    if not items:
        return JSONResponse(status_code=400, content={"text": "No file could be uploaded.", "errors": errors})

    body = {"items": items}
    if errors:
        body["errors"] = errors
    return JSONResponse(status_code=201, content=body)


@main_router.get("/v1/files/supported-types")
def get_supported_file_types():
    """Return accepted file extensions grouped by category so the client
    can filter its file picker and show a help tooltip."""
    categories = {
        "gwas_summary_stats": [e for e in sorted(ALLOWED_FILE_EXTENSIONS)
                               if e in {".tsv", ".txt", ".csv", ".gz", ".bgz"}],
        "sequence_data":      [e for e in sorted(ALLOWED_FILE_EXTENSIONS)
                               if "fastq" in e or "fq" in e],
        "reference_sequences": [e for e in sorted(ALLOWED_FILE_EXTENSIONS)
                                if "fasta" in e or e in {".fa", ".fa.gz"}],
        "genomics":           [e for e in sorted(ALLOWED_FILE_EXTENSIONS)
                               if e in {".vcf", ".vcf.gz", ".bed", ".bam", ".sam"}],
        "gene_annotations":   [e for e in sorted(ALLOWED_FILE_EXTENSIONS)
                               if e in {".gff", ".gff3", ".gtf"}],
        "documents":          [e for e in sorted(ALLOWED_FILE_EXTENSIONS)
                               if e in {".pdf"}],
    }
    return JSONResponse(status_code=200, content={
        "supported_extensions": sorted(ALLOWED_FILE_EXTENSIONS),
        "categories": categories,
        "max_file_size_mb": 50,
    })


@main_router.post("/v1/query")
async def post_v1_query(
    request: Request,
    auth: AuthContext = Depends(token_required),
    ai_assistant=Depends(get_ai_assistant),
):
    """Send a message. Unlike /query, this never executes anything -- it
    only proposes (annotation only, in v1). Matches 'Send a message' in
    the Threads API v1 proposal.

    Accepts either application/json (unchanged), or multipart/form-data
    with a "body" field holding the same JSON plus the actual file(s)
    attached in the same request -- lets a client attach a brand-new file
    and send the message in one round trip instead of always needing a
    separate /v1/files upload first. A files part entry can still name an
    already-uploaded file by "id" (unchanged) or, only in the multipart
    case, by "name" to match one of the files attached to this request.
    """
    try:
        content_type = request.headers.get("content-type", "")
        uploaded_by_name = {}
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            try:
                body = json.loads(form.get("body") or "{}")
            except (TypeError, ValueError):
                return JSONResponse(status_code=400, content={"text": "'body' form field must be valid JSON."})
            for value in form.values():
                if getattr(value, "filename", None):
                    uploaded_by_name[value.filename] = value
        else:
            body = await request.json()
        thread_id = body.get("thread_id")
        client_message_id = body.get("client_message_id")
        context = body.get("context")
        parts = body.get("parts") or []

        if not thread_id:
            return JSONResponse(status_code=400, content={"text": "thread_id is required."})
        if not _is_uuid4(thread_id):
            return JSONResponse(status_code=400, content={"text": "thread_id must be a UUIDv4."})

        user_id = auth.user_id

        if mongo_db_manager.thread_owner_conflict(thread_id, user_id):
            return JSONResponse(status_code=404, content={"text": "No such thread for this user."})

        is_new_thread = mongo_db_manager.get_thread_summary(thread_id, user_id) is None

        resolved_sources = None
        if is_new_thread:
            species = (context or {}).get("species") or _DEFAULT_SPECIES
            resolved_sources, source_err = _resolve_sources(species, (context or {}).get("sources"))
            if source_err:
                return JSONResponse(status_code=400, content={"text": source_err})
            context = {"species": species, "sources": resolved_sources}

        allowed_part_types = {"text", "files", "about"}
        if any(p.get("type") not in allowed_part_types for p in parts):
            return JSONResponse(status_code=400, content={"text": "Only text, files and about parts are accepted here."})

        question = " ".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
        if not question:
            return JSONResponse(status_code=400, content={"text": "A text part is required."})

        about = next((p for p in parts if p.get("type") == "about"), None)
        graph_id = about.get("id") if about else None
        resource = about.get("kind") if about else None

        files_part_raw = next((p for p in parts if p.get("type") == "files"), None)
        resolved_files = None
        if files_part_raw:
            requested = files_part_raw.get("files") or []
            resolved_files = []
            for entry in requested:
                if not isinstance(entry, dict):
                    continue
                file_id = entry.get("id")
                if file_id:
                    found = mongo_db_manager.get_content_file_by_id(user_id, file_id)
                    if not found:
                        return JSONResponse(status_code=400, content={"text": f"No such file '{file_id}' for this user."})
                    resolved_files.append({"id": found["content_id"], "name": found.get("filename") or file_id})
                    continue
                name = entry.get("name")
                attached = uploaded_by_name.get(name) if name else None
                if not attached:
                    return JSONResponse(status_code=400, content={"text": f"A files part entry must have either 'id' or a 'name' matching a file attached to this request (got: {entry})."})
                upload_result = ai_assistant.rag.save_raw_file(attached, user_id)
                if not isinstance(upload_result, dict) or not upload_result.get("content_id"):
                    error = upload_result.get("error") if isinstance(upload_result, dict) else "Upload failed."
                    return JSONResponse(status_code=400, content={"text": f"Could not store attached file '{name}': {error}"})
                resolved_files.append({"id": upload_result["content_id"], "name": upload_result.get("name") or name})
            if not resolved_files:
                resolved_files = None

        # Auto-attach uploaded files to the thread's context resources and content_ids
        if resolved_files and thread_id:
            content_ids = [f["id"] for f in resolved_files]
            mongo_db_manager.add_thread_content_ids(thread_id, user_id, content_ids)
            for f in resolved_files:
                mongo_db_manager.add_thread_resource(thread_id, user_id, "file", f["id"])

        if client_message_id:
            existing = mongo_db_manager.find_message_by_client_id(thread_id, user_id, client_message_id)
            if existing:
                doc = mongo_db_manager.get_thread(thread_id, user_id)
                idx = next((i for i, m in enumerate(doc["messages"]) if m.get("client_message_id") == client_message_id), len(doc["messages"]) - 1)
                return JSONResponse(status_code=200, content=_format_v1_message(thread_id, doc["messages"][idx], idx))

        opening_context = None
        if is_new_thread and context:
            opening_context = mongo_db_manager.set_thread_context(
                thread_id, user_id, context.get("species"), resolved_sources
            )["context"]

        if not redis_manager.acquire_thread_lock(thread_id):
            return JSONResponse(status_code=409, content={"text": "A reply is already in progress in this thread."})

        try:
            agent_response = await run_in_threadpool(
                ai_assistant.agent,
                question, user_id, auth.token,
                graph_id=graph_id, resource=resource,
                thread_id=thread_id, propose_only=True,
                attached_files=resolved_files,
            )
        except Exception as e:
            logger.error(f"/v1/query failed: {e}", exc_info=True)
            return JSONResponse(status_code=502, content={"text": f"The model or a tool failed: {e}"})
        finally:
            redis_manager.release_thread_lock(thread_id)

        quick_replies = None
        if agent_response.get("json_format"):
            quick_replies = _generate_quick_replies(
                ai_assistant.basic_llm, question, agent_response.get("text", "")
            ) or None

        ai_assistant._record_thread_turn(
            thread_id, user_id, question, agent_response,
            graph_id=graph_id, resource=resource, client_message_id=client_message_id,
            files_part=resolved_files, context_to_record=opening_context,
            quick_replies=quick_replies,
        )

        doc = mongo_db_manager.get_thread(thread_id, user_id)
        last_idx = len(doc["messages"]) - 1
        formatted = _format_v1_message(thread_id, doc["messages"][last_idx], last_idx)

        # Include attached files summary at the top level so the client
        # can update its file-context UI without re-fetching the thread.
        if resolved_files:
            formatted["attached_files"] = resolved_files

        return JSONResponse(status_code=201, content=formatted)
    except Exception as e:
        logger.error(f"Error in /v1/query: {e}", exc_info=True)
        return JSONResponse(status_code=502, content={"text": f"Error processing request: {e}"})


@main_router.put("/v1/threads/{thread_id}/context")
async def put_thread_context(
    thread_id: str,
    request: Request,
    auth: AuthContext = Depends(token_required),
):
    """Sets a thread's species/sources. 201 with a divider message if it
    actually changed, 204 if the new value matches what's already set --
    matching the Threads API v1 proposal exactly."""
    try:
        if mongo_db_manager.thread_owner_conflict(thread_id, auth.user_id):
            return JSONResponse(status_code=404, content={"text": "No such thread for this user."})

        body = await request.json()
        species = body.get("species")
        sources_in = body.get("sources")
        resources_in = body.get("resources")
        if not species:
            return JSONResponse(status_code=400, content={"text": "species is required."})
        if sources_in is not None and not isinstance(sources_in, list):
            return JSONResponse(status_code=400, content={"text": "sources must be a list."})
        if resources_in is not None:
            if not isinstance(resources_in, list):
                return JSONResponse(status_code=400, content={"text": "resources must be a list."})
            for r in resources_in:
                if not isinstance(r, dict) or r.get("kind") not in ("annotation", "hypothesis", "file") or not r.get("id"):
                    return JSONResponse(status_code=400, content={"text": "Each resource needs a kind ('annotation', 'hypothesis', or 'file') and an id."})

        sources, err = _resolve_sources(species, sources_in)
        if err:
            return JSONResponse(status_code=400, content={"text": err})

        logger.info(
            f"PUT /v1/threads/{thread_id}/context -- user={auth.user_id}, species={species!r}, "
            f"sources={sources_in!r}, resources={resources_in!r}"
        )
        result = mongo_db_manager.set_thread_context(thread_id, auth.user_id, species, sources, resources_in)
        if not result["changed"]:
            logger.info(f"PUT /v1/threads/{thread_id}/context -- no change, returning 204.")
            return Response(status_code=204)

        thread_doc = mongo_db_manager.append_message(
            thread_id, auth.user_id, "system", "",
            context_part=result["context"],
        )
        idx = thread_doc.get("message_count", 1) - 1
        return JSONResponse(status_code=201, content=_format_v1_message(thread_id, thread_doc["messages"][-1], idx))
    except Exception as e:
        logger.error(f"Error setting context for thread {thread_id}: {e}")
        return JSONResponse(status_code=502, content={"text": f"Error setting context: {e}"})


@main_router.post("/v1/threads/{thread_id}/messages")
async def record_thread_run(
    thread_id: str,
    request: Request,
    auth: AuthContext = Depends(token_required),
):
    """Records that a run started elsewhere (the platform server's own
    call to annotation/hypothesis), purely so the thread shows a live-status
    card. No model call, no compute -- matches 'Record a run' in the
    Threads API v1 proposal."""
    try:
        if mongo_db_manager.thread_owner_conflict(thread_id, auth.user_id):
            return JSONResponse(status_code=404, content={"text": "No such thread for this user."})

        body = await request.json()
        client_message_id = body.get("client_message_id")
        parts = body.get("parts") or []
        run_parts = [p for p in parts if p.get("type") == "run"]
        if len(run_parts) != 1:
            return JSONResponse(status_code=400, content={"text": "Exactly one 'run' part is required."})

        if client_message_id:
            existing = mongo_db_manager.find_message_by_client_id(thread_id, auth.user_id, client_message_id)
            if existing:
                return JSONResponse(status_code=200, content={"text": "Already recorded."})

        run_part = {k: v for k, v in run_parts[0].items() if k != "type"}
        logger.info(f"POST /v1/threads/{thread_id}/messages -- user={auth.user_id}, run_part={run_part!r}")
        if run_part.get("kind") in ("annotation", "hypothesis") and run_part.get("id"):
            mongo_db_manager.add_thread_resource(thread_id, auth.user_id, run_part["kind"], run_part["id"])
        thread_doc = mongo_db_manager.append_message(
            thread_id, auth.user_id, "system", "",
            client_message_id=client_message_id,
            run_part=run_part,
        )
        idx = thread_doc.get("message_count", 1) - 1
        return JSONResponse(status_code=201, content=_format_v1_message(thread_id, thread_doc["messages"][-1], idx))
    except Exception as e:
        logger.error(f"Error recording run for thread {thread_id}: {e}")
        return JSONResponse(status_code=502, content={"text": f"Error recording run: {e}"})


@main_router.patch("/v1/threads/{thread_id}")
async def rename_thread(
    thread_id: str,
    request: Request,
    auth: AuthContext = Depends(token_required),
):
    """Renames a thread. Clearing the title (empty string) falls back to the
    first message's text again, so it's never empty -- matches 'The thread'
    in the Threads API v1 proposal."""
    try:
        if mongo_db_manager.thread_owner_conflict(thread_id, auth.user_id):
            return JSONResponse(status_code=404, content={"text": "No such thread for this user."})

        body = await request.json()
        title = body.get("title")

        summary = mongo_db_manager.get_thread_summary(thread_id, auth.user_id)
        if not summary:
            return JSONResponse(status_code=404, content={"text": "No such thread for this user."})

        mongo_db_manager.set_thread_title(thread_id, auth.user_id, title)
        return JSONResponse(status_code=200, content=mongo_db_manager.get_thread_summary(thread_id, auth.user_id))
    except Exception as e:
        logger.error(f"Error renaming thread {thread_id}: {e}")
        return JSONResponse(status_code=502, content={"text": f"Error renaming thread: {e}"})


@main_router.delete("/v1/threads/{thread_id}")
def delete_thread_route(
    thread_id: str,
    auth: AuthContext = Depends(token_required),
):
    """Deletes a thread. Never touches the annotations/hypotheses/galaxy
    runs it referenced -- those live entirely in their own services."""
    try:
        if mongo_db_manager.thread_owner_conflict(thread_id, auth.user_id):
            return JSONResponse(status_code=404, content={"text": "No such thread for this user."})
        deleted = mongo_db_manager.delete_thread(thread_id, auth.user_id)
        if not deleted:
            return JSONResponse(status_code=404, content={"text": "No such thread for this user."})
        return Response(status_code=204)
    except Exception as e:
        logger.error(f"Error deleting thread {thread_id}: {e}")
        return JSONResponse(status_code=502, content={"text": f"Error deleting thread: {e}"})


@main_router.delete("/v1/threads/{thread_id}/messages/{message_id}")
def delete_thread_message(
    thread_id: str,
    message_id: str,
    auth: AuthContext = Depends(token_required),
):
    """Deletes one message by its stable id (<thread_id>_msg_<seq>) -- e.g.
    lets a user dismiss a run card without touching the real run. seq is
    stamped once when the message is created, so this id never changes
    even if an earlier message in the thread gets deleted."""
    try:
        if mongo_db_manager.thread_owner_conflict(thread_id, auth.user_id):
            return JSONResponse(status_code=404, content={"text": "No such thread or message for this user."})

        prefix = f"{thread_id}_msg_"
        if not message_id.startswith(prefix):
            return JSONResponse(status_code=404, content={"text": "No such thread or message for this user."})
        try:
            seq = int(message_id[len(prefix):])
        except ValueError:
            return JSONResponse(status_code=404, content={"text": "No such thread or message for this user."})

        deleted = mongo_db_manager.delete_message(thread_id, auth.user_id, seq)
        if not deleted:
            return JSONResponse(status_code=404, content={"text": "No such thread or message for this user."})
        return Response(status_code=204)
    except Exception as e:
        logger.error(f"Error deleting message {message_id} in thread {thread_id}: {e}")
        return JSONResponse(status_code=502, content={"text": f"Error deleting message: {e}"})


@main_router.get("/")
def health_check():
    return "This is health check"
