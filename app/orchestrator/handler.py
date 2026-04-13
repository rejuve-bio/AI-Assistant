from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import uuid
import logging
import base64
import sys
import time
from app.calculator import loaders
from .artifacts import ensure_run_dirs, build_manifest, Artifact

from langchain_experimental.tools import PythonREPLTool
from langchain.agents import initialize_agent, AgentType

from app.llm_handle.llm_models import LLMInterface
from app.socket_manager import emit_to_user
from app.prompts.orchestrator_prompts import ORCHESTRATOR_PROMPT_PREFIX

logger = logging.getLogger(__name__)


@dataclass
class CodeExecOptions:
    timeout_seconds: int = 120  # Increased default to 120 seconds
    max_memory_mb: int = 2048  # Increased default to 2048 MB
    output_formats: Optional[List[str]] = None  # e.g., ["png", "csv"]
    allow_network: bool = False
    max_iterations: int = 20  # Maximum agent iterations before stopping


class Orchestrator:
    """Coordinator for the code-execution agent lifecycle with full LLM integration.

    Responsibilities:
    - Ingest and normalize documents (CSV/HTML/XML/PDF/URL)
    - Profile data and plan analysis steps (via ReAct planning with LLM)
    - Execute LLM-generated code safely in a sandbox and collect outputs
    - Render figures/tables and persist artifacts with provenance
    - Iterative error correction via ReAct framework
    """

    def __init__(
        self,
        llm: LLMInterface,
        *,
        artifacts_root: Optional[str] = None,
        rag=None,
        annotation_graph=None,
        hypothesis_generation=None,
        galaxy_handler=None,
        biogpt=None,
        memory_store=None
    ) -> None:
        self.llm = llm
        self.artifacts_root = artifacts_root
        self.rag = rag
        self.annotation_graph = annotation_graph
        self.hypothesis_generation = hypothesis_generation
        self.galaxy_handler = galaxy_handler
        self.biogpt = biogpt
        self.memory_store = memory_store
        logger.info(f"Orchestrator initialized with LLM: {type(llm).__name__}")
        # Convert LLMInterface to LangChain LLM for PythonREPLTool
        self._langchain_llm = self._convert_to_langchain_llm(llm)

    
    def _convert_to_langchain_llm(self, llm: LLMInterface):
        """Convert LLMInterface to LangChain ChatModel for PythonREPLTool agent."""
        from langchain_openai import ChatOpenAI
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_groq import ChatGroq
        import os
        
        # Check if llm has model_provider attribute
        if hasattr(llm, 'model_provider'):
            if llm.model_provider == "openai":
                api_key = getattr(llm, 'api_key', None) or os.getenv("OPENAI_API_KEY")
                model_name = getattr(llm, 'model_name', 'gpt-3.5-turbo')
                return ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)
            elif llm.model_provider == "gemini":
                api_key = getattr(llm, 'api_key', None) or os.getenv("GEMINI_API_KEY")
                model_name = getattr(llm, 'model_name', 'gemini-pro')
                return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
            elif llm.model_provider == "groq":
                api_key = getattr(llm, 'api_key', None) or os.getenv("GROQ_API_KEY")
                model_name = getattr(llm, 'model_name', 'llama-3.1-70b-versatile')
                return ChatGroq(model=model_name, temperature=0, groq_api_key=api_key)
        
        # Fallback: try to detect from class name
        llm_class_name = type(llm).__name__
        if 'OpenAI' in llm_class_name:
            api_key = getattr(llm, 'api_key', None) or os.getenv("OPENAI_API_KEY")
            model_name = getattr(llm, 'model_name', 'gpt-3.5-turbo')
            return ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)
        elif 'Gemini' in llm_class_name:
            api_key = getattr(llm, 'api_key', None) or os.getenv("GEMINI_API_KEY")
            model_name = getattr(llm, 'model_name', 'gemini-pro')
            return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
        elif 'Groq' in llm_class_name:
            api_key = getattr(llm, 'api_key', None) or os.getenv("GROQ_API_KEY")
            model_name = getattr(llm, 'model_name', 'llama-3.1-70b-versatile')
            return ChatGroq(model=model_name, temperature=0, groq_api_key=api_key)
        
        # Default to OpenAI if unsure
        logger.warning(f"Unknown LLM type {llm_class_name}, defaulting to OpenAI")
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

    def load_documents(
        self,
        *,
        files: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load and normalize inputs using loaders module."""
        return loaders.normalize_inputs(files=files, urls=urls)

    def profile_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight profiling: shapes, dtypes, missingness, basic stats.

        Expects `data` from normalize_inputs with key `tables`.
        """
        profiles: List[Dict[str, Any]] = []
        for item in data.get("tables", []):
            df = item.get("dataframe")
            if df is None:
                continue
            try:
                numeric_cols = [c for c in getattr(df, "columns", []) if str(df[c].dtype).startswith(("int", "float"))]
                # Truncate profile info if too many columns
                all_cols = getattr(df, "columns", [])
                if len(all_cols) > 50:
                    # Only keep first 20 for profile to save tokens
                    kept_cols = all_cols[:20]
                    dtypes = {str(c): str(df[c].dtype) for c in kept_cols}
                    missing = {str(c): int(df[c].isna().sum()) for c in kept_cols}
                    # Add a note about truncation in the dict keys or separate field if needed
                    # For now, just limiting the dicts is enough as the prompt shows "Missing values: {...}"
                    dtypes["..."] = f"({len(all_cols)-20} more)"
                    missing["..."] = f"({len(all_cols)-20} more)"
                else:
                    dtypes = {str(c): str(df[c].dtype) for c in all_cols}
                    missing = {str(c): int(df[c].isna().sum()) for c in all_cols}

                desc = df[numeric_cols].describe().to_dict() if numeric_cols else {}
                profiles.append({
                    "shape": [int(df.shape[0]), int(df.shape[1])],
                    "dtypes": dtypes,
                    "missing": missing,
                    "numeric_summary": desc,
                    "source": item.get("source"),
                })
            except Exception:
                profiles.append({"source": item.get("source"), "error": "profiling failed"})

        return {"profiles": profiles}

    # Removed: plan_with_react() - replaced by PythonREPLTool agent
    # Removed: run_in_sandbox() - replaced by PythonREPLTool agent

    def render_artifacts(
        self,
        run_id: str,
        results: Dict[str, Any],
        options: CodeExecOptions,
    ) -> Dict[str, Any]:
        """Write a minimal manifest; figures/tables will be added later."""
        root = self.artifacts_root or os.path.join("tmp", "artifacts")
        dirs = ensure_run_dirs(root, run_id)
        artifacts: List[Artifact] = []
        env = {"limits": {"timeout": options.timeout_seconds, "max_memory_mb": options.max_memory_mb}}
        manifest = build_manifest(run_id, inputs={}, params={}, artifacts=artifacts, environment=env)
        manifest_path = os.path.join(dirs["base"], "manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return {"manifest": manifest, "paths": {"manifest": manifest_path}}

    def execute(
        self,
        *,
        instructions: str,
        files: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        options: Optional[CodeExecOptions] = None,
        user_id: Optional[str] = None,
        token: Optional[str] = None,
        memory_store: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """High-level orchestration using CalculatorAgent as a tool."""
        opts = options or CodeExecOptions()
        run_id = uuid.uuid4().hex
        
        try:
            # Step 1: Load documents
            if user_id:
                emit_to_user(user=user_id, message="Parsing documents...")
            loaded = self.load_documents(files=files, urls=urls, user_id=user_id)
            
            # Step 2: Profile data
            if user_id:
                emit_to_user(user=user_id, message="Profiling data...")
            prof = self.profile_data(loaded)
            
            # Step 3: Set up output directory for artifacts
            root = self.artifacts_root or os.path.join("tmp", "artifacts")
            dirs = ensure_run_dirs(root, run_id)
            output_dir = dirs["figures"]
            
            # Step 4: Prepare context for CalculatorAgent
            if user_id:
                emit_to_user(user=user_id, message="Preparing analysis...")
            
            # Use the passed-in memory store or the instance-level one
            current_memory_store = memory_store or self.memory_store
            
            # Build file paths info (for CSV/HTML/XML - extracted tables)
            file_paths_info = []
            for idx, table_item in enumerate(loaded.get("tables", [])):
                df = table_item.get("dataframe")
                source = table_item.get("source", f"table_{idx}")
                if df is not None:
                    # Save DataFrame to CSV so CalculatorAgent can load it
                    csv_path = os.path.join(output_dir, f"data_{idx}.csv")
                    df.to_csv(csv_path, index=False)
                    
                    # Truncate columns if too many
                    cols = list(df.columns.tolist())
                    if len(cols) > 50:
                        cols_display = cols[:20] + [f"... and {len(cols) - 20} more columns. Use df.columns to inspect."]
                    else:
                        cols_display = cols
                        
                    file_paths_info.append({
                        "path": csv_path,
                        "source": source,
                        "shape": [int(df.shape[0]), int(df.shape[1])],
                        "columns": cols_display
                    })
            
            # Build original files info (for PDF/URL - pass original file paths)
            original_files_info = []
            if files:
                for file_path in files:
                    original_files_info.append({
                        "path": file_path,
                        "type": "original_file",
                        "source": file_path
                    })
            if urls:
                for url in urls:
                    original_files_info.append({
                        "path": url,
                        "type": "url",
                        "source": url
                    })
            
            # Context data to pass to CalculatorAgent
            context_data = {
                "file_paths": file_paths_info,
                "original_files": original_files_info,
                "profiles": prof.get("profiles", [])
            }
            
            # Shared state for side-channel structural data passage (bypassing LLM string limits)
            shared_state = {}
            
            # Step 5: Build all available tools for the Orchestrator
            from app.calculator.agent import CalculatorAgent
            from app.tools.agent_tools import RAGTool, AnnotationTool, HypothesisTool, GalaxyTool, BioGPTTool, MemoryWriteTool, MemoryReadTool
            from langchain.tools import Tool
            
            # Get project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            
            tools = []
            
            # Always add CalculatorTool
            calculator_agent = CalculatorAgent(llm=self._langchain_llm)
            
            # Create a wrapper function that calls calculator_agent.run() with the required parameters
            def calculator_wrapper(query: str) -> str:
                """Wrapper to call CalculatorAgent.run with all required parameters."""
                return calculator_agent.run(
                    instructions=query,
                    context_data=context_data,
                    output_dir=output_dir,
                    project_root=project_root,
                    timeout_seconds=opts.timeout_seconds,
                    max_iterations=opts.max_iterations,
                    max_memory_mb=opts.max_memory_mb
                )
            
            calculator_tool = Tool(
                name="CalculatorTool",
                description="Perform calculations, data analysis, generate plots, and execute Python code on data. Use this for any computational or data analysis tasks.",
                func=calculator_wrapper
            )
            tools.append(calculator_tool)
            
            # Add RAG tool if available
            if self.rag:
                tools.append(RAGTool(rag_instance=self.rag))
        
            # Add Annotation Tool (if enabled)
            if self.annotation_graph:
                tools.append(AnnotationTool(
                    db_handler=self.annotation_graph,
                    token=token,
                    user_id=user_id or "orchestrator",
                    shared_state=shared_state
                ))

            # Add Hypothesis Tool (if enabled)
            if self.hypothesis_generation:
                tools.append(HypothesisTool(
                    hypothesis_instance=self.hypothesis_generation,
                    token=token,
                    user_id=user_id or "orchestrator",
                    shared_state=shared_state
                ))

            # Add Galaxy Tool (if enabled)
            if self.galaxy_handler:
                tools.append(GalaxyTool(galaxy_handler=self.galaxy_handler))

            # Add BioGPT Tool (if enabled)
            if self.biogpt:
                tools.append(BioGPTTool(biogpt_agent=self.biogpt))
            
            # Add Memory Tools (if available)
            if current_memory_store:
                tools.append(MemoryWriteTool(memory_store=current_memory_store))
                tools.append(MemoryReadTool(memory_store=current_memory_store))
            
            # Initialize the orchestrator agent with all tools
            orchestrator = initialize_agent(
                tools=tools,
                llm=self._langchain_llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                agent_kwargs={"prefix": ORCHESTRATOR_PROMPT_PREFIX}
            )
            
            # Step 6: Execute Orchestrator
            if user_id:
                emit_to_user(user=user_id, message="Orchestrating analysis...")
            
            execution_output = ""
            execution_errors = ""
            
            try:
                # Prepend memory context if available
                final_instructions = instructions
                if current_memory_store:
                    memory_summary = current_memory_store.read_all()
                    final_instructions = f"## SESSION MEMORY (Known Facts):\n{memory_summary}\n\n## USER REQUEST:\n{instructions}"
                
                result = orchestrator.run(final_instructions)
                execution_output = str(result)
            except Exception as e:
                logger.error(f"Orchestrator failed: {e}")
                execution_errors = str(e)
                execution_output = f"Orchestration failed: {e}"
            
            # Step 7: Collect artifacts (figures, tables) - same as before
            artifacts_list: List[Artifact] = []
            artifacts_with_data = []
            
            # Scan output directory for generated figures
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    # Skip data CSV files we created
                    if filename.startswith("data_"):
                        continue
                    if filename.endswith(('.png', '.svg', '.pdf', '.html', '.jpg', '.jpeg')):
                        file_path = os.path.join(output_dir, filename)
                        try:
                            size = os.path.getsize(file_path)
                            art = Artifact(
                                id=str(uuid.uuid4()),
                                type="figure",
                                name=os.path.splitext(filename)[0],
                                format=filename.split('.')[-1],
                                path_or_url=file_path,
                                bytes_size=size,
                                metadata={"step": "generated", "filename": filename}
                            )
                            artifacts_list.append(art)
                            
                            # Read and encode image files for direct display (limit to 10MB)
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')) and size < 10 * 1024 * 1024:
                                try:
                                    with open(file_path, 'rb') as f:
                                        image_data = f.read()
                                        base64_data = base64.b64encode(image_data).decode('utf-8')
                                        
                                        # Determine MIME type
                                        ext = filename.lower().split('.')[-1]
                                        mime_type = {
                                            'png': 'image/png',
                                            'jpg': 'image/jpeg',
                                            'jpeg': 'image/jpeg',
                                            'svg': 'image/svg+xml'
                                        }.get(ext, 'image/png')
                                        
                                        artifacts_with_data.append({
                                            "name": art.name,
                                            "format": art.format,
                                            "path": file_path,
                                            "data": f"data:{mime_type};base64,{base64_data}",
                                            "size_bytes": size
                                        })
                                except Exception as e:
                                    logger.warning(f"Failed to encode image {filename}: {e}")
                                    artifacts_with_data.append({
                                        "name": art.name,
                                        "format": art.format,
                                        "path": file_path,
                                        "size_bytes": size
                                    })
                            else:
                                artifacts_with_data.append({
                                    "name": art.name,
                                    "format": art.format,
                                    "path": file_path,
                                    "size_bytes": size
                                })
                        except Exception as e:
                            logger.warning(f"Error processing artifact {filename}: {e}")
            
            # Step 8: Build manifest and return results
            if user_id:
                emit_to_user(user=user_id, message="Rendering artifacts...")
            
            env = {
                "limits": {"timeout": opts.timeout_seconds, "max_memory_mb": opts.max_memory_mb},
                "llm": type(self.llm).__name__,
            }
            manifest = build_manifest(
                run_id=run_id,
                inputs={"files": files or [], "urls": urls or [], "instructions": instructions},
                params={"timeout": opts.timeout_seconds, "max_memory_mb": opts.max_memory_mb},
                artifacts=artifacts_list,
                environment=env
            )
            
            manifest_path = os.path.join(dirs["base"], "manifest.json")
            try:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write manifest: {e}")
            
            # Generate human-readable summary
            summary_lines = [execution_output]
            if artifacts_list:
                summary_lines.append(f"\nGenerated {len(artifacts_list)} artifact(s):")
                for art in artifacts_list:
                    summary_lines.append(f"  - {art.name}.{art.format}")
            
            if execution_errors:
                summary_lines.append(f"\nNote: {execution_errors}")
            
            # Inject preserved graph data from side-channel
            final_resource = {"type": "orchestrator", "id": run_id}
            if shared_state and "resource" in shared_state:
                final_resource = shared_state["resource"]
            
            return {
                "text": "\n".join(summary_lines),
                "manifest": manifest,
                "outputs": [{"step": 1, "description": "Orchestration", "stdout": execution_output, "stderr": execution_errors}],
                "artifacts": artifacts_with_data if artifacts_with_data else [{"name": a.name, "format": a.format, "path": a.path_or_url} for a in artifacts_list],
                "resource": final_resource,
            }
            
        except Exception as e:
            logger.error(f"Error in code execution pipeline: {e}", exc_info=True)
            return {
                "text": f"Error during code execution: {str(e)}",
                "manifest": {},
                "resource": {"type": "orchestrator", "id": run_id},
                "error": str(e),
            }

    # --- Tool-like wrappers to be exposed to the agent later ---

    def tool_load_document(self, *, files: Optional[List[str]], urls: Optional[List[str]]) -> Dict[str, Any]:
        return self.load_documents(files=files, urls=urls)

    def tool_profile_data(self, *, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.profile_data(data)

    # Removed: tool_run_python_sandboxed() - replaced by PythonREPLTool agent

    def tool_render_plot(self, *, run_id: str, kind: str, params: Dict[str, Any], options: Optional[CodeExecOptions] = None) -> Dict[str, Any]:
        # Placeholder; plotting handled in a later chunk
        return {"status": "not_implemented", "kind": kind}

    def tool_export_table(self, *, run_id: str, name: str, fmt: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Placeholder; export handled in a later chunk
        return {"status": "not_implemented", "name": name, "format": fmt}


