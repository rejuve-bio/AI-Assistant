from __future__ import annotations
import os
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import initialize_agent, AgentType
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

class CalculatorAgent:
    """
    Specialized agent for performing calculations, data analysis, and plotting.
    It uses the PythonREPLTool to execute code in a controlled environment.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(
        self,
        instructions: str,
        context_data: Dict[str, Any],
        output_dir: str,
        project_root: str,
        timeout_seconds: int = 120,
        max_iterations: int = 20,
        max_memory_mb: int = 2048
    ) -> str:
        """
        Execute the calculator agent with the given instructions and context.
        
        Args:
            instructions: The specific analysis request (e.g., "Plot the mean of column X").
            context_data: Dictionary containing 'file_paths', 'original_files', 'profiles'.
            output_dir: Directory to save generated artifacts.
            project_root: Root directory of the project (for sys.path).
            timeout_seconds: Max execution time.
            max_iterations: Max agent steps.
            max_memory_mb: Memory limit (passed to prompt/env).
            
        Returns:
            str: The final output/summary of the execution.
        """
        run_id = uuid.uuid4().hex
        logger.info(f"CalculatorAgent running with instructions: {instructions}")

        # 1. Setup PythonREPLTool
        # We add project_root to sys.path via globals or prompt instructions
        repl_globals = {
            "__builtins__": __builtins__,
        }
        # Note: We rely on the prompt to insert sys.path for now, as PythonREPLTool 
        # might reset state. But we can also try to inject it if the tool supports it.
        python_repl = PythonREPLTool(globals=repl_globals)
        tools = [python_repl]

        # 2. Build the specialized prompt
        prompt = self._build_enhanced_prompt(
            instructions=instructions,
            file_paths=context_data.get("file_paths", []),
            original_files=context_data.get("original_files", []),
            profiles=context_data.get("profiles", []),
            output_dir=output_dir,
            project_root=project_root
        )

        # 3. Initialize the ReAct Agent
        try:
            agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=max_iterations,
                max_execution_time=timeout_seconds,
            )
        except TypeError:
            # Fallback for older LangChain versions
            logger.warning("Agent initialization doesn't support max_iterations/max_execution_time, using defaults")
            agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
            )

        # 4. Execute with Retry Logic
        max_retries = 3
        retry_delay = 2
        execution_output = ""

        for attempt in range(max_retries):
            try:
                result = agent.run(prompt)
                execution_output = str(result) if result else "Execution completed"
                break
            except Exception as e:
                error_str = str(e)
                logger.warning(f"CalculatorAgent attempt {attempt + 1} error: {error_str}")

                is_limit_error = (
                    "iteration limit" in error_str.lower() or
                    "time limit" in error_str.lower() or
                    "stopped due to" in error_str.lower()
                )

                if is_limit_error:
                    execution_output = f"Stopped due to limits: {error_str}"
                    break

                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    execution_output = f"Failed after retries: {error_str}"

        return execution_output

    def _build_enhanced_prompt(
        self,
        instructions: str,
        file_paths: List[Dict[str, Any]],
        original_files: List[Dict[str, Any]],
        profiles: List[Dict[str, Any]],
        output_dir: str,
        project_root: str
    ) -> str:
        """Build enhanced prompt for PythonREPLTool with file paths, profiles, and helper info."""
        prompt_parts = [
            "You are a Python data analysis assistant. Execute Python code to complete the user's request.",
            "",
            "CRITICAL RULES:",
            "1. DO NOT install packages - use only libraries that are already installed",
            "2. DO NOT use subprocess, pip, or any package installation commands",
            "3. Use the loaders module to load files if needed: from app.calculator.loaders import load_pdf, load_csv, load_html, load_xml, load_url",
            "4. For PDFs, use pdfplumber or camelot (already installed) - DO NOT use tabula-py",
            "5. IF DATASET SCHEMA IS LARGE OR TRUNCATED: You MUST run `df.columns` or `df.head()` to inspect the data before writing your analysis code. Do NOT guess column names.",
            "",
            "USER REQUEST:",
            instructions,
            "",
        ]
        
        # Add original file paths (PDF/URL)
        if original_files:
            prompt_parts.append("ORIGINAL FILES (if you need to process them directly):")
            for orig_file in original_files:
                file_path = orig_file['path']
                file_type = orig_file.get('type', 'file')
                if file_type == 'url':
                    prompt_parts.append(f"  - URL: {file_path}")
                    prompt_parts.append(f"    To load: from app.calculator.loaders import load_url; data = load_url('{file_path}')")
                else:
                    ext = os.path.splitext(file_path)[1].lower()
                    prompt_parts.append(f"  - File: {file_path} (type: {ext})")
                    if ext == '.pdf':
                        prompt_parts.append(f"    To load: from app.calculator.loaders import load_pdf; data = load_pdf('{file_path}')")
                        prompt_parts.append(f"    Or use: import pdfplumber; with pdfplumber.open('{file_path}') as pdf: tables = [page.extract_table() for page in pdf.pages]")
                    elif ext == '.csv':
                        prompt_parts.append(f"    To load: import pandas as pd; df = pd.read_csv('{file_path}')")
                    elif ext in ['.html', '.htm']:
                        prompt_parts.append(f"    To load: from app.calculator.loaders import load_html; data = load_html('{file_path}')")
                    elif ext == '.xml':
                        prompt_parts.append(f"    To load: from app.calculator.loaders import load_xml; data = load_xml('{file_path}')")
            prompt_parts.append("")
        
        # Add extracted tables
        if file_paths:
            prompt_parts.append("EXTRACTED TABLES (already loaded as CSV files):")
            for fp_info in file_paths:
                prompt_parts.append(f"  - File: {fp_info['path']}")
                prompt_parts.append(f"    Source: {fp_info['source']}")
                prompt_parts.append(f"    Shape: {fp_info['shape']}")
                prompt_parts.append(f"    Columns: {fp_info['columns']}")
            prompt_parts.append("")
            prompt_parts.append("To load these CSV files, use:")
            prompt_parts.append("  import pandas as pd")
            prompt_parts.append("  df = pd.read_csv('path_to_file.csv')")
            prompt_parts.append("")
        
        if profiles:
            prompt_parts.append("DATA PROFILES:")
            for prof in profiles[:3]:
                if "error" not in prof:
                    prompt_parts.append(f"  Source: {prof.get('source', 'unknown')}")
                    prompt_parts.append(f"    Shape: {prof.get('shape')}")
                    prompt_parts.append(f"    Missing values: {prof.get('missing', {})}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "OUTPUT DIRECTORY:",
            f"  Save all plots to: {output_dir}",
            "",
            "IMPORTANT: Complete the task efficiently. Use helper functions when available:",
            f"  1. Add project root to sys.path: sys.path.insert(0, r'{project_root}')",
            "  2. Import helpers: from app.calculator.plotting import save_correlation_heatmap",
            "  3. For correlation heatmap: save_correlation_heatmap(df, os.path.join(output_dir, 'heatmap.png'))",
            "  4. For statistics: from app.calculator.stats import correlations",
            "",
            "AVAILABLE LIBRARIES:",
            "  - pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns)",
            "  - pdfplumber (for PDF text/table extraction)",
            "  - camelot (for PDF table extraction)",
            "  - app.calculator.loaders (load_pdf, load_csv, load_html, load_xml, load_url)",
            "",
            "EXECUTE CODE NOW to complete: " + instructions,
        ])
        
        return "\n".join(prompt_parts)
