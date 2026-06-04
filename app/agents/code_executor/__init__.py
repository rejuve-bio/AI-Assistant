from ._subprocess import Subprocess
from typing import List, Optional
import logging
import os

logger = logging.getLogger(__name__)

MAX_RETRIES = 5


TOOL_EXT = {
    "python": "script.py",
    "R":      "script.R",
    "plink":  "script.sh",
    "bash":   "script.sh",
}

CODE_GEN_PROMPT = """You are an expert bioinformatics programmer working inside the Rejuve AI Assistant platform.

Task instruction:
{step_input}

Available input files in the input/ directory:
{available_files}

── BIOMNI PLATFORM TOOLS (use these instead of reimplementing) ──────────────
{biomni_functions}

All Biomni tools are importable in your Python script. Import paths:
  from app.tools.biomni.database_connectors import (
      query_uniprot, query_alphafold, query_string, query_kegg,
      query_opentargets, query_clinvar, query_gnomad, query_ensembl,
      query_cbioportal, query_reactome, query_gwas_catalog, query_openfda,
      query_chembl, query_pdb, query_gtex, query_encode, blast_sequence,
      run_deseq2
  )
  from app.tools.biomni.genetics import run_finemapping, liftover, identify_tf_binding_sites
  from app.tools.biomni.genomics import run_gsea, annotate_scrna, compute_scrna_embeddings
  from app.tools.biomni.pharmacology import predict_admet, get_drug_interactions, predict_binding_affinity, run_docking, drug_repurposing
  from app.tools.biomni.molecular_biology import design_sgrna, design_primers, simulate_restriction_digest
  from app.tools.biomni.literature import search_pubmed, search_arxiv, search_scholar, get_doi_supplementary, search_clinical_trials, extract_url_content
  from app.tools.biomni.data_lake import query_depmap, query_disgenets, query_bindingdb, query_msigdb, query_omim

ALWAYS prefer Biomni tools over reimplementing database queries or analysis from scratch.
──────────────────────────────────────────────────────────────────────────────

Write a complete, self-contained {tool} script that performs this task.

Rules:
- Input files are in the relative path input/ — do NOT use absolute paths
- Write all output files to output/
- Print a clear, human-readable summary of results to stdout
- For PLINK scripts: use plink2 (on PATH); prefix output with output/result
- Standard Python libraries available: pandas, numpy, scipy, matplotlib, seaborn,
  scikit-learn, gseapy, networkx, biopython, statsmodels, scanpy, anndata, pydeseq2
- Standard R libraries: ggplot2, dplyr, DESeq2, limma, edgeR, WGCNA, survival
- Do NOT include markdown fences, just raw code

Write the complete code now:"""

FIX_PROMPT = """Fix the following {tool} code.

Error to fix: {fix_instruction}

Stderr output:
{stderr}

Current code:
{code}

Return ONLY the corrected code, no markdown."""

CRITIC_PROMPT = """A {tool} script was executed for this task:
'{step_input}'

Exit code: {exit_code}
Stdout (truncated): {stdout}
Stderr (truncated): {stderr}

Evaluate the result and respond with EXACTLY one of:
PASS       — output looks correct and complete
FIX: <short description of what to change in the code>
FAIL: <reason this cannot be fixed by code changes>"""


class CodeExecutor:
    def __init__(self, advanced_llm, basic_llm, biomni_retriever=None):
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.sandbox = Subprocess()
        self.biomni_retriever = biomni_retriever

    def execute(self, step_input: str, tool: str, step_id: int, user_id: str,
                retry_count: int = 0, file_paths: Optional[List[str]] = None,
                session_id: str = None) -> dict:
        code = self._generate_code(step_input, tool, file_paths)
        logger.info(
            f"\n{'='*60}\n"
            f"[STEP {step_id}] GENERATED {tool.upper()} CODE\n"
            f"{'='*60}\n{code}\n{'='*60}"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(f"[STEP {step_id}] attempt {attempt}/{MAX_RETRIES} | tool={tool} | files={len(file_paths or [])}")

            # Fast syntax check — skip sandbox entirely if code won't parse
            syntax_ok, syntax_err = self._syntax_check(code, tool)
            if not syntax_ok:
                logger.warning(f"[STEP {step_id}] Syntax error (attempt {attempt}): {syntax_err}")
                code = self._fix_code(
                    code,
                    {"stderr": syntax_err, "stdout": "", "exit_code": 1, "output_files": []},
                    syntax_err, tool,
                )
                continue

            result = self.sandbox.run(code=code, tool=tool, file_paths=file_paths,
                                      user_id=user_id, step_id=step_id, session_id=session_id)

            logger.info(
                f"[STEP {step_id}] exit_code={result.get('exit_code')} | "
                f"stdout={len(result.get('stdout',''))}chars | stderr={len(result.get('stderr',''))}chars"
            )
            if result.get("stdout"):
                logger.info(f"[STEP {step_id}] STDOUT:\n{result['stdout'][:1000]}")
            if result.get("stderr"):
                logger.warning(f"[STEP {step_id}] STDERR:\n{result['stderr'][:500]}")

            verdict, fix = self._critic(step_input, code, result, tool)
            logger.info(f"[STEP {step_id}] critic={verdict}" + (f" | {fix[:120]}" if fix else ""))

            if verdict == "pass":
                logger.info(f"[STEP {step_id}] PASSED on attempt {attempt} | files={result.get('output_files', [])}")
                return result

            if verdict == "hard_fail":
                logger.warning(f"[STEP {step_id}] HARD FAIL: {fix}")
                result["error"] = fix
                return result

            logger.info(f"[STEP {step_id}] applying fix for attempt {attempt + 1}")
            code = self._fix_code(code, result, fix, tool)
            logger.info(
                f"\n{'='*60}\n"
                f"[STEP {step_id}] FIXED {tool.upper()} CODE (attempt {attempt + 1})\n"
                f"{'='*60}\n{code}\n{'='*60}"
            )

        result["error"] = f"Execution failed after {MAX_RETRIES} attempts."
        return result

    def _syntax_check(self, code: str, tool: str) -> tuple:
        """Returns (ok, error_message). Runs before the sandbox to catch obvious errors free."""
        if tool == "python":
            import ast
            try:
                ast.parse(code)
                return True, ""
            except SyntaxError as e:
                return False, f"SyntaxError at line {e.lineno}: {e.msg}"

        elif tool in ("bash", "plink"):
            import tempfile, subprocess
            with tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
                f.write(code)
                tmp = f.name
            try:
                r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True, timeout=5)
                return (True, "") if r.returncode == 0 else (False, r.stderr.strip())
            except Exception:
                return True, ""
            finally:
                os.unlink(tmp)

        elif tool == "R":
            import tempfile, subprocess
            with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
                f.write(code)
                tmp = f.name
            try:
                r = subprocess.run(["Rscript", "--parse", tmp], capture_output=True, text=True, timeout=5)
                return (True, "") if r.returncode == 0 else (False, r.stderr.strip())
            except Exception:
                return True, ""
            finally:
                os.unlink(tmp)

        return True, ""

    def _generate_code(self, step_input: str, tool: str,
                        file_paths: Optional[List[str]] = None) -> str:
        if file_paths:
            available = "\n".join(f"  - input/{os.path.basename(p)}" for p in file_paths)
        else:
            available = "  (no files uploaded — generate example data or use public datasets)"

        # Inject relevant Biomni tool signatures for Python scripts.
        # R/bash/plink scripts can call Python tools via subprocess if needed,
        # but the primary integration path is Python.
        biomni_section = ""
        if self.biomni_retriever:
            relevant = self.biomni_retriever.get_relevant_functions(step_input)
            if relevant:
                biomni_section = relevant + "\n"

        prompt = CODE_GEN_PROMPT.format(
            step_input=step_input, tool=tool,
            available_files=available, biomni_functions=biomni_section,
        )
        try:
            code = self.advanced_llm.generate(prompt)
            return _strip_fences(code)
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return f'print("Code generation failed: {e}")'

    def _critic(self, step_input: str, code: str, result: dict, tool: str):
        if result.get("error") and not result.get("stdout") and not result.get("stderr"):
            return "hard_fail", result["error"]

        if result.get("exit_code") == 0 and result.get("stdout"):
            return "pass", ""

        prompt = CRITIC_PROMPT.format(
            tool=tool,
            step_input=step_input[:200],
            exit_code=result.get("exit_code", -1),
            stdout=result.get("stdout", "")[:400],
            stderr=result.get("stderr", "")[:400],
        )
        try:
            verdict_raw = self.basic_llm.generate(prompt).strip()
        except Exception:
            return "retry_with_fix", "Fix any syntax or import errors."

        if verdict_raw.upper().startswith("PASS"):
            return "pass", ""
        if verdict_raw.upper().startswith("FAIL:"):
            return "hard_fail", verdict_raw[5:].strip()
        fix = verdict_raw[4:].strip() if verdict_raw.upper().startswith("FIX:") else verdict_raw
        return "retry_with_fix", fix

    def _fix_code(self, code: str, result: dict, fix_instruction: str, tool: str) -> str:
        prompt = FIX_PROMPT.format(
            tool=tool,
            fix_instruction=fix_instruction,
            stderr=result.get("stderr", "")[:500],
            code=code[:2000],
        )
        try:
            return _strip_fences(self.advanced_llm.generate(prompt))
        except Exception:
            return code


def _strip_fences(text: str) -> str:
    for lang in ("python", "r", "R", "bash", "sh", ""):
        text = text.replace(f"```{lang}", "")
    return text.replace("```", "").strip()
