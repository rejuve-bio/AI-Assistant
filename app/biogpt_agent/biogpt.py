import torch
import logging
import time
from threading import Lock, Semaphore
import os
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
import psutil

logger = logging.getLogger(__name__)

def _compute_biogpt_concurrency():
    """
    Auto-scale BioGPT CPU usage based on available cores, reserving a fraction
    for other production services running on the same server.

    BIOGPT_CPU_FRACTION  — share of total cores available to BioGPT (default 0.5)
    BIOGPT_THREADS       — CPU threads per single inference (default: biogpt_cores // 4, min 1)
    BIOGPT_MAX_CONCURRENT — max parallel inferences (default: biogpt_cores // threads, min 1)
    All three are overridable via env vars for any deployment.
    """
    total_cores = os.cpu_count() or 1
    fraction = float(os.getenv("BIOGPT_CPU_FRACTION", "0.1"))
    biogpt_cores = max(1, int(total_cores * fraction))

    threads = int(os.getenv("BIOGPT_THREADS", str(max(1, biogpt_cores // 2))))
    max_concurrent = int(os.getenv("BIOGPT_MAX_CONCURRENT", "1"))

    logger.info(
        f"BioGPT CPU allocation: {biogpt_cores}/{total_cores} cores "
        f"({fraction*100:.0f}%), {threads} threads/inference, "
        f"{max_concurrent} max concurrent"
    )
    return threads, max_concurrent


class BioGPTAgentOpenVINO:
    """
    Lazy-loaded BioGPT agent using Intel OpenVINO for optimized CPU inference.
    Model: kirubel1738/biogpt-bioqa-8bit-openvino
    """

    _model = None
    _tokenizer = None
    _device = "CPU"
    _lock = Lock()

    _threads_per_inference, _max_concurrent = _compute_biogpt_concurrency()
    _inference_semaphore: Semaphore = Semaphore(_max_concurrent)
    # In-memory response cache: {normalized_query: (timestamp, answer)}
    _response_cache: dict = {}
    _cache_ttl: int = int(os.getenv("BIOGPT_CACHE_TTL", "3600"))

    def __init__(self, llm=None, model_name="kirubel1738/biogpt-bioqa-8bit-openvino"):
        self.model_name = model_name
        self.llm = llm

        self.ultra_lean_config = {
            "PERFORMANCE_HINT": "LATENCY",
            "ENABLE_MMAP": "YES",
            "CACHE_DIR": os.getenv("BIOGPT_CACHE_DIR", "/root/.cache/huggingface/openvino_cache"),
            "INFERENCE_NUM_THREADS": str(BioGPTAgentOpenVINO._threads_per_inference),
            "NUM_STREAMS": "1",
            "KV_CACHE_PRECISION": "u8",
        }

    def _get_ram_usage(self):
        """Returns the current RAM usage of the process in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def _scan_cache_files(self, cache_dir: str):
        """Scans cache_dir recursively and returns (files, total_size)."""
        files = []
        total_size = 0
        for dirpath, _, filenames in os.walk(cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    stat = os.stat(fp)
                    total_size += stat.st_size
                    files.append((stat.st_mtime, stat.st_size, fp))
                except OSError:
                    continue
        return files, total_size

    def _manage_cache_size(self, limit_gb=5.0):
        """Enforces a size limit on the cache directory by deleting oldest files."""
        cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        if not os.path.exists(cache_dir):
            return

        limit_bytes = limit_gb * 1024 * 1024 * 1024
        files, total_size = self._scan_cache_files(cache_dir)

        # Check if cleanup is needed
        if total_size > limit_bytes:
            logger.warning(f"Cache size ({total_size/1e9:.2f}GB) exceeds limit ({limit_gb}GB). Cleaning up...")

            # Sort by oldest modified time first
            files.sort(key=lambda x: x[0])

            deleted_size = 0
            for _, size, fp in files:
                if total_size - deleted_size <= limit_bytes:
                    break # Target reached

                try:
                    os.remove(fp)
                    deleted_size += size
                    logger.info(f"Deleted old cache file: {fp} ({size/1e6:.1f}MB)")
                except OSError as e:
                    logger.error(f"Failed to delete {fp}: {e}")

            logger.info(f"Cache cleanup complete. Freed {deleted_size/1e9:.2f}GB.")

    def _load_if_needed(self):
        """Load model/tokenizer once lazily using OpenVINO."""
        if BioGPTAgentOpenVINO._model is None:
            with BioGPTAgentOpenVINO._lock:
                # double-check inside lock
                if BioGPTAgentOpenVINO._model is None:
                    # Manage cache size before loading to prevent disk overflow
                    self._manage_cache_size()
                    
                    start_ram = self._get_ram_usage()
                    logger.info(f"Lazy-loading OpenVINO BioGPT model: {self.model_name}...")
                    logger.info(f"Base RAM Usage: {start_ram:.2f} MB")

                    # Load tokenizer
                    BioGPTAgentOpenVINO._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    
                    # Load model using OpenVINO
                    logger.info("Loading OpenVINO model...")
                    BioGPTAgentOpenVINO._model = OVModelForCausalLM.from_pretrained(
                        self.model_name,
                        ov_config=self.ultra_lean_config
                    )
                    
                    # Set to evaluation mode (though OpenVINO models are typically for inference)
                    # BioGPTAgentOpenVINO._model.eval() 

                    after_load_ram = self._get_ram_usage()
                    logger.info(f"OpenVINO BioGPT loaded successfully.")
                    logger.info(f"RAM after loading: {after_load_ram:.2f} MB (Added: {after_load_ram - start_ram:.2f} MB)")

    def generate_answer(self, query: str, max_length: int = 200) -> str:
        """Generate answer using OpenVINO-optimized BioGPT model."""
        cache_key = query.strip().lower()

        # Return cached answer if still fresh
        cached = BioGPTAgentOpenVINO._response_cache.get(cache_key)
        if cached:
            ts, answer = cached
            if time.time() - ts < BioGPTAgentOpenVINO._cache_ttl:
                logger.info(f"BioGPT cache hit for: {query[:80]}")
                return answer

        try:
            self._load_if_needed()

            logger.info(f"Generating answer for: {query}")
            with BioGPTAgentOpenVINO._inference_semaphore:
                inputs = BioGPTAgentOpenVINO._tokenizer(query, return_tensors="pt")
                with torch.no_grad():
                    output_ids = BioGPTAgentOpenVINO._model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        pad_token_id=BioGPTAgentOpenVINO._tokenizer.eos_token_id,
                        use_cache=True,
                        do_sample=False,
                    )

            generated_text = BioGPTAgentOpenVINO._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            answer = generated_text[len(query):].strip()
            if answer.startswith(query):
                answer = answer[len(query):].strip()
            answer = answer.strip()

            BioGPTAgentOpenVINO._response_cache[cache_key] = (time.time(), answer)
            return answer

        except Exception as e:
            logger.error(f"Error in BioGPT generation: {str(e)}", exc_info=True)
            return f"BIOGPT ERROR: {str(e)}"
