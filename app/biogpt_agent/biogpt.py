from transformers import BioGptTokenizer, BioGptForCausalLM
import torch
import logging
from threading import Lock

logger = logging.getLogger(__name__)

class BioGPTAgent:
    """
    Lazy-loaded BioGPT agent.
    """

    _model = None
    _tokenizer = None
    _device = None
    _lock = Lock()

    def __init__(self, llm=None, model_name="kirubel1738/biogpt-bioqa-lora-merged"):
        self.model_name = model_name
        self.llm = llm  

    def generate_answer(self, query: str, max_length: int = 150) -> str:
        """
        under development for storage purposes.
        """
        
        return None

# =================================================================================================
# PREVIOUS OPENVINO IMPLEMENTATION (COMMENTED OUT FOR COMPATIBILITY)
# =================================================================================================
    # def _load_if_needed(self):
    #     """Load model/tokenizer once lazily."""
    #     if BioGPTAgent._model is None:
    #         with BioGPTAgent._lock:
    #             # double-check inside lock
    #             if BioGPTAgent._model is None:
    #                 logger.info("Lazy-loading BioGPT model...")

    #                 BioGPTAgent._tokenizer = BioGptTokenizer.from_pretrained(self.model_name)
    #                 BioGPTAgent._model = BioGptForCausalLM.from_pretrained(self.model_name)

    #                 BioGPTAgent._device = "cuda" if torch.cuda.is_available() else "cpu"
    #                 BioGPTAgent._model.to(BioGPTAgent._device)
    #                 BioGPTAgent._model.eval()  # Set to evaluation mode

    #                 logger.info(f"BioGPT loaded on {BioGPTAgent._device}")

    # def generate_answer(self, query: str, max_length: int = 150) -> str:
    #     try:
    #         self._load_if_needed()

    #         inputs = BioGPTAgent._tokenizer(query, return_tensors="pt").to(BioGPTAgent._device)

    #         with torch.no_grad():
    #             output_ids = BioGPTAgent._model.generate(
    #                 **inputs,
    #                 max_length=max_length,
    #                 do_sample=True,
    #                 temperature=0.7,
    #                 top_p=0.9,
    #                 pad_token_id=BioGPTAgent._tokenizer.eos_token_id,
    #             )

    #         answer = BioGPTAgent._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
    #         # ✅ Remove the question from the start of the answer
    #         if answer.startswith(query):
    #             answer = answer[len(query):].strip()
            
    #         return answer.strip()

    #     except Exception as e:
    #         logger.error(f"Error in BioGPT generation: {str(e)}", exc_info=True)
    #         return f"BIOGPT ERROR: {str(e)}"

            


# from transformers import AutoTokenizer
# from optimum.intel import OVModelForCausalLM
# import torch
# import logging
# from threading import Lock
# import psutil
# import os
# 
# logger = logging.getLogger(__name__)
# 
# class BioGPTAgentOpenVINO:
#     """
#     Lazy-loaded BioGPT agent using Intel OpenVINO for optimized CPU inference.
#     Model: kirubel1738/biogpt-bioqa-8bit-openvino
#     """
# 
#     _model = None
#     _tokenizer = None
#     _device = "CPU"
#     _lock = Lock()
# 
#     def __init__(self, llm=None, model_name="kirubel1738/biogpt-bioqa-8bit-openvino"):
#         self.model_name = model_name
#         self.llm = llm  
#         
#         # Performance settings for minimal RAM and optimized CPU inference
#         # Use environment variable for threads to allow scaling between local and server
#         num_threads = os.getenv("BIOGPT_THREADS", "1")
#         
#         self.ultra_lean_config = {
#             "PERFORMANCE_HINT": "LATENCY",
#             "ENABLE_MMAP": "YES",
#             "CACHE_DIR": "",
#             "INFERENCE_NUM_THREADS": num_threads,         
#             "NUM_STREAMS": "1",                  
#             "KV_CACHE_PRECISION": "u8",          
#         }
# 
#     def _get_ram_usage(self):
#         """Returns the current RAM usage of the process in MB."""
#         process = psutil.Process(os.getpid())
#         return process.memory_info().rss / (1024 * 1024)
# 
#     def _manage_cache_size(self, limit_gb=5.0):
#         """Enforces a size limit on the cache directory by deleting oldest files."""
#         cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
#         if not os.path.exists(cache_dir):
#             return
# 
#         limit_bytes = limit_gb * 1024 * 1024 * 1024
#         files = []
#         total_size = 0
# 
#         # 1. Scan all files
#         for dirpath, _, filenames in os.walk(cache_dir):
#             for f in filenames:
#                 fp = os.path.join(dirpath, f)
#                 try:
#                     stat = os.stat(fp)
#                     total_size += stat.st_size
#                     files.append((stat.st_mtime, stat.st_size, fp))
#                 except OSError:
#                     continue
# 
#         # 2. Check if cleanup is needed
#         if total_size > limit_bytes:
#             logger.warning(f"Cache size ({total_size/1e9:.2f}GB) exceeds limit ({limit_gb}GB). Cleaning up...")
#             
#             # 3. Sort by oldest modified time first
#             files.sort(key=lambda x: x[0])
#             
#             deleted_size = 0
#             for _, size, fp in files:
#                 if total_size - deleted_size <= limit_bytes:
#                     break # Target reached
#                 
#                 try:
#                     os.remove(fp)
#                     deleted_size += size
#                     logger.info(f"Deleted old cache file: {fp} ({size/1e6:.1f}MB)")
#                 except OSError as e:
#                     logger.error(f"Failed to delete {fp}: {e}")
# 
#             logger.info(f"Cache cleanup complete. Freed {deleted_size/1e9:.2f}GB.")
# 
#     def _load_if_needed(self):
#         """Load model/tokenizer once lazily using OpenVINO."""
#         if BioGPTAgent._model is None:
#             with BioGPTAgent._lock:
#                 # double-check inside lock
#                 if BioGPTAgent._model is None:
#                     # Manage cache size before loading to prevent disk overflow
#                     self._manage_cache_size()
#                     
#                     start_ram = self._get_ram_usage()
#                     logger.info(f"Lazy-loading OpenVINO BioGPT model: {self.model_name}...")
#                     logger.info(f"Base RAM Usage: {start_ram:.2f} MB")
# 
#                     # Load tokenizer
#                     BioGPTAgent._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#                     
#                     # Load model using OpenVINO
#                     logger.info("Loading OpenVINO model...")
#                     BioGPTAgent._model = OVModelForCausalLM.from_pretrained(
#                         self.model_name,
#                         ov_config=self.ultra_lean_config
#                     )
#                     
#                     # Set to evaluation mode (though OpenVINO models are typically for inference)
#                     # BioGPTAgent._model.eval() 
# 
#                     after_load_ram = self._get_ram_usage()
#                     logger.info(f"OpenVINO BioGPT loaded successfully.")
#                     logger.info(f"RAM after loading: {after_load_ram:.2f} MB (Added: {after_load_ram - start_ram:.2f} MB)")
# 
#     def generate_answer(self, query: str, max_length: int = 256) -> str:
#         """
#         Generate answer using OpenVINO-optimized BioGPT model.
#         
#         Args:
#             query: The biomedical question
#             max_length: Maximum length for generation (default: 256)
#             
#         Returns:
#             Generated answer text
#         """
#         try:
#             self._load_if_needed()
# 
#             # Encode input
#             inputs = BioGPTAgent._tokenizer(query, return_tensors="pt")
#             
#             # Generate with optimized parameters
#             logger.info(f"Generating answer for: {query}")
#             with torch.no_grad():
#                 output_ids = BioGPTAgent._model.generate(
#                     **inputs,
#                     max_new_tokens=150,  # Generate up to 150 new tokens
#                     pad_token_id=BioGPTAgent._tokenizer.eos_token_id,
#                     use_cache=True,      # Enable key/value cache for faster generation
#                     do_sample=True,      # Enable sampling for non-deterministic responses
#                     temperature=0.7,     # Control randomness (higher = more creative)
#                     top_k=50,            # Sample from top 50 most likely tokens
#                     top_p=0.95           # Nucleus sampling
#                 )
# 
#             # Decode the generated text
#             generated_text = BioGPTAgent._tokenizer.decode(output_ids[0], skip_special_tokens=True)
#             
#             # Extract answer by removing the input query
#             answer = generated_text[len(query):].strip()
#             
#             # Additional cleanup: sometimes the model repeats the question
#             if answer.startswith(query):
#                 answer = answer[len(query):].strip()
#             
#             return answer.strip()
# 
#         except Exception as e:
#             logger.error(f"Error in BioGPT generation: {str(e)}", exc_info=True)
#             return f"BIOGPT ERROR: {str(e)}"
