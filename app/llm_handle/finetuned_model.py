from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model and tokenizer
model_name = "yohannestayezz/biomed_finetuned_gentag" # PharMolix/BioMedGPT-LM-7B model finetuned on genetag dataset from huggingface
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        
        logger.info(f"Received prompt: {prompt[:50]}...")  # Log first 50 chars of the inputted prompt

        if not prompt or not isinstance(prompt, str):
            logger.warning("Invalid or empty prompt received.")
            return jsonify({"error": "Invalid prompt"}), 400

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        logger.info("Text generation successful.")
        return jsonify({"generated_text": generated_text})

    except Exception as e:
        logger.exception("Exception occurred during text generation.")
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(port=8000)