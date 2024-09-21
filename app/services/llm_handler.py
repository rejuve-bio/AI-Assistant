# handles different types of llm models
import os
import google.generativeai as genai

class Gemini:

    def __init__(self) -> None:
        genai.configure(api_key=os.environ.get("API_KEY"))

    def __call__(self, prompt: str):
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response._result.candidates[0].content.parts[0].text     
