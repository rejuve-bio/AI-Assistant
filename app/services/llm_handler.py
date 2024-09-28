# handles different types of llm models
import os
import requests
from dotenv import load_dotenv
import openai 
from openai import OpenAI

load_dotenv()

MODEL_NAME="gpt-4o"
class Gemini:
    def __init__(self):
        self.api_key = os.environ.get("API_KEY")
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    
    def __call__(self, prompt: str):
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{"parts":[{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "topK": 1,
                "topP": 1
            }
        }
        response = requests.post(f"{self.api_url}?key={self.api_key}", headers=headers, json=data)
        response.raise_for_status()

        # print("[INFO] Successfully received response from Gemini API.")
        content = response.json()['candidates'][0]['content']['parts'][0]['text']
        return content


class OpenAI():

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_APIKEY")
        self.client = OpenAI()

    def __call__(self, prompt):
        completion = self.client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        response = completion.choices[0].message
        return response.content
    