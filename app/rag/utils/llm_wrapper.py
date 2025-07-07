import os
from openai import OpenAI
import google.generativeai as genai


class LLMWrapper:
    """
    Wrapper for OpenAI and Google Gemini LLM APIs.
    Provides unified interface for chat completions across different providers.
    """

    def __init__(self, use_openai=True):
        self.use_openai = use_openai

        if use_openai:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel("gemini-1.5-flash-latest")

    def chat(self, system_prompt, user_prompt):
        if self.use_openai:
            resp = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        else:
            prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.model.generate_content(prompt)
            return response.text
