import os
from dotenv import load_dotenv
from openai import OpenAI
from hybridtablerag.llm.base import BaseLLM

load_dotenv()

class OpenAIClient(BaseLLM):

    def __init__(self, model_name: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Check your .env file.")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content