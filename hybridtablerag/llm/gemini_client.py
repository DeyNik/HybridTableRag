"""
llm/gemini_client.py
====================
BUG FIXED: was calling self.client.generate_text() which does not exist
in the google-generativeai SDK.  Correct method is model.generate_content().
"""

import os
import google.generativeai as genai
from hybridtablerag.llm.base import BaseLLM


class GeminiClient(BaseLLM):

    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment or passed explicitly.")

        genai.configure(api_key=api_key)

        # Store model name for re-use
        self.model_name = model_name

        # GenerativeModel is the correct entry point for generate_content()
        self.model = genai.GenerativeModel(model_name=model_name)

    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        """
        Send a prompt and return the text response.

        Uses generate_content() — the correct SDK method.
        The old code called self.client.generate_text() which:
          1. 'Client' has no attribute 'generate_text'  → AttributeError
          2. Even if it did, generate_text is deprecated and removed in v0.5+
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini generate_content failed: {e}") from e