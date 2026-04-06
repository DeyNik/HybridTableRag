"""
llm/factory.py
==============
LLM client factory. Reads LLM_PROVIDER from .env and returns the correct client.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (safe to call multiple times)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "azure_openai").strip().lower()

    if provider == "gemini":
        from hybridtablerag.llm.gemini_client import GeminiClient
        model = os.getenv("LLM_MODEL")
        if not model:
            raise ValueError("LLM_MODEL is required for Gemini")
        return GeminiClient(model_name=model)

    elif provider in ("openai", "open_ai"):
        from hybridtablerag.llm.openai_client import OpenAIClient
        model = os.getenv("LLM_MODEL")
        if not model:
            raise ValueError("LLM_MODEL is required for OpenAI")
        return OpenAIClient(model_name=model)

    elif provider in ("azure_openai", "azure"):
        from hybridtablerag.llm.azureopenai_client import AzureOpenAIClient
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("LLM_MODEL")
        if not deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT or LLM_MODEL required for Azure")
        return AzureOpenAIClient(deployment_name=deployment)

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'")