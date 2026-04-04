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
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    
    # Gemini 
    if provider == "gemini":
        from hybridtablerag.llm.gemini_client import GeminiClient
        return GeminiClient(model_name=os.getenv("LLM_MODEL"))

    # OpenAI 
    elif provider in ("openai", "open_ai"):
        from hybridtablerag.llm.openai_client import OpenAIClient
        return OpenAIClient(model_name=os.getenv("LLM_MODEL"))

    # Azure OpenAI 
    elif provider in ("azure_openai", "azure"):
        # Azure uses deployment name, fallback to LLM_MODEL if not set
        from hybridtablerag.llm.azureopenai_client import AzureOpenAIClient
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("LLM_MODEL")
        return AzureOpenAIClient(deployment_name=deployment)

    # Fallback
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'.\n"
            f"Supported: gemini | openai | azure_openai | nvidia"
        )