from hybridtablerag.llm.ollama_client import OllamaClient
from hybridtablerag.llm.openai_client import OpenAIClient
from hybridtablerag.llm.gemini_client import GeminiClient

def get_llm(provider: str, model_name: str):

    provider = provider.lower()

    if provider == "ollama":
        return OllamaClient(model_name)

    elif provider == "openai":
        return OpenAIClient(model_name)

    elif provider == "gemini":
        return GeminiClient(model_name)

    else:
        raise ValueError(f"Unsupported provider: {provider}")