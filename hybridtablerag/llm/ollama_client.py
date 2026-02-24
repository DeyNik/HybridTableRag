import ollama
from hybridtablerag.llm.base import BaseLLM

class OllamaClient(BaseLLM):

    def __init__(self, model_name: str = "qwen2.5:14b"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        return response["message"]["content"]