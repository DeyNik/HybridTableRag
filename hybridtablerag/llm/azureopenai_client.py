import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from hybridtablerag.llm.base import BaseLLM

load_dotenv()


class AzureOpenAIClient(BaseLLM):

    def __init__(
        self,
        deployment_name: str = None,
        api_version: str = None,
    ):
        api_key   = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint  = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        version   = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found. Check your .env file.")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT not found. Check your .env file.")
        if not deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT not found. Check your .env file.")

        self.deployment_name = deployment
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=version,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment_name,   # Azure uses deployment name here, not model name
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()