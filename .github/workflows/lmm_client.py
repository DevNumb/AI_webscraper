# src/llm_client.py
import os
import requests
from typing import Optional, List, Dict, Any

from langchain.llms.base import LLM
from langchain.schema import LLMResult

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# A thin helper to call OpenRouter
def call_openrouter_model(model: str, messages: List[Dict[str,str]], extra: Optional[Dict]=None, timeout=60):
    payload = {
        "model": model,
        "messages": messages
    }
    if extra:
        payload["extra_body"] = extra
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    # openrouter returns choices[0].message.content
    return data

# LangChain-compatible LLM wrapper for OpenRouter
class OpenRouterLLM(LLM):
    model_name: str = "deepseek-r1:free"
    temperature: float = 0.0

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Using chat format; we put prompt in a single user message
        msgs = [{"role": "user", "content": prompt}]
        extra = {"temperature": self.temperature}
        data = call_openrouter_model(self.model_name, msgs, extra=extra)
        content = data["choices"][0]["message"]["content"]
        return content

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # For simplicity not implemented (synchronous)
        raise NotImplementedError
