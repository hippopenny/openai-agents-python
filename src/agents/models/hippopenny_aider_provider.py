from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from agents import (
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

BASE_URL = os.getenv("HIPPOPENNY_AIDER_BASE_URL") or "http://127.0.0.1:8000/v1"
API_KEY = os.getenv("HIPPOPENNY_AIDER_API_KEY") or "auto"
MODEL_NAME = os.getenv("HIPPOPENNY_AIDER_MODEL_NAME") or "aider"


"""This is a custom provider to hippo penny aider openai proxy server. Steps:
1. Create a custom OpenAI client.
2. Create a ModelProvider that uses the custom client.
"""

class HippoPennyAiderModelProvider(ModelProvider):
    set_tracing_disabled(disabled=True)
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, 
                                          openai_client=AsyncOpenAI(base_url=BASE_URL, 
                                                                    api_key=API_KEY))

