"""OpenAI-compatible LLM client.

Pointed at OpenRouter by default. Override with OPENROUTER_BASE_URL +
MODEL_NAME env vars to use any other OpenAI-compatible endpoint.
"""

import os

from openai import AsyncOpenAI


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url=os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
    )


def model_name() -> str:
    return os.environ.get("MODEL_NAME", "z-ai/glm-5")


def tools_enabled() -> bool:
    """Master switch for whether the generic LLM loop exposes our own
    tools (web_search, web_fetch, ...) to the model.

    Set via Amber config per deployment. Turn OFF for benchmarks where
    the green agent has already baked task-specific tools into its system
    prompt (e.g. tau2-bench) so the model doesn't mix tool surfaces.
    """
    raw = os.environ.get("TOOLS_ENABLED", "true").strip().lower()
    return raw not in ("0", "false", "no", "off", "")
