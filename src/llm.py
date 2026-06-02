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


def retrieval_enabled() -> bool:
    """Whether to expose the `retrieve_documents` tool (OfficeQA corpus).

    The tool is backed by a baked-in FAISS + BM25 index over the
    Treasury Bulletin corpus. Turn ON for OfficeQA; leave OFF elsewhere
    so other benchmarks don't see an irrelevant tool in their schema.
    """
    raw = os.environ.get("RETRIEVAL_ENABLED", "false").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def reasoning_enabled() -> bool:
    """Whether to let the model use its hidden reasoning channel.

    Reasoning models (e.g. GLM-5) sometimes burn the entire completion
    budget thinking and emit zero visible content (`finish_reason=stop`,
    `msg.content==""`). For benchmarks where that bug bites — currently
    tau2-bench — turn this OFF via the amber manifest. Keep ON elsewhere
    so the model can do single-shot reasoning when it helps.
    """
    raw = os.environ.get("REASONING_ENABLED", "true").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def json_output() -> bool:
    """Enforce JSON-object output on the LLM via OpenAI's response_format.

    Set via Amber config per deployment. Turn ON for benchmarks whose
    green agent parses our response as JSON (e.g. tau2-bench expects
    {"name": "<tool>", "arguments": {...}}). Off by default — most
    benchmarks want natural-language responses.

    Note: the OpenAI API requires the literal word "JSON" to appear
    somewhere in the prompt when this is enabled, otherwise it 400s.
    For benchmarks where this flag is on, the green's policy text is
    expected to mention JSON.
    """
    raw = os.environ.get("JSON_OUTPUT", "false").strip().lower()
    return raw not in ("0", "false", "no", "off", "")
