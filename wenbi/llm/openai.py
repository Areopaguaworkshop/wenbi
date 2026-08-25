"""DSPy configuration for OpenAI's Chat Completions-compatible API."""
import os

import dspy


DEFAULT_MODEL = "openai/gpt-5.6-terra"


def get_openai_lm(
    model_name: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 1.0,
    **kwargs,
):
    """Return a DSPy LM backed by the OpenAI API."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for --translation-engine openai")

    model_name = model_name or DEFAULT_MODEL
    if not model_name.startswith("openai/"):
        model_name = f"openai/{model_name}"
    if model_name.removeprefix("openai/").startswith("gpt-5"):
        temperature = 1.0

    return dspy.LM(
        model=model_name,
        api_key=api_key,
        api_base="https://api.openai.com/v1",
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
        **kwargs,
    )