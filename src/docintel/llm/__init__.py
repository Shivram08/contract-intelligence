"""Model clients, chosen by model name.

Every caller speaks the Anthropic Messages shape; Gemini is adapted to it in
``docintel.llm.gemini``. The provider is a function of the model id alone, so a
results row that names its model also names its provider.
"""

from __future__ import annotations

from typing import Any

__all__ = ["is_gemini", "make_client"]


def is_gemini(model: str) -> bool:
    return model.startswith("gemini-")


def make_client(model: str) -> Any:
    """A live client for ``model``. Raises if the provider has no credentials."""
    from docintel.config import resolve_anthropic_api_key, resolve_gemini_api_key

    if is_gemini(model):
        from docintel.llm.gemini import GeminiClient

        key = resolve_gemini_api_key()
        if key is None:
            raise RuntimeError(f"{model} needs a Gemini key: set GCP_KEY or GEMINI_API_KEY in .env")
        return GeminiClient(api_key=key)

    import anthropic

    api_key = resolve_anthropic_api_key()
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
