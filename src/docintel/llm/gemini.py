"""Gemini behind the Anthropic Messages interface.

The agent loop, both single-call baselines, the response cache and the API all
speak one shape: ``client.messages.create(model=..., system=..., tools=...,
messages=...)`` returning content blocks of ``type`` text / tool_use. This
module makes Gemini's ``generateContent`` REST endpoint answer in that shape,
so switching provider is a client swap and nothing upstream changes. Keeping
one request shape also keeps the cache honest: Sonnet and Gemini entries live
side by side, separated by the ``model`` field in the key.

Plain ``httpx`` rather than the google-genai SDK, for the same reason the cache
stores plain JSON: a committed, replayable artifact must not depend on an SDK
version's object model.

Three things differ from Anthropic and matter:

**Thought signatures are mandatory on echo.** Gemini 3 attaches an opaque
``thoughtSignature`` to function-call parts, and a later request that sends the
call back without it is rejected with HTTP 400 (verified live). The signature
rides on the content block as ``thought_signature``, so it survives the loop's
echo, the cache's JSON round trip, and replay -- and is part of the cache key
exactly as a live turn's would be.

**No strict tool schemas.** Anthropic validates ``strict: true`` tool input
server-side. Gemini's ``parametersJsonSchema`` constrains generation but makes no
such guarantee, so ``strict`` is dropped and the loop's
retry-on-parse-failure path becomes the only structural guarantee. A
``MALFORMED_FUNCTION_CALL`` finish surfaces as a turn with no tool call, which
the harness counts as a schema failure rather than retrying it away here.

**Thinking tokens are output tokens.** ``thoughtsTokenCount`` is billed at the
output rate, so it is folded into ``output_tokens``. Thinking level is only sent
when the request carries ``thinking`` -- a keyed cache field -- so a change of
level can never replay an entry recorded at another level.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Final

import httpx

__all__ = ["GEMINI_BASE_URL", "Block", "GeminiAPIError", "GeminiClient", "GeminiResponse"]

GEMINI_BASE_URL: Final = "https://generativelanguage.googleapis.com/v1beta"

#: Finish reasons that mean the provider declined to answer. Mapped to the
#: Anthropic ``refusal`` stop reason, which the loop already handles.
_REFUSALS: Final = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}
)

#: Worth retrying: rate limiting and transient server errors. Anything else is
#: a request we built wrong, and retrying it only spends time.
_RETRYABLE_STATUS: Final = frozenset({429, 500, 502, 503, 504})


class GeminiAPIError(RuntimeError):
    """A non-retryable error, or a retryable one that never cleared."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Gemini API {status}: {body[:500]}")
        self.status = status


class Block(dict):  # type: ignore[type-arg]
    """A content block: a dict (so it serializes and echoes) with attribute access.

    Same contract as the cache's replay block, deliberately: a live block and a
    replayed block must canonicalize identically, or every turn after a cache
    hit misses. Only non-None keys are stored.
    """

    def __init__(self, **fields: Any) -> None:
        super().__init__({k: v for k, v in fields.items() if v is not None})

    def __getattr__(self, name: str) -> Any:
        return self.get(name)


class _Usage:
    __slots__ = (
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "input_tokens",
        "output_tokens",
        "thinking_tokens",
    )

    def __init__(self, metadata: dict[str, Any]) -> None:
        prompt = int(metadata.get("promptTokenCount", 0) or 0)
        prompt += int(metadata.get("toolUsePromptTokenCount", 0) or 0)
        cached = int(metadata.get("cachedContentTokenCount", 0) or 0)
        thoughts = int(metadata.get("thoughtsTokenCount", 0) or 0)
        # promptTokenCount includes the cached prefix; split it so cached
        # tokens are priced at the cache rate, never twice.
        self.input_tokens = max(prompt - cached, 0)
        self.cache_read_input_tokens = cached
        self.cache_creation_input_tokens = 0
        self.output_tokens = int(metadata.get("candidatesTokenCount", 0) or 0) + thoughts
        #: Already inside ``output_tokens``; kept separately so the cost of
        #: thinking is visible rather than inferred.
        self.thinking_tokens = thoughts

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


class GeminiResponse:
    """Quacks like an Anthropic ``Message`` for everything the loop reads."""

    def __init__(self, payload: dict[str, Any], model: str) -> None:
        self.model = model
        self.usage = _Usage(payload.get("usageMetadata") or {})
        self.stop_details: dict[str, Any] | None = None
        self.content: list[Block] = []

        candidates = payload.get("candidates") or []
        if not candidates:
            # The prompt itself was blocked: no candidate at all.
            reason = (payload.get("promptFeedback") or {}).get("blockReason", "BLOCKED")
            self.stop_reason = "refusal"
            self.stop_details = {"category": str(reason).lower()}
            return

        candidate = candidates[0]
        finish = str(candidate.get("finishReason", "STOP"))
        for index, part in enumerate((candidate.get("content") or {}).get("parts") or []):
            if part.get("thought"):
                continue  # a thought summary, not an answer
            signature = part.get("thoughtSignature")
            call = part.get("functionCall")
            if call is not None:
                self.content.append(
                    Block(
                        type="tool_use",
                        id=call.get("id") or f"call_{index}",
                        name=call.get("name", ""),
                        input=call.get("args") or {},
                        thought_signature=signature,
                    )
                )
            elif "text" in part:
                self.content.append(
                    Block(type="text", text=part["text"], thought_signature=signature)
                )

        if finish in _REFUSALS:
            self.stop_reason = "refusal"
            self.stop_details = {"category": finish.lower()}
        elif finish == "MAX_TOKENS":
            self.stop_reason = "max_tokens"
        elif any(b["type"] == "tool_use" for b in self.content):
            self.stop_reason = "tool_use"
        else:
            self.stop_reason = "end_turn"
            if finish != "STOP":
                # MALFORMED_FUNCTION_CALL and friends: recorded, not hidden.
                self.stop_details = {"category": finish.lower()}

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "content": [dict(block) for block in self.content],
            "stop_reason": self.stop_reason,
            "stop_details": self.stop_details,
            "usage": self.usage.model_dump(),
            "model": self.model,
        }
        if exclude_none:
            return {k: v for k, v in payload.items() if v is not None}
        return payload


# --------------------------------------------------------------------------
# Request translation
# --------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    """Flatten a tool_result's content (a string, or a list of text blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return "" if content is None else str(content)


def _field(block: Any, name: str) -> Any:
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def translate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic ``messages`` -> Gemini ``contents``.

    Tool results carry only a ``tool_use_id``; Gemini wants the function name
    too, so it is looked up from the assistant turn that issued the call.
    """
    names: dict[str, str] = {}
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        content = message["content"]
        if isinstance(content, str):
            contents.append({"role": role, "parts": [{"text": content}]})
            continue

        parts: list[dict[str, Any]] = []
        for block in content:
            kind = _field(block, "type")
            signature = _field(block, "thought_signature")
            if kind == "text":
                part: dict[str, Any] = {"text": _field(block, "text") or ""}
            elif kind == "tool_use":
                call_id = str(_field(block, "id"))
                names[call_id] = str(_field(block, "name"))
                part = {
                    "functionCall": {
                        "name": names[call_id],
                        "args": _field(block, "input") or {},
                        "id": call_id,
                    }
                }
            elif kind == "tool_result":
                call_id = str(_field(block, "tool_use_id"))
                text = _text_of(_field(block, "content"))
                key = "error" if _field(block, "is_error") else "content"
                part = {
                    "functionResponse": {
                        "name": names.get(call_id, "unknown"),
                        "id": call_id,
                        "response": {key: text},
                    }
                }
            else:
                raise ValueError(f"unsupported content block type for Gemini: {kind!r}")
            if signature:
                part["thoughtSignature"] = signature
            parts.append(part)
        contents.append({"role": role, "parts": parts})
    return contents


def translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic tool definitions -> one Gemini ``functionDeclarations`` tool.

    ``strict`` has no Gemini equivalent and is dropped; see the module docstring
    for what that costs.
    """
    return [
        {
            "functionDeclarations": [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parametersJsonSchema": tool.get("input_schema") or {"type": "object"},
                }
                for tool in tools
            ]
        }
    ]


def translate_tool_choice(choice: dict[str, Any] | None) -> dict[str, Any] | None:
    if not choice or choice.get("type") == "auto":
        return None  # Gemini's default is AUTO
    kind = choice.get("type")
    if kind == "any":
        return {"functionCallingConfig": {"mode": "ANY"}}
    if kind == "tool":
        return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [choice["name"]]}}
    if kind == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    raise ValueError(f"unsupported tool_choice for Gemini: {choice!r}")


def build_request(**kwargs: Any) -> dict[str, Any]:
    """The ``generateContent`` body for one Anthropic-shaped ``create`` call."""
    body: dict[str, Any] = {"contents": translate_messages(kwargs["messages"])}

    system = kwargs.get("system")
    if system:
        text = system if isinstance(system, str) else "\n\n".join(b["text"] for b in system)
        body["systemInstruction"] = {"parts": [{"text": text}]}

    if kwargs.get("tools"):
        body["tools"] = translate_tools(kwargs["tools"])
    tool_config = translate_tool_choice(kwargs.get("tool_choice"))
    if tool_config is not None:
        body["toolConfig"] = tool_config

    generation: dict[str, Any] = {}
    if kwargs.get("max_tokens") is not None:
        generation["maxOutputTokens"] = kwargs["max_tokens"]
    for ours, theirs in (("temperature", "temperature"), ("top_p", "topP"), ("top_k", "topK")):
        if kwargs.get(ours) is not None:
            generation[theirs] = kwargs[ours]
    thinking = kwargs.get("thinking")
    if isinstance(thinking, dict) and thinking.get("level"):
        generation["thinkingConfig"] = {"thinkingLevel": thinking["level"]}
    if generation:
        body["generationConfig"] = generation
    # cache_control is Anthropic's explicit prompt caching. Gemini caches
    # implicitly on a shared prefix, so it is accepted and ignored.
    return body


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class _Messages:
    def __init__(self, client: GeminiClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> GeminiResponse:
        model = kwargs["model"]
        body = build_request(**kwargs)
        payload = self._client.post(f"/models/{model}:generateContent", body)
        return GeminiResponse(payload, model)


class GeminiClient:
    """Exposes ``.messages.create`` so it drops in wherever an Anthropic client goes."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = GEMINI_BASE_URL,
        timeout_s: float = 180.0,
        max_attempts: int = 4,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("a Gemini API key is required")
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            transport=transport,
            # Header, not ?key=, so the key never appears in a logged URL.
            headers={"x-goog-api-key": api_key, "content-type": "application/json"},
        )
        self._max_attempts = max_attempts
        self._sleep = sleep
        self.messages = _Messages(self)

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        last: GeminiAPIError | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._http.post(path, content=json.dumps(body))
            except httpx.TransportError as exc:
                last = GeminiAPIError(0, f"{type(exc).__name__}: {exc}")
            else:
                if response.status_code == 200:
                    result: dict[str, Any] = response.json()
                    return result
                last = GeminiAPIError(response.status_code, response.text)
                if response.status_code not in _RETRYABLE_STATUS:
                    raise last
            if attempt + 1 < self._max_attempts:
                self._sleep(min(2.0**attempt, 20.0))
        assert last is not None
        raise last

    def close(self) -> None:
        self._http.close()
