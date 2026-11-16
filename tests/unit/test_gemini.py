"""Unit tests for the Gemini adapter. No network: an httpx MockTransport stands in.

The adapter's job is to make Gemini indistinguishable from the Anthropic client
to everything upstream. The tests that matter most are the two that broke the
Anthropic path once already: a replayed turn must hash identically to a live
one, and whatever the provider needs echoed back (here, thought signatures)
must survive the cache's JSON round trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from evals.cache import CacheMode, CachingClient, ResponseCache, cache_key

from docintel.agent.loop import compute_cost
from docintel.llm import is_gemini
from docintel.llm.gemini import (
    GeminiAPIError,
    GeminiClient,
    GeminiResponse,
    build_request,
    translate_messages,
)

MODEL = "gemini-3.5-flash-lite"
HELLO: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "submit_extraction",
        "strict": True,
        "description": "submit",
        "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
    }
]


def _call_payload(signature: str = "sig-abc") -> dict[str, Any]:
    call = {"name": "search_contract", "args": {"query": "law"}, "id": "call_1"}
    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": call, "thoughtSignature": signature}],
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1000,
            "cachedContentTokenCount": 600,
            "candidatesTokenCount": 40,
            "thoughtsTokenCount": 10,
        },
    }


class Recorder:
    """A transport that records request bodies and replies from a script."""

    def __init__(self, *replies: httpx.Response) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        self.headers.append(request.headers)
        return self.replies.pop(0)


def _client(recorder: Recorder) -> GeminiClient:
    return GeminiClient("test-key", transport=httpx.MockTransport(recorder), sleep=lambda _: None)


class TestRequestTranslation:
    def test_tools_drop_strict_and_use_json_schema(self) -> None:
        body = build_request(model=MODEL, messages=HELLO, tools=TOOLS)
        declaration = body["tools"][0]["functionDeclarations"][0]
        assert declaration["parametersJsonSchema"] == TOOLS[0]["input_schema"]
        assert "strict" not in declaration

    def test_forced_tool_becomes_any_with_allowed_name(self) -> None:
        body = build_request(
            model=MODEL,
            messages=HELLO,
            tools=TOOLS,
            tool_choice={"type": "tool", "name": "submit_extraction"},
        )
        assert body["toolConfig"] == {
            "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["submit_extraction"]}
        }

    def test_no_tool_choice_leaves_gemini_on_auto(self) -> None:
        assert "toolConfig" not in build_request(model=MODEL, messages=HELLO, tools=TOOLS)

    def test_system_blocks_become_system_instruction(self) -> None:
        system = [{"type": "text", "text": "be precise", "cache_control": {"type": "ephemeral"}}]
        body = build_request(model=MODEL, messages=HELLO, system=system)
        assert body["systemInstruction"] == {"parts": [{"text": "be precise"}]}

    def test_thinking_is_only_sent_when_requested(self) -> None:
        plain = build_request(model=MODEL, messages=HELLO, max_tokens=10)
        assert "thinkingConfig" not in plain["generationConfig"]
        tuned = build_request(model=MODEL, messages=HELLO, thinking={"level": "low"})
        assert tuned["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}

    def test_echoed_call_keeps_its_signature_and_result_gets_its_name(self) -> None:
        tool_use = {
            "type": "tool_use",
            "id": "call_1",
            "name": "search_contract",
            "input": {"query": "law"},
            "thought_signature": "sig-abc",
        }
        result = {"type": "tool_result", "tool_use_id": "call_1", "content": "found"}
        contents = translate_messages(
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [tool_use]},
                {"role": "user", "content": [result]},
            ]
        )
        model_part = contents[1]["parts"][0]
        assert contents[1]["role"] == "model"
        assert model_part["thoughtSignature"] == "sig-abc"
        assert model_part["functionCall"] == {
            "name": "search_contract",
            "args": {"query": "law"},
            "id": "call_1",
        }
        assert contents[2]["parts"][0]["functionResponse"] == {
            "name": "search_contract",
            "id": "call_1",
            "response": {"content": "found"},
        }

    def test_error_results_are_marked_as_errors(self) -> None:
        tool_use = {"type": "tool_use", "id": "c", "name": "t", "input": {}}
        result = {"type": "tool_result", "tool_use_id": "c", "is_error": True, "content": "No"}
        contents = translate_messages(
            [{"role": "assistant", "content": [tool_use]}, {"role": "user", "content": [result]}]
        )
        assert contents[1]["parts"][0]["functionResponse"]["response"] == {"error": "No"}


class TestResponseMapping:
    def test_function_call_becomes_tool_use_with_signature(self) -> None:
        response = GeminiResponse(_call_payload(), MODEL)
        (block,) = response.content
        assert (block.type, block.id, block.name, block.input) == (
            "tool_use",
            "call_1",
            "search_contract",
            {"query": "law"},
        )
        assert block.thought_signature == "sig-abc"
        assert response.stop_reason == "tool_use"

    def test_cached_tokens_split_out_and_thinking_counted_as_output(self) -> None:
        usage = GeminiResponse(_call_payload(), MODEL).usage
        assert usage.input_tokens == 400
        assert usage.cache_read_input_tokens == 600
        assert usage.output_tokens == 50
        assert usage.thinking_tokens == 10

    def test_cost_uses_the_cache_rate_for_cached_tokens(self) -> None:
        usage = GeminiResponse(_call_payload(), MODEL).usage
        # 400 x $0.30 + 600 x $0.03 + 50 x $2.50, per 1M
        assert compute_cost(MODEL, usage) == pytest.approx((120 + 18 + 125) / 1_000_000)

    def test_safety_block_is_a_refusal(self) -> None:
        payload = {"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}
        response = GeminiResponse(payload, MODEL)
        assert response.stop_reason == "refusal"
        assert response.stop_details == {"category": "safety"}

    def test_blocked_prompt_with_no_candidates_is_a_refusal(self) -> None:
        response = GeminiResponse({"promptFeedback": {"blockReason": "OTHER"}}, MODEL)
        assert response.stop_reason == "refusal"

    def test_malformed_call_is_recorded_not_hidden(self) -> None:
        payload = {"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL", "content": {}}]}
        response = GeminiResponse(payload, MODEL)
        assert response.content == []
        assert response.stop_reason == "end_turn"
        assert response.stop_details == {"category": "malformed_function_call"}


class TestTransport:
    def test_key_travels_in_a_header_not_the_url(self) -> None:
        recorder = Recorder(httpx.Response(200, json=_call_payload()))
        _client(recorder).messages.create(model=MODEL, messages=HELLO)
        assert recorder.headers[0]["x-goog-api-key"] == "test-key"

    def test_retries_a_rate_limit_then_succeeds(self) -> None:
        recorder = Recorder(
            httpx.Response(429, text="slow down"), httpx.Response(200, json=_call_payload())
        )
        response = _client(recorder).messages.create(model=MODEL, messages=HELLO)
        assert response.stop_reason == "tool_use"
        assert len(recorder.bodies) == 2

    def test_a_bad_request_is_not_retried(self) -> None:
        recorder = Recorder(httpx.Response(400, text="missing thought_signature"))
        with pytest.raises(GeminiAPIError, match="400"):
            _client(recorder).messages.create(model=MODEL, messages=HELLO)
        assert len(recorder.bodies) == 1


class TestCacheRoundTrip:
    def test_replayed_turn_hashes_like_the_live_turn(self, tmp_path: Path) -> None:
        """The multi-turn replay bug, re-asserted for this provider."""
        recorder = Recorder(httpx.Response(200, json=_call_payload()))
        cache = ResponseCache(directory=tmp_path, mode=CacheMode.READ_WRITE)
        live = CachingClient(client=_client(recorder), cache=cache)
        request: dict[str, Any] = {"model": MODEL, "messages": [{"role": "user", "content": "go"}]}

        live_content = live.messages.create(**request).content
        replay = CachingClient(
            client=None, cache=ResponseCache(directory=tmp_path, mode=CacheMode.READ_ONLY)
        )
        replay_content = replay.messages.create(**request).content

        def next_turn(content: Any) -> dict[str, Any]:
            history = [*request["messages"], {"role": "assistant", "content": content}]
            return {**request, "messages": history}

        assert cache_key(next_turn(live_content)) == cache_key(next_turn(replay_content))
        assert replay_content[0].thought_signature == "sig-abc"

    def test_live_cost_is_tallied_and_replay_is_free(self, tmp_path: Path) -> None:
        recorder = Recorder(httpx.Response(200, json=_call_payload()))
        cache = ResponseCache(directory=tmp_path, mode=CacheMode.READ_WRITE)
        client = CachingClient(client=_client(recorder), cache=cache)
        request: dict[str, Any] = {"model": MODEL, "messages": [{"role": "user", "content": "go"}]}
        client.messages.create(**request)
        client.messages.create(**request)
        assert cache.stats.live_calls == 1
        assert cache.stats.live_cost_usd == pytest.approx(263 / 1_000_000)


def test_provider_follows_from_model_name() -> None:
    assert is_gemini("gemini-3.5-flash-lite")
    assert not is_gemini("claude-sonnet-5")
