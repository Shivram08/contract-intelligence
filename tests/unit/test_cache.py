"""Unit tests for the response cache.

Two properties matter, and they pull in opposite directions:

**Nothing that changes the response may be missing from the key.** If it is, a
real regression replays as a cache hit and the eval reports the old number --
the worst possible failure for a regression gate, because it is silent and it
looks like a pass.

**Nothing that only changes the cost may be in the key.** `cache_control` is
the example: including it would invalidate every entry the first time caching
strategy is tuned, turning a free CI check into a full re-baseline.

Replay fidelity gets its own section because it broke once. `_canonical` fell
through to `repr()` for anything that was not a Pydantic model, so responses
round-tripped with every field empty and nothing raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from evals.cache import (
    KEYED_FIELDS,
    CacheMissError,
    CacheMode,
    CacheStats,
    CachingClient,
    ResponseCache,
    cache_key,
)

BASE_REQUEST: dict[str, Any] = {
    "model": "claude-sonnet-5",
    "system": [{"type": "text", "text": "you are a contract analyst"}],
    "tools": [{"name": "search_contract", "description": "search"}],
    "messages": [{"role": "user", "content": "extract clauses"}],
    "max_tokens": 8000,
}


# --- doubles -----------------------------------------------------------------


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 20
    cache_read_input_tokens: int = 50
    cache_creation_input_tokens: int = 10


@dataclass
class FakeBlock:
    type: str = "tool_use"
    name: str = "get_schema"
    id: str = "toolu_1"
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeResponse:
    content: list[Any] = field(default_factory=list)
    stop_reason: str = "tool_use"
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "claude-sonnet-5"


class CountingClient:
    """Records how many live calls actually happened."""

    def __init__(self, response: Any = None) -> None:
        self.calls = 0
        self._response = response or FakeResponse(content=[FakeBlock()])
        outer = self

        class _Messages:
            def create(self, **kwargs: Any) -> Any:
                outer.calls += 1
                return outer._response

        self.messages = _Messages()


def cache_at(path: Path, mode: CacheMode = CacheMode.READ_WRITE) -> ResponseCache:
    return ResponseCache(directory=path, mode=mode)


# --- the key -----------------------------------------------------------------


class TestKeyIncludesEverythingThatChangesTheResponse:
    @pytest.mark.parametrize(
        ("field_name", "new_value"),
        [
            ("model", "claude-opus-5"),
            ("system", [{"type": "text", "text": "different prompt"}]),
            ("tools", [{"name": "search_contract", "description": "reworded"}]),
            ("messages", [{"role": "user", "content": "a different question"}]),
            ("max_tokens", 4000),
        ],
    )
    def test_changing_it_changes_the_key(self, field_name: str, new_value: Any) -> None:
        assert cache_key(BASE_REQUEST) != cache_key({**BASE_REQUEST, field_name: new_value})

    def test_a_reworded_tool_description_changes_the_key(self) -> None:
        """Behaviour depends on tool descriptions as much as on the prompt.
        Hashing only the prompt would let that regression replay as a hit."""
        reworded = [{"name": "search_contract", "description": "COMPLETELY DIFFERENT"}]
        assert cache_key(BASE_REQUEST) != cache_key({**BASE_REQUEST, "tools": reworded})

    def test_adding_a_message_changes_the_key(self) -> None:
        longer = [*BASE_REQUEST["messages"], {"role": "assistant", "content": "ok"}]
        assert cache_key(BASE_REQUEST) != cache_key({**BASE_REQUEST, "messages": longer})

    def test_every_keyed_field_is_actually_consulted(self) -> None:
        """Guards the list itself: a field in KEYED_FIELDS that the hash ignores
        is a false promise."""
        for name in KEYED_FIELDS:
            with_field = {**BASE_REQUEST, name: "sentinel-value-xyz"}
            without = {k: v for k, v in with_field.items() if k != name}
            assert cache_key(with_field) != cache_key(without), name


class TestKeyExcludesCostOnlyFields:
    def test_cache_control_is_not_in_the_key(self) -> None:
        """It changes what the request costs, not what it returns."""
        with_cc = {**BASE_REQUEST, "cache_control": {"type": "ephemeral"}}
        assert cache_key(BASE_REQUEST) == cache_key(with_cc)

    def test_unknown_extra_fields_are_ignored(self) -> None:
        assert cache_key(BASE_REQUEST) == cache_key({**BASE_REQUEST, "metadata": {"x": 1}})


class TestKeyStability:
    def test_same_request_gives_the_same_key(self) -> None:
        assert cache_key(BASE_REQUEST) == cache_key(dict(BASE_REQUEST))

    def test_dict_ordering_does_not_matter(self) -> None:
        reordered = dict(reversed(list(BASE_REQUEST.items())))
        assert cache_key(BASE_REQUEST) == cache_key(reordered)

    def test_nested_dict_ordering_does_not_matter(self) -> None:
        a = {**BASE_REQUEST, "tools": [{"name": "t", "description": "d"}]}
        b = {**BASE_REQUEST, "tools": [{"description": "d", "name": "t"}]}
        assert cache_key(a) == cache_key(b)

    def test_key_is_a_hex_sha256(self) -> None:
        key = cache_key(BASE_REQUEST)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# --- behaviour ---------------------------------------------------------------


class TestReadWrite:
    def test_second_identical_request_does_not_reach_the_client(self, tmp_path: Path) -> None:
        client = CountingClient()
        c = CachingClient(client=client, cache=cache_at(tmp_path))
        c.messages.create(**BASE_REQUEST)
        c.messages.create(**BASE_REQUEST)
        assert client.calls == 1

    def test_a_different_request_does_reach_the_client(self, tmp_path: Path) -> None:
        client = CountingClient()
        c = CachingClient(client=client, cache=cache_at(tmp_path))
        c.messages.create(**BASE_REQUEST)
        c.messages.create(**{**BASE_REQUEST, "max_tokens": 1})
        assert client.calls == 2

    def test_stats_track_hits_and_misses(self, tmp_path: Path) -> None:
        cache = cache_at(tmp_path)
        c = CachingClient(client=CountingClient(), cache=cache)
        c.messages.create(**BASE_REQUEST)
        c.messages.create(**BASE_REQUEST)
        assert (cache.stats.hits, cache.stats.misses, cache.stats.writes) == (1, 1, 1)

    def test_entries_persist_across_cache_instances(self, tmp_path: Path) -> None:
        CachingClient(client=CountingClient(), cache=cache_at(tmp_path)).messages.create(
            **BASE_REQUEST
        )
        second = CountingClient()
        CachingClient(client=second, cache=cache_at(tmp_path)).messages.create(**BASE_REQUEST)
        assert second.calls == 0

    def test_entries_are_sharded_by_key_prefix(self, tmp_path: Path) -> None:
        """One file per entry, so a partial re-baseline rewrites only what
        changed and a miss appears in git status as a specific new file."""
        CachingClient(client=CountingClient(), cache=cache_at(tmp_path)).messages.create(
            **BASE_REQUEST
        )
        key = cache_key(BASE_REQUEST)
        assert (tmp_path / key[:2] / f"{key}.json").exists()


class TestReadOnlyIsTheCiSafetyProperty:
    def test_a_hit_is_served(self, tmp_path: Path) -> None:
        CachingClient(client=CountingClient(), cache=cache_at(tmp_path)).messages.create(
            **BASE_REQUEST
        )
        readonly = CachingClient(client=None, cache=cache_at(tmp_path, CacheMode.READ_ONLY))
        assert readonly.messages.create(**BASE_REQUEST).stop_reason == "tool_use"

    def test_a_miss_raises_rather_than_calling_the_api(self, tmp_path: Path) -> None:
        """The whole point. Falling through to the live API would turn a $0
        pull-request check into a surprise bill."""
        client = CountingClient()
        readonly = CachingClient(client=client, cache=cache_at(tmp_path, CacheMode.READ_ONLY))
        with pytest.raises(CacheMissError):
            readonly.messages.create(**BASE_REQUEST)
        assert client.calls == 0

    def test_the_error_says_how_to_record_the_entry(self, tmp_path: Path) -> None:
        readonly = CachingClient(client=None, cache=cache_at(tmp_path, CacheMode.READ_ONLY))
        with pytest.raises(CacheMissError, match="refresh-cache"):
            readonly.messages.create(**BASE_REQUEST)

    def test_read_only_never_writes(self, tmp_path: Path) -> None:
        cache = cache_at(tmp_path, CacheMode.READ_ONLY)
        cache.put("a" * 64, {"stop_reason": "end_turn"})
        assert len(cache) == 0


class TestRefreshAndOff:
    def test_refresh_ignores_an_existing_entry_and_overwrites(self, tmp_path: Path) -> None:
        CachingClient(client=CountingClient(), cache=cache_at(tmp_path)).messages.create(
            **BASE_REQUEST
        )
        client = CountingClient()
        CachingClient(client=client, cache=cache_at(tmp_path, CacheMode.REFRESH)).messages.create(
            **BASE_REQUEST
        )
        assert client.calls == 1

    def test_off_never_reads_or_writes(self, tmp_path: Path) -> None:
        client = CountingClient()
        cache = cache_at(tmp_path, CacheMode.OFF)
        c = CachingClient(client=client, cache=cache)
        c.messages.create(**BASE_REQUEST)
        c.messages.create(**BASE_REQUEST)
        assert client.calls == 2
        assert len(cache) == 0


class TestReplayFidelity:
    """This broke once: `_canonical` fell through to repr() for anything that
    was not a Pydantic model, so replayed responses had every field empty."""

    def test_tool_use_block_survives_the_round_trip(self, tmp_path: Path) -> None:
        client = CountingClient(
            FakeResponse(content=[FakeBlock(name="submit_extraction", id="toolu_9")])
        )
        c = CachingClient(client=client, cache=cache_at(tmp_path))
        c.messages.create(**BASE_REQUEST)
        replayed = c.messages.create(**BASE_REQUEST)
        block = replayed.content[0]
        assert (block.type, block.name, block.id) == ("tool_use", "submit_extraction", "toolu_9")

    def test_tool_input_survives(self, tmp_path: Path) -> None:
        client = CountingClient(
            FakeResponse(content=[FakeBlock(input={"query": "governing law", "top_k": 5})])
        )
        c = CachingClient(client=client, cache=cache_at(tmp_path))
        c.messages.create(**BASE_REQUEST)
        assert c.messages.create(**BASE_REQUEST).content[0].input == {
            "query": "governing law",
            "top_k": 5,
        }

    def test_usage_survives_so_replayed_cost_is_still_reported(self, tmp_path: Path) -> None:
        """A replay that loses usage would report $0 per document, which is
        wrong in a way that looks like a win."""
        c = CachingClient(client=CountingClient(), cache=cache_at(tmp_path))
        c.messages.create(**BASE_REQUEST)
        usage = c.messages.create(**BASE_REQUEST).usage
        assert usage.input_tokens == 100
        assert usage.output_tokens == 20
        assert usage.cache_read_input_tokens == 50

    def test_stop_reason_survives(self, tmp_path: Path) -> None:
        client = CountingClient(FakeResponse(stop_reason="end_turn"))
        c = CachingClient(client=client, cache=cache_at(tmp_path))
        c.messages.create(**BASE_REQUEST)
        assert c.messages.create(**BASE_REQUEST).stop_reason == "end_turn"

    def test_multiple_blocks_survive_in_order(self, tmp_path: Path) -> None:
        client = CountingClient(
            FakeResponse(content=[FakeBlock(id="a"), FakeBlock(id="b"), FakeBlock(id="c")])
        )
        c = CachingClient(client=client, cache=cache_at(tmp_path))
        c.messages.create(**BASE_REQUEST)
        assert [b.id for b in c.messages.create(**BASE_REQUEST).content] == ["a", "b", "c"]

    def test_a_malformed_entry_degrades_rather_than_crashing(self, tmp_path: Path) -> None:
        """The cache is a committed artifact read by future code."""
        cache = cache_at(tmp_path)
        key = cache_key(BASE_REQUEST)
        cache.put(key, {"content": "not-a-list", "usage": "not-a-dict"})
        replayed = CachingClient(client=None, cache=cache).messages.create(**BASE_REQUEST)
        assert replayed.content == []
        assert replayed.usage.input_tokens == 0


class TestStats:
    def test_hit_rate_of_an_empty_run_is_zero_not_a_crash(self) -> None:
        assert CacheStats().hit_rate == 0.0

    def test_hit_rate(self) -> None:
        assert CacheStats(hits=3, misses=1).hit_rate == pytest.approx(0.75)

    def test_str_is_readable(self) -> None:
        assert "3/4 hits" in str(CacheStats(hits=3, misses=1, writes=1))
