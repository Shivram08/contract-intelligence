"""Content-hash response cache, so replaying an eval costs nothing.

This is the piece that makes LLM evals in CI actually viable rather than
theoretical, and the M2 measurement is what proves it: extraction costs about
$0.23 per contract, so ``CLAUDE.md`` section 6's suggested $3 nightly cap buys
roughly 13 documents. A full live run of the golden set is ~$11 per baseline.
Nobody runs that on every pull request. Replaying it from cache takes seconds
and costs zero.

**The key is a hash of everything that can change the response**: model, system
prompt, tool definitions, the full message history, and the sampling parameters.
Not the prompt alone -- a reworded tool description changes behaviour just as
much as a reworded system prompt, and hashing only the prompt would let a real
regression replay as a hit.

``cache_control`` is deliberately excluded from the key. It changes what the
request *costs*, not what it returns, so including it would invalidate the whole
cache the first time caching strategy is tuned.

**In CI the cache is read-only and a miss is a hard error.** That is the whole
safety property. A cache that silently falls through to the live API on a miss
would turn a $0 pull-request check into a surprise bill, and would need a real
API key sitting in CI to do it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

__all__ = [
    "CacheMissError",
    "CacheMode",
    "CacheStats",
    "CachingClient",
    "ResponseCache",
    "cache_key",
    "fingerprint",
]

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR: Final = REPO_ROOT / "evals" / "cache"

#: Request fields that determine the response. Anything absent from this tuple
#: is either irrelevant to the output (``cache_control``) or not used here.
KEYED_FIELDS: Final[tuple[str, ...]] = (
    "model",
    "system",
    "tools",
    "messages",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "tool_choice",
    "thinking",
    "output_config",
)


class CacheMode(StrEnum):
    """How a cache should behave on a miss."""

    READ_WRITE = "read_write"
    """Normal development: serve hits, call the API on a miss, store the result."""

    READ_ONLY = "read_only"
    """CI. A miss raises ``CacheMissError`` rather than reaching the network."""

    REFRESH = "refresh"
    """Ignore existing entries and overwrite them. For a deliberate re-baseline."""

    OFF = "off"
    """Bypass entirely. Every call is live."""


class CacheMissError(RuntimeError):
    """A required cache entry is absent.

    Raised in READ_ONLY mode. The message names the key so a developer can see
    which request needs recording, and says how to record it.
    """


def _canonical(value: Any) -> Any:
    """Reduce a request value to something stable under JSON serialization.

    SDK objects (``ToolUseBlock``, ``TextBlock``) appear in the message history
    once the loop echoes an assistant turn back, and they are not JSON
    serializable. They are converted through their own dict form where possible,
    which keeps the hash stable across SDK versions that add unrelated fields --
    ``model_dump`` on a Pydantic block omits unset optionals.
    """
    if isinstance(value, dict):
        return {key: _canonical(sub) for key, sub in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    # Pydantic v2 SDK models -- the real path for Anthropic responses.
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _canonical(dump(exclude_none=True))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonical(dataclasses.asdict(value))
    # Any other object with attributes. Reached by test doubles and by SDK
    # types that stop being Pydantic models in a future version. Falling
    # through to repr() here was a bug: it stored a string where a dict was
    # expected, so replayed responses came back with every field empty and
    # nothing raised.
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict) and attrs:
        return _canonical(attrs)
    slots = getattr(type(value), "__slots__", None)
    if slots:
        return _canonical({name: getattr(value, name, None) for name in slots})
    return repr(value)


def _serialize(request: dict[str, Any]) -> str:
    keyed = {field: _canonical(request[field]) for field in KEYED_FIELDS if field in request}
    return json.dumps(keyed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def cache_key(request: dict[str, Any]) -> str:
    """Stable hash of the parts of a request that determine its response."""
    return hashlib.sha256(_serialize(request).encode("utf-8")).hexdigest()


def fingerprint(request: dict[str, Any]) -> dict[str, Any]:
    """Per-component hashes, stored on every entry.

    Recorded so that a future replay divergence can be *diffed* rather than
    guessed at. Diagnosing the one above meant reasoning from response bodies
    because nothing recorded what the request had looked like.
    """

    def digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]

    messages = request.get("messages") or []
    return {
        "model": request.get("model"),
        "temperature": request.get("temperature"),
        "top_p": request.get("top_p"),
        "max_tokens": request.get("max_tokens"),
        "system_sha": digest(request.get("system")),
        "tools_sha": digest(request.get("tools")),
        "messages_sha": digest(messages),
        "message_count": len(messages),
        "per_message_sha": [digest(m) for m in messages],
    }


@dataclass(slots=True)
class CacheStats:
    """Hit/miss accounting for one eval run."""

    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def __str__(self) -> str:
        return (
            f"{self.hits}/{self.total} hits ({self.hit_rate:.1%}), "
            f"{self.misses} misses, {self.writes} written"
        )


@dataclass(slots=True)
class ResponseCache:
    """A directory of JSON responses keyed by content hash.

    One file per entry, sharded by the first two hex characters, rather than a
    single index file. Three reasons: a partial re-baseline rewrites only what
    changed, a cache miss shows up in ``git status`` as a specific new file, and
    concurrent eval workers never contend on one file.
    """

    directory: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    mode: CacheMode = CacheMode.READ_WRITE
    stats: CacheStats = field(default_factory=CacheStats)

    def path_for(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if self.mode in {CacheMode.OFF, CacheMode.REFRESH}:
            return None
        path = self.path_for(key)
        if not path.exists():
            return None
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        # Entries store the response under "response"; the rest is provenance.
        response = payload.get("response")
        return response if isinstance(response, dict) else None

    def put(self, key: str, response: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
        if self.mode in {CacheMode.OFF, CacheMode.READ_ONLY}:
            return
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"meta": meta or {}, "response": response}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.stats.writes += 1

    def keys(self) -> Iterator[str]:
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("*/*.json")):
            yield path.stem

    def __len__(self) -> int:
        return sum(1 for _ in self.keys())

    def size_bytes(self) -> int:
        if not self.directory.exists():
            return 0
        return sum(p.stat().st_size for p in self.directory.glob("*/*.json"))


class _CachedMessages:
    """The ``client.messages`` surface, with caching around ``create``."""

    def __init__(self, client: Any, cache: ResponseCache) -> None:
        self._client = client
        self._cache = cache

    def create(self, **kwargs: Any) -> Any:
        key = cache_key(kwargs)

        cached = self._cache.get(key)
        if cached is not None:
            self._cache.stats.hits += 1
            return _rehydrate(cached)

        self._cache.stats.misses += 1

        if self._cache.mode is CacheMode.READ_ONLY:
            raise CacheMissError(
                f"no cached response for {key[:12]} "
                f"(model={kwargs.get('model')}, {len(kwargs.get('messages', []))} messages).\n"
                "The cache is read-only, so this run will not call the API. "
                "Record it with:\n"
                "  uv run python -m evals.run_eval --refresh-cache"
            )

        if self._client is None:
            raise CacheMissError(
                f"no cached response for {key[:12]} and no live client was provided."
            )

        response = self._client.messages.create(**kwargs)
        self._cache.put(
            key,
            _dehydrate(response),
            meta=fingerprint(kwargs),
        )
        return response


@dataclass(slots=True)
class CachingClient:
    """Drop-in replacement for an Anthropic client that consults a cache first.

    The agent loop is untouched: it only ever calls
    ``client.messages.create(...)``, so swapping the client is the entire
    integration. That is deliberate -- threading a cache flag through the loop
    would put eval-harness concerns inside the extraction path.
    """

    client: Any = None
    cache: ResponseCache = field(default_factory=ResponseCache)

    def __post_init__(self) -> None:
        self.messages = _CachedMessages(self.client, self.cache)

    messages: _CachedMessages = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    @property
    def stats(self) -> CacheStats:
        return self.cache.stats


# --------------------------------------------------------------------------
# Serialization
#
# Responses are stored as plain JSON and read back as lightweight objects with
# the attributes the loop reads. Reconstructing real SDK model instances would
# couple the cache format to the installed SDK version, so a cache recorded
# today would break on the next upgrade -- exactly what a committed, replayable
# artifact must not do.
# --------------------------------------------------------------------------


def _dehydrate(response: Any) -> dict[str, Any]:
    """Convert an SDK response into plain JSON."""
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        result = dump(exclude_none=True)
        assert isinstance(result, dict)
        return result
    return {
        "content": _canonical(getattr(response, "content", [])),
        "stop_reason": getattr(response, "stop_reason", None),
        "usage": _canonical(getattr(response, "usage", None)),
    }


class _Replayed:
    """Base for objects read back from cache.

    ``model_dump`` exists so ``_canonical`` takes the *same* branch for a
    replayed object as for a live SDK model. Without it the slots fallback
    emitted every attribute including unset ones, so a replayed ``tool_use``
    block serialized with ``text: None, thinking: None`` while the live SDK
    block -- ``model_dump(exclude_none=True)`` -- omitted them.

    The consequence was subtle and total: turn 1 replayed fine, then the loop
    echoed the replayed assistant content into turn 2's request, the
    serialization differed, the key differed, and every turn after the first
    missed. Multi-turn replay was impossible, which would have left the CI gate
    unsound for both agent baselines while appearing to work.
    """

    __slots__: tuple[str, ...] = ()

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {name: getattr(self, name, None) for name in self.__slots__}
        if not exclude_none:
            return payload
        return {key: value for key, value in payload.items() if value is not None}


class _Block(dict):  # type: ignore[type-arg]
    """A content block read back from cache.

    A ``dict`` subclass with attribute access, and both halves are load-bearing.

    The loop reads ``block.type`` / ``.name`` / ``.input`` / ``.id``, so
    attribute access is required. But it also echoes the assistant turn back
    into the next request, so the same object is handed to the SDK -- and a
    custom object there raises ``TypeError: Object of type _Block is not JSON
    serializable`` the first time a cache hit is followed by a miss. Being a
    dict makes it serializable; the SDK accepts plain dicts as content blocks.

    Only non-None keys are stored, which makes ``_canonical`` (dict branch)
    produce exactly what a live block's ``model_dump(exclude_none=True)``
    produces -- so a replayed turn and a live turn hash identically.
    """

    def __init__(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            payload = {}
        # `input` is legitimately {} on a no-argument tool call and must survive;
        # only None is dropped.
        cleaned = {key: value for key, value in payload.items() if value is not None}
        if payload.get("type") == "tool_use":
            cleaned.setdefault("input", {})
        super().__init__(cleaned)

    def __getattr__(self, name: str) -> Any:
        return self.get(name)


class _Usage(_Replayed):
    """Usage read back from cache."""

    __slots__ = (
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "input_tokens",
        "output_tokens",
    )

    def __init__(self, payload: Any) -> None:
        # Tolerant of a non-dict: the cache is a committed artifact read back by
        # future code, and a malformed or older entry must degrade to zeroed
        # usage rather than crash a whole eval run.
        if not isinstance(payload, dict):
            payload = {}
        self.input_tokens = payload.get("input_tokens", 0) or 0
        self.output_tokens = payload.get("output_tokens", 0) or 0
        self.cache_read_input_tokens = payload.get("cache_read_input_tokens", 0) or 0
        self.cache_creation_input_tokens = payload.get("cache_creation_input_tokens", 0) or 0


class _Response(_Replayed):
    """A response read back from cache, quacking like the SDK's Message."""

    __slots__ = ("content", "model", "stop_details", "stop_reason", "usage")

    def __init__(self, payload: dict[str, Any]) -> None:
        raw_content = payload.get("content")
        self.content = [_Block(b) for b in raw_content] if isinstance(raw_content, list) else []
        self.stop_reason = payload.get("stop_reason")
        self.usage = _Usage(payload.get("usage"))
        self.stop_details = payload.get("stop_details")
        self.model = payload.get("model")


def _rehydrate(payload: dict[str, Any]) -> _Response:
    return _Response(payload)


def cache_from_env(directory: Path | None = None, mode: CacheMode | None = None) -> ResponseCache:
    """Build a cache, defaulting to READ_ONLY when running in CI.

    Defaulting on ``CI`` is a safety default rather than a convenience: it means
    a workflow that forgets to pass a mode cannot make live calls.
    """
    if mode is None:
        mode = CacheMode.READ_ONLY if os.environ.get("CI") else CacheMode.READ_WRITE
    return ResponseCache(directory=directory or DEFAULT_CACHE_DIR, mode=mode)
