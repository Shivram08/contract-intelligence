"""Tool schemas and implementations for the extraction agent.

Four tools, and the split between them is deliberate:

``search_contract`` and ``read_span`` are the agent's eyes. Retrieval returns
passages; ``read_span`` widens the window when a clause is cut off or when the
surrounding context decides whether a phrase is operative or a carve-out. Giving
the agent a read tool rather than only search is what lets it distinguish
"liability is capped" from "liability is capped, except that..." -- a distinction
retrieval alone cannot make because both live in the same chunk.

``get_schema`` returns the clause definitions verbatim from CUAD. Putting them in
a tool rather than the system prompt keeps the cached prefix small and stable,
and lets the agent pull only the definitions it needs.

There is deliberately **no** ``validate_extraction`` dry-run tool. An earlier
version had one, and it was the most expensive thing in the loop: the agent
wrote the whole 12-clause payload to validate it, again after fixing, and on one
measured contract five times over, at 3-4k output tokens a time. Asking it in the
prompt to validate at most once did not change that. Grounding and the rules now
run once after submission, in ``extract.py`` -- which also keeps the grounding
violation rate honest, since a rejected-and-retried submission would report zero
violations by construction. See ``accept_submission``.

``submit_extraction`` is the terminal tool. Its schema is ``strict: true``, which
makes the API validate the structure before it ever reaches Pydantic -- so
malformed output is rejected at the boundary rather than parsed hopefully.

All tool inputs arrive as dicts and are re-validated with Pydantic anyway: the
API's strict mode guarantees shape, not semantics, and nothing here trusts the
model's arithmetic about offsets.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from docintel.schemas import (
    BOOLEAN_CLAUSES,
    CLAUSE_DEFINITIONS,
    CLAUSE_TIERS,
    ClauseExtraction,
    ClauseType,
    Document,
    Evidence,
    RetrievalHit,
)
from docintel.validation.rules import JurisdictionIndex, load_jurisdictions

__all__ = [
    "TOOL_NAMES",
    "ToolContext",
    "ToolError",
    "accept_submission",
    "build_tool_schemas",
    "execute_tool",
    "parse_submission",
    "validate_submission",
]


class ToolError(Exception):
    """A tool could not run. Returned to the model as an error result."""


#: The widest span `read_span` will return. A model asking for 200k characters
#: has lost the plot, and honouring it would blow the context budget in one call.
MAX_READ_CHARS: Final = 8_000

#: Context added either side of a requested span, so a clause cut off at a chunk
#: boundary can be recovered without a second call.
READ_PADDING: Final = 400

TOOL_NAMES: Final[tuple[str, ...]] = (
    "search_contract",
    "read_span",
    "get_schema",
    "submit_extraction",
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


def _clause_type_enum() -> list[str]:
    return [clause.value for clause in ClauseType]


def _evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "quote": {
                "type": "string",
                "description": (
                    "Text copied verbatim from the contract. Not tidied, not "
                    "re-wrapped, not corrected."
                ),
            },
            "char_start": {
                "type": "integer",
                "description": "Start offset, from the search or read result. Must be >= 0.",
            },
            "char_end": {
                "type": "integer",
                "description": "End offset. Must be greater than char_start.",
            },
        },
        "required": ["quote", "char_start", "char_end"],
        "additionalProperties": False,
    }


def _clause_schema() -> dict[str, Any]:
    """Schema for one clause extraction.

    Deliberately free of ``minimum``/``maximum`` and of type arrays, because
    this schema is reused by ``submit_extraction``, which runs with
    ``strict: true``. Strict mode does not support numerical or string
    constraints, and it takes single types or ``anyOf`` rather than
    ``{"type": ["string", "null"]}``. The SDK strips unsupported keywords only
    on the ``parse()`` / ``output_config`` path -- a raw tool dict is sent as
    written, so an unsupported keyword here is a 400 on every request.

    The bounds are not lost: ``Evidence`` and ``ClauseExtraction`` enforce them
    with ``Field(ge=...)``, and a violation becomes a specific complaint handed
    back to the model rather than a rejected request.
    """
    return {
        "type": "object",
        "properties": {
            "clause_type": {"type": "string", "enum": _clause_type_enum()},
            "present": {
                "type": "boolean",
                "description": "Whether the contract contains this clause as defined.",
            },
            "value": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": (
                    "Normalized value, or null. Always null for the six presence-only clause types."
                ),
            },
            "raw_text": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "The clause text, verbatim. Never a paraphrase.",
            },
            "evidence": {
                "type": "array",
                "items": _evidence_schema(),
                "description": "At least one span when present is true; empty when false.",
            },
            "confidence": {
                "type": "number",
                "description": (
                    "Probability this verdict is correct, between 0 and 1. Vary "
                    "it; a constant value across clauses is flagged as a defect."
                ),
            },
        },
        "required": ["clause_type", "present", "value", "raw_text", "evidence", "confidence"],
        "additionalProperties": False,
    }


def build_tool_schemas() -> list[dict[str, Any]]:
    """The tool definitions sent to the API.

    Order is fixed and the content is deterministic, because tools are rendered
    before ``system`` and ``messages`` in the cached prefix -- a tool list that
    varies between calls silently invalidates the prompt cache for everything
    after it.
    """
    return [
        {
            "name": "search_contract",
            "description": (
                "Hybrid lexical + dense search over this contract's chunks. "
                "Phrase the query as the contract would phrase the clause, not "
                "as the clause is named. Returns passages with their character "
                "offsets."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "How many passages to return. Default 5.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "read_span",
            "description": (
                "Read a character range of the contract directly, with a little "
                "context either side. Use when a search result is cut off "
                "mid-clause, or to check whether a phrase sits inside a "
                "definition or a carve-out."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "char_start": {"type": "integer", "minimum": 0},
                    "char_end": {"type": "integer", "minimum": 0},
                },
                "required": ["char_start", "char_end"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_schema",
            "description": (
                "The official definition, tier, and expected value format for "
                "clause types. Omit clause_types to get all 12."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "clause_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": _clause_type_enum()},
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "name": "submit_extraction",
            # Strict mode: the API validates this shape before the SDK returns,
            # so a malformed submission fails at the boundary instead of being
            # parsed hopefully on our side.
            "strict": True,
            "description": (
                "Submit the final extraction for all 12 clause types. Call "
                "exactly once, after validate_extraction comes back clean."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "clauses": {
                        "type": "array",
                        "items": _clause_schema(),
                        "description": "One entry per clause type, all 12.",
                    }
                },
                "required": ["clauses"],
                "additionalProperties": False,
            },
        },
    ]


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ToolContext:
    """Everything the tools are allowed to touch.

    ``search`` is injected as a callable rather than a retriever object so the
    agent loop can be tested without a database, and so an ablation can swap in
    a single-arm or no-rerank retriever without touching this module.
    """

    document: Document
    search: Any = None
    jurisdictions: JurisdictionIndex = field(default_factory=load_jurisdictions)
    #: Populated as the agent calls tools, for tracing and for the eval harness.
    calls: list[dict[str, Any]] = field(default_factory=list)


def _format_hits(hits: Sequence[RetrievalHit]) -> str:
    if not hits:
        return "No passages matched. Try different wording, or the clause may be absent."
    lines: list[str] = []
    for position, hit in enumerate(hits, start=1):
        heading = f" | {hit.chunk.heading}" if hit.chunk.heading else ""
        lines.append(
            f"[{position}] chars {hit.chunk.char_start}-{hit.chunk.char_end}{heading}\n"
            f"{hit.chunk.text}"
        )
    return "\n\n".join(lines)


def _tool_search_contract(ctx: ToolContext, args: dict[str, Any]) -> str:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query must be a non-empty string")
    if ctx.search is None:
        raise ToolError("search is unavailable in this context")

    top_k = args.get("top_k", 5)
    if not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ToolError("top_k must be an integer between 1 and 20")

    hits = ctx.search(query, top_k)
    return _format_hits(hits)


def _tool_read_span(ctx: ToolContext, args: dict[str, Any]) -> str:
    start, end = args.get("char_start"), args.get("char_end")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ToolError("char_start and char_end must be integers")
    if end <= start:
        raise ToolError(f"char_end ({end}) must exceed char_start ({start})")

    length = len(ctx.document.text)
    if start >= length:
        raise ToolError(f"char_start {start} is past the end of a {length}-character document")
    if end - start > MAX_READ_CHARS:
        raise ToolError(
            f"requested {end - start} characters; the maximum is {MAX_READ_CHARS}. "
            "Narrow the range or make several calls."
        )

    padded_start = max(0, start - READ_PADDING)
    padded_end = min(length, end + READ_PADDING)
    return (
        f"chars {padded_start}-{padded_end} of {length}"
        f" (you asked for {start}-{end}; context added either side)\n\n"
        f"{ctx.document.text[padded_start:padded_end]}"
    )


def _tool_get_schema(ctx: ToolContext, args: dict[str, Any]) -> str:
    requested = args.get("clause_types") or _clause_type_enum()
    unknown = [name for name in requested if name not in set(_clause_type_enum())]
    if unknown:
        raise ToolError(f"unknown clause types: {unknown}")

    value_formats = {
        ClauseType.GOVERNING_LAW: "jurisdiction id, e.g. US-NY (see ids below)",
        ClauseType.PARTIES: "Name (Role); Name (Role)",
        ClauseType.EFFECTIVE_DATE: "ISO 8601 prefix: YYYY, YYYY-MM, or YYYY-MM-DD",
        ClauseType.EXPIRATION_DATE: "ISO 8601 prefix, or PERPETUAL",
        ClauseType.RENEWAL_TERM: "days, plus 'recurring' if it rolls over",
        ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL: "integer days",
    }

    blocks: list[str] = []
    for name in requested:
        clause = ClauseType(name)
        # CUAD's own definition. Read from CLAUSE_DEFINITIONS, never from
        # `clause.__doc__` -- enum members inherit the class docstring, so that
        # would hand the agent the same meta-text for all twelve clauses.
        definition = CLAUSE_DEFINITIONS[clause]
        value_format = (
            "null (presence-only clause)"
            if clause in BOOLEAN_CLAUSES
            else value_formats.get(clause, "string")
        )
        blocks.append(
            f"## {clause.value}\n"
            f"tier: {CLAUSE_TIERS[clause].value}\n"
            f"definition: {definition}\n"
            f"value: {value_format}"
        )

    if ClauseType.GOVERNING_LAW.value in requested:
        ids = sorted(ctx.jurisdictions.all_ids)
        blocks.append(f"## valid governing_law ids ({len(ids)})\n{', '.join(ids)}")

    return "\n\n".join(blocks)


def _coerce_clauses(raw: Any) -> list[ClauseExtraction]:
    """Turn the model's JSON into validated ClauseExtraction objects.

    Raises ToolError with the Pydantic message on failure, so the agent sees a
    specific, actionable complaint rather than a generic parse error.
    """
    if not isinstance(raw, list):
        raise ToolError("clauses must be an array")

    clauses: list[ClauseExtraction] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ToolError(f"clauses[{index}] must be an object")
        try:
            evidence = [
                Evidence(
                    quote=str(span["quote"]),
                    char_start=int(span["char_start"]),
                    char_end=int(span["char_end"]),
                )
                for span in item.get("evidence") or []
            ]
            clauses.append(
                ClauseExtraction(
                    clause_type=ClauseType(item["clause_type"]),
                    present=bool(item["present"]),
                    value=item.get("value"),
                    raw_text=item.get("raw_text"),
                    evidence=evidence,
                    confidence=float(item["confidence"]),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ToolError(f"clauses[{index}] is invalid: {exc}") from exc
    return clauses


def validate_submission(args: dict[str, Any]) -> list[ClauseExtraction]:
    """Parse and completeness-check a ``submit_extraction`` payload.

    The single definition of "a valid submission", called by both the tool
    handler and the agent loop. An earlier version had the loop call
    ``parse_submission`` directly while the completeness check lived only in the
    tool handler -- so a five-clause submission was accepted by the loop and
    rejected by the tool, depending on which path ran. That is the same class of
    drift this module's docstring warns about for ``validate_extraction``.
    """
    clauses = _coerce_clauses(args.get("clauses"))
    missing = sorted(set(ClauseType) - {clause.clause_type for clause in clauses})
    if missing:
        raise ToolError(
            f"missing {len(missing)} clause type(s): {[m.value for m in missing]}. "
            f"Submit all {len(ClauseType)}."
        )
    return clauses


def accept_submission(ctx: ToolContext, args: dict[str, Any]) -> list[ClauseExtraction]:
    """Acceptance check for a submission: structure and completeness only.

    **Grounding and the deterministic rules are deliberately NOT checked here**,
    and that is a measurement decision rather than an oversight.

    Rejecting an ungrounded submission and letting the agent retry would make
    the pipeline self-healing, which sounds strictly better. It is not: it
    destroys the grounding violation rate as a metric. If every submission
    carrying a fabricated quote is bounced back until it complies, the final
    output shows zero violations *by construction*, and the number being
    reported is "did the agent eventually comply" rather than "how often does
    this model fabricate". ``CLAUDE.md`` section 6 wants the latter as a
    first-class figure, and section 13 sets a threshold on it.

    So grounding and the rules run once, afterwards, in ``extract.py``, where
    their findings become part of the recorded result and route the case to
    review. The agent gets one attempt and its raw behaviour is what gets
    measured.

    What this replaced: a separate ``validate_extraction`` dry-run tool. That
    was the most expensive thing in the loop -- the agent wrote the entire
    12-clause payload to validate, again after fixing, and on one measured
    contract five times in total, at 3-4k output tokens each. Instructing it to
    "validate at most once" in the prompt did not work. Removing the tool makes
    the happy path write the payload exactly once, mechanically.
    """
    return validate_submission(args)


def _tool_submit_extraction(ctx: ToolContext, args: dict[str, Any]) -> str:
    clauses = accept_submission(ctx, args)
    return f"Accepted {len(clauses)} clause extractions."


_HANDLERS: Final = {
    "search_contract": _tool_search_contract,
    "read_span": _tool_read_span,
    "get_schema": _tool_get_schema,
    "submit_extraction": _tool_submit_extraction,
}


def execute_tool(ctx: ToolContext, name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Run one tool. Returns ``(result_text, is_error)``.

    Errors come back as tool results rather than raised exceptions, because a
    model that mistypes an argument should get a specific complaint and another
    turn -- killing the extraction over a bad ``top_k`` wastes everything spent
    so far.
    """
    ctx.calls.append({"tool": name, "input": args})
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)}", True
    try:
        return handler(ctx, args), False
    except ToolError as exc:
        return f"Error: {exc}", True
    except Exception as exc:
        # Deliberately broad: a bug in one tool must not abort an extraction
        # that has already spent real tokens. The model gets the error and retries.
        return f"Error: {type(exc).__name__}: {exc}", True


def parse_submission(args: dict[str, Any]) -> list[ClauseExtraction]:
    """Validate a ``submit_extraction`` payload into domain objects."""
    return _coerce_clauses(args.get("clauses"))


def tool_schema_json() -> str:
    """Deterministic serialization of the tool list, for the prompt hash."""
    return json.dumps(build_tool_schemas(), sort_keys=True, separators=(",", ":"))
