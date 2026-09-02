"""Text primitives shared by ingestion, evaluation, and the grounding verifier.

These live in one module because they encode a single decision, established in
``docs/DATA_AUDIT.md`` check 4: **raw text is the coordinate system.** CUAD's
character offsets are exact against the unmodified UTF-8 file, and whitespace
normalization shortens the string, which shifts every subsequent offset. So the
pipeline never normalizes a stored document -- it normalizes only the two strings
being compared, at the moment of comparison.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

_WHITESPACE: Final = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run to a single space and strip the ends.

    For *comparison only*. Applying this to a source document destroys the
    character offsets that index into it.
    """
    return _WHITESPACE.sub(" ", text).strip()


def contract_key(title: str) -> str:
    """Normalize a contract title for matching a JSON entry to a TXT filename.

    CUAD encodes the acute accent in ``LECLANCHE S.A.`` as a combining character
    in ``CUAD_v1.json`` and as a precomposed character on disk. Without NFC
    folding that contract silently drops out of the corpus and the count reads
    509 with no error raised anywhere.
    """
    return unicodedata.normalize("NFC", title)


class SpanStatus(StrEnum):
    """Outcome of checking one annotated span against its source document."""

    EXACT = "exact"
    """Byte-identical slice. The only status that needs no remediation."""

    WHITESPACE_ONLY = "whitespace_only"
    """Slice differs from the gold text only in whitespace."""

    OUT_OF_BOUNDS = "out_of_bounds"
    """The span runs past the end of the source document."""

    MISMATCH = "mismatch"
    """Slice is different text. Offsets have drifted against this source."""


@dataclass(frozen=True, slots=True)
class SpanCheck:
    """The result of verifying one span, with enough context to debug a failure."""

    status: SpanStatus
    char_start: int
    gold_text: str
    found_text: str
    #: Where the gold text actually appears, if it appears verbatim elsewhere.
    #: A consistent non-null value across failures means a fixed offset shift;
    #: null means the text is simply absent from this source.
    actual_offset: int | None = None

    @property
    def ok(self) -> bool:
        """Whether the offsets are usable as-is, with no repair step."""
        return self.status is SpanStatus.EXACT


def verify_span(source: str, char_start: int, gold_text: str) -> SpanCheck:
    """Check that ``source[char_start:]`` reproduces ``gold_text``.

    Returns the most specific status that applies. ``actual_offset`` is filled
    in for failures so a systematic shift can be told apart from missing text.
    """
    found = source[char_start : char_start + len(gold_text)] if char_start >= 0 else ""

    if char_start < 0 or (gold_text and char_start >= len(source)):
        status = SpanStatus.OUT_OF_BOUNDS
    elif found == gold_text:
        status = SpanStatus.EXACT
    elif _matches_ignoring_whitespace(source, char_start, gold_text):
        # Recoverable: the right text is here, but the byte offsets do not line
        # up. Deliberately not `ok` -- grading it as a pass would hide exactly
        # the drift the audit exists to measure.
        status = SpanStatus.WHITESPACE_ONLY
    elif len(found) < len(gold_text):
        # Ran off the end of the document and did not recover.
        status = SpanStatus.OUT_OF_BOUNDS
    else:
        status = SpanStatus.MISMATCH

    return SpanCheck(
        status=status,
        char_start=char_start,
        gold_text=gold_text,
        found_text=found,
        actual_offset=None if status is SpanStatus.EXACT else _find_or_none(source, gold_text),
    )


def _matches_ignoring_whitespace(source: str, char_start: int, gold_text: str) -> bool:
    """Whether the gold text starts at ``char_start`` once whitespace is collapsed.

    Length-tolerant on purpose. ``verify_span`` slices by ``len(gold_text)``, so
    a source with *extra* whitespace truncates the slice and would otherwise be
    reported as a flat mismatch -- hiding the difference between "offsets are
    slightly off" and "this text is not in the document at all". Only reached
    after the exact check fails, so the cost stays off the happy path.
    """
    return normalize_whitespace(source[char_start:]).startswith(normalize_whitespace(gold_text))


def _find_or_none(haystack: str, needle: str) -> int | None:
    if not needle:
        return None
    index = haystack.find(needle)
    return None if index < 0 else index
