"""Structure-aware chunking for contracts.

Fixed-size windows cut through the middle of clauses, which is precisely wrong
for a system whose job is to return a clause as evidence. This chunker splits on
section headings first and falls back to token counts only where it has to.

Three facts from ``docs/DATA_AUDIT.md`` drive the design:

1. **Headings are usually not at the start of a line.** 59% of CUAD contracts are
   run-on text with a mean line length of 340 characters (p90 1,453); section
   numbers appear mid-line after runs of spaces. A ``^``-anchored regex finds
   almost nothing on the majority of the corpus.
2. **Some contracts have no headings at all.** 57 of 510 have zero numbered or
   article headings and 108 have fewer than five, so the token-count fallback is a
   primary path rather than an edge case.
3. **Offsets index into raw text.** Chunks are pure slices -- never stripped,
   reflowed, or normalized -- so an evidence quote traces back to an exact
   character range in the source file.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np

from docintel.schemas import Chunk, Document

__all__ = [
    "ChunkingConfig",
    "Heading",
    "TiktokenCounter",
    "TokenCounter",
    "TokenIndex",
    "WordTokenCounter",
    "chunk_document",
    "find_headings",
]


class TokenIndex:
    """Character offsets of every token start in one document.

    Chunking asks "how many tokens are in characters [a, b)?" thousands of times
    per document, at overlapping and growing ranges. Answering that by slicing
    the string and re-tokenizing is quadratic: the first version of this chunker
    took 74 minutes on CUAD's 510 contracts, almost all of it re-encoding the
    same text. Tokenizing once and answering range queries by binary search
    makes each query O(log n) and the whole corpus a couple of minutes.

    One caveat, measured rather than assumed: a chunk's tokens counted this way
    can differ slightly from re-tokenizing the chunk's text on its own, because
    BPE merges across boundaries that a slice removes. Over the whole corpus this
    puts 0.6% of chunks above ``max_tokens``, by a median of 1 token and at most
    3 -- against a 512-token budget. That is accepted rather than engineered
    around: eliminating it means re-tokenizing every candidate boundary, which is
    the quadratic behaviour this class exists to remove. Anything sizing a real
    context window should carry more headroom than 3 tokens anyway.
    """

    __slots__ = ("_starts",)

    def __init__(self, token_starts: Sequence[int]) -> None:
        self._starts = list(token_starts)

    def count_range(self, start: int, end: int) -> int:
        """Tokens beginning within the half-open character range [start, end)."""
        if end <= start:
            return 0
        return bisect_left(self._starts, end) - bisect_left(self._starts, start)

    def __len__(self) -> int:
        return len(self._starts)


class TokenCounter(Protocol):
    """Counts tokens, and builds a document-wide offset index.

    Injected rather than imported so unit tests can use a trivial word counter:
    the chunker's logic is about boundaries, and asserting against a real BPE
    tokenizer would make the arithmetic in tests unreadable without testing
    anything extra.
    """

    def count(self, text: str) -> int: ...

    def index(self, text: str) -> TokenIndex: ...


class WordTokenCounter:
    """One token per whitespace-delimited word. For tests and rough sizing."""

    def count(self, text: str) -> int:
        return len(text.split())

    def index(self, text: str) -> TokenIndex:
        return TokenIndex([m.start() for m in re.finditer(r"\S+", text)])


class TiktokenCounter:
    """Production counter. The encoding is loaded once and reused."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        # `disallowed_special=()` because contract text legitimately contains
        # strings shaped like "<|endoftext|>", and the tokenizer raises on them
        # by default.
        return len(self._encoding.encode(text, disallowed_special=()))

    def index(self, text: str) -> TokenIndex:
        """Map every token to the character offset where it starts.

        tiktoken operates on UTF-8 bytes, so token boundaries are byte offsets
        and have to be translated. A byte begins a character iff it is not a
        continuation byte (``0b10xxxxxx``), which makes the byte-to-character map
        a single vectorised pass.
        """
        if not text:
            return TokenIndex([])

        tokens = self._encoding.encode(text, disallowed_special=())
        if not tokens:
            return TokenIndex([])

        piece_lengths = [len(piece) for piece in self._encoding.decode_tokens_bytes(tokens)]
        byte_starts = np.concatenate(([0], np.cumsum(piece_lengths)[:-1]))

        raw = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        # Byte offset at which each character begins.
        char_byte_starts = np.flatnonzero((raw & 0xC0) != 0x80)

        # A token boundary can land mid-character (BPE splits on bytes), so snap
        # back to the character containing that byte.
        char_starts = np.searchsorted(char_byte_starts, byte_starts, side="right") - 1
        return TokenIndex(char_starts.tolist())


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Chunking parameters. Mirrors ``config.ChunkingSettings``."""

    max_tokens: int = 512
    min_tokens: int = 64
    #: Applied only to hard splits. Clamped to ``max_tokens - 1`` at use, so
    #: lowering ``max_tokens`` for a test or an ablation does not also require
    #: lowering this. See ``_overlap_words``.
    overlap_tokens: int = 64

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.min_tokens < 0 or self.overlap_tokens < 0:
            raise ValueError("min_tokens and overlap_tokens must be non-negative")


@dataclass(frozen=True, slots=True)
class Heading:
    """A detected section heading and where it starts."""

    #: The numbering or keyword that identified it: "2.1", "ARTICLE I".
    label: str
    #: The heading line as it appears, trimmed. Used as chunk metadata.
    text: str
    char_start: int


# A heading starts at the beginning of the document, after a newline, or after a
# run of two or more spaces. That last alternative is what makes this work on the
# 59% of contracts that are run-on text.
_BOUNDARY: Final = r"(?:\A|(?<=\n)|(?<=\s{2}))"

#: "1." / "2.1" / "11.3.2", followed by whitespace then a capital, a quote, or an
#: opening paren. Each numeric component is capped at two digits so years (1934)
#: and dollar amounts do not match.
_NUMBERED: Final = re.compile(
    _BOUNDARY + r"[ \t]*(?P<label>\d{1,2}(?:\.\d{1,2}){0,3})\.?[ \t]+(?=[A-Z\"(])"
)

#: "ARTICLE IV", "Section 4.2". Requires a following capital or line end so that
#: cross-references like "Section 13 of the Exchange Act" do not match.
_ARTICLE: Final = re.compile(
    _BOUNDARY + r"[ \t]*(?P<label>(?:ARTICLE|Article|SECTION|Section)[ \t]+"
    r"(?:[IVXLC]{1,6}|\d{1,2}(?:\.\d{1,2}){0,2}))"
    r"(?=[ \t]*(?:\n|[:.—-]|[ \t]+[A-Z]))"
)

#: How far past a heading to look for its title text, in characters.
_HEADING_TEXT_WINDOW: Final = 80


def find_headings(text: str) -> list[Heading]:
    """Locate section headings, in document order.

    Returns an empty list for unstructured text, which is the correct answer for
    57 of CUAD's 510 contracts rather than a failure.

    There is deliberately no separate "is this a citation" filter. Cross-
    references like "pursuant to Section 13 of the Exchange Act" are excluded
    structurally: ``_BOUNDARY`` requires the match to follow a line break or a
    run of two or more spaces, and both patterns require a capital letter after
    the number. Prose references satisfy neither. An earlier version did filter
    on "is the preceding word lowercase", which rejected the majority of real
    headings, since run-on contracts put headings mid-line after a space run.
    """
    found: dict[int, Heading] = {}

    for pattern in (_NUMBERED, _ARTICLE):
        for match in pattern.finditer(text):
            label = match.group("label")
            start = match.start("label")
            # A numbered and an article pattern can fire at the same offset
            # ("Section 4.2" vs "4.2"); keep the longer, more specific label.
            existing = found.get(start)
            if existing is None or len(label) > len(existing.label):
                found[start] = Heading(
                    label=label,
                    text=_heading_text(text, start),
                    char_start=start,
                )

    return [found[key] for key in sorted(found)]


def _heading_text(text: str, start: int) -> str:
    """The heading label plus its title, up to the end of the line or a sentence."""
    window = text[start : start + _HEADING_TEXT_WINDOW]
    for terminator in ("\n", "  "):
        index = window.find(terminator)
        if index > 0:
            window = window[:index]
    return window.strip()


@dataclass(frozen=True, slots=True)
class _Span:
    """A half-open character range in the document, with its heading."""

    start: int
    end: int
    heading: Heading | None


def chunk_document(
    document: Document,
    config: ChunkingConfig | None = None,
    counter: TokenCounter | None = None,
) -> list[Chunk]:
    """Split a document into retrievable chunks anchored to raw-text offsets."""
    config = config or ChunkingConfig()
    counter = counter or TiktokenCounter()
    # Tokenize once. Every size question below is a binary search over this.
    index = counter.index(document.text)

    pieces: list[_Span] = []
    for section in _split_into_sections(document.text):
        pieces.extend(_split_section(document.text, section, config, index))

    pieces = _merge_undersized(pieces, config, index)

    return [
        Chunk(
            chunk_id=f"{document.document_id}::{ordinal:04d}",
            document_id=document.document_id,
            text=document.text[piece.start : piece.end],
            char_start=piece.start,
            char_end=piece.end,
            ordinal=ordinal,
            heading=piece.heading.text if piece.heading else None,
            token_count=index.count_range(piece.start, piece.end),
        )
        for ordinal, piece in enumerate(pieces)
    ]


def _split_into_sections(text: str) -> list[_Span]:
    """Cut the document at heading boundaries, keeping any preamble."""
    headings = find_headings(text)
    if not headings:
        span = _trim(text, 0, len(text))
        return [_Span(*span, heading=None)] if span else []

    sections: list[_Span] = []

    preamble = _trim(text, 0, headings[0].char_start)
    if preamble:
        sections.append(_Span(*preamble, heading=None))

    for position, heading in enumerate(headings):
        end = headings[position + 1].char_start if position + 1 < len(headings) else len(text)
        trimmed = _trim(text, heading.char_start, end)
        if trimmed:
            sections.append(_Span(*trimmed, heading=heading))

    return sections


def _split_section(
    text: str, section: _Span, config: ChunkingConfig, index: TokenIndex
) -> list[_Span]:
    """Break one section down to fit ``max_tokens``.

    Paragraph boundaries are preferred; a single paragraph over the limit is
    hard-split with overlap.
    """
    if index.count_range(section.start, section.end) <= config.max_tokens:
        return [section]

    pieces: list[_Span] = []
    for para_start, para_end in _paragraphs(text, section.start, section.end):
        if index.count_range(para_start, para_end) > config.max_tokens:
            pieces.extend(
                _Span(piece_start, piece_end, section.heading)
                for piece_start, piece_end in _hard_split(text, para_start, para_end, config, index)
            )
            continue

        # Extend the open piece if the paragraph still fits, otherwise start one.
        if (
            pieces
            and pieces[-1].end <= para_start
            and index.count_range(pieces[-1].start, para_end) <= config.max_tokens
        ):
            pieces[-1] = _Span(pieces[-1].start, para_end, section.heading)
            continue
        pieces.append(_Span(para_start, para_end, section.heading))

    return pieces


def _paragraphs(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Paragraph ranges within a section.

    Splits on blank lines. Where a section has none -- common in run-on
    contracts -- the whole section comes back as one paragraph and the caller
    falls through to the hard split.
    """
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in re.finditer(r"\n[ \t]*\n", text[start:end]):
        trimmed = _trim(text, cursor, start + match.start())
        if trimmed:
            spans.append(trimmed)
        cursor = start + match.end()
    trimmed = _trim(text, cursor, end)
    if trimmed:
        spans.append(trimmed)
    return spans or ([(start, end)] if end > start else [])


def _hard_split(
    text: str, start: int, end: int, config: ChunkingConfig, index: TokenIndex
) -> list[tuple[int, int]]:
    """Split a single oversized paragraph on word boundaries, with overlap.

    Cuts land between words so no chunk begins or ends mid-word. Overlap keeps a
    clause that straddles a cut fully present in at least one piece.
    """
    boundaries = _word_boundaries(text, start, end)
    if len(boundaries) <= 1:
        return [(start, end)]

    spans: list[tuple[int, int]] = []
    position = 0
    while position < len(boundaries):
        take = _take_while_fitting(boundaries, position, config.max_tokens, index)
        spans.append((boundaries[position][0], boundaries[take - 1][1]))

        if take >= len(boundaries):
            break

        # Step back by the overlap, but always advance at least one word, or
        # this loops forever once overlap approaches max_tokens.
        step = max(1, (take - position) - _overlap_words(config))
        position += step

    return spans


def _take_while_fitting(
    boundaries: list[tuple[int, int]],
    position: int,
    max_tokens: int,
    index: TokenIndex,
) -> int:
    """Index one past the last word that fits in ``max_tokens`` from ``position``.

    Binary search is cheap now that a range count is a binary search itself
    rather than a re-tokenization.
    """
    low, high = position + 1, len(boundaries)
    while low < high:
        mid = (low + high + 1) // 2
        if index.count_range(boundaries[position][0], boundaries[mid - 1][1]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return low


def _overlap_words(config: ChunkingConfig) -> int:
    """Overlap expressed in words rather than tokens.

    Approximate on purpose: it only has to be stable and strictly smaller than
    the chunk, so that hard splits always make forward progress.
    """
    return min(config.overlap_tokens, max(0, config.max_tokens - 1))


def _word_boundaries(text: str, start: int, end: int) -> list[tuple[int, int]]:
    return [(start + m.start(), start + m.end()) for m in re.finditer(r"\S+", text[start:end])]


def _merge_undersized(
    pieces: list[_Span], config: ChunkingConfig, index: TokenIndex
) -> list[_Span]:
    """Fold chunks below ``min_tokens`` into the following chunk.

    Without this, a bare heading line ("2. NOTICES.") becomes its own chunk and
    retrieves as a match with no content behind it.
    """
    if config.min_tokens <= 0 or not pieces:
        return pieces

    merged: list[_Span] = []
    for piece in pieces:
        if not merged:
            merged.append(piece)
            continue

        previous = merged[-1]
        if index.count_range(previous.start, previous.end) >= config.min_tokens:
            merged.append(piece)
            continue

        # Only merge forward across contiguous text, and never past `max_tokens`.
        if (
            piece.start >= previous.end
            and index.count_range(previous.start, piece.end) <= config.max_tokens
        ):
            merged[-1] = _Span(previous.start, piece.end, previous.heading or piece.heading)
        else:
            merged.append(piece)

    return merged


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink a range past surrounding whitespace, or None if nothing is left.

    Chunks must not begin or end on whitespace: the offsets stay exact, but a
    chunk padded with newlines wastes context and reads badly as evidence.
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None
