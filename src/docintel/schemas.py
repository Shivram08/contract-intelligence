"""Typed contracts shared across the pipeline.

M1 scope: documents, chunks, and retrieval results. The extraction schemas
(``ClauseExtraction``, ``ExtractionResult``) arrive with the agent in M2.

The invariant that runs through all of this: **a chunk's text is a verbatim
slice of its document.** ``document.text[chunk.char_start:chunk.char_end] ==
chunk.text``, always. Every offset in the system is an offset into the raw,
unmodified document, so an evidence quote can be traced from a model response
back through a chunk to an exact character range in the source file. Break that
and the grounding verifier silently starts rejecting correct extractions -- see
``docs/DATA_AUDIT.md`` check 4.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["Chunk", "Document", "RetrievalHit", "ScoredChunk"]


class Document(BaseModel):
    """A contract in its offset-authoritative form.

    ``text`` is the file exactly as read: no whitespace collapsing, no newline
    translation, no Unicode re-normalization. It is the coordinate system that
    every offset in the pipeline refers to.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    text: str
    #: Where the text came from, for provenance in errors and traces.
    source_path: str | None = None
    #: Free-form; CUAD carries agreement type, split assignment, and so on.
    metadata: dict[str, str] = Field(default_factory=dict)

    def slice(self, char_start: int, char_end: int) -> str:
        """Read a character range out of the raw text."""
        return self.text[char_start:char_end]

    def __len__(self) -> int:
        return len(self.text)


class Chunk(BaseModel):
    """A retrievable passage, anchored to an exact range in its document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    #: Position in the document, 0-based. Retrieval uses this to fetch neighbours.
    ordinal: int = Field(ge=0)
    #: The section heading this chunk sits under, when one was detected.
    #: None for preamble text and for documents with no detectable structure --
    #: 41 of CUAD's 510 contracts have no numbered headings at all.
    heading: str | None = None
    token_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_range(self) -> Chunk:
        if self.char_end < self.char_start:
            raise ValueError(
                f"chunk {self.chunk_id}: char_end {self.char_end} precedes "
                f"char_start {self.char_start}"
            )
        span = self.char_end - self.char_start
        if span != len(self.text):
            # Catches the classic bug where a chunker strips or normalizes text
            # but keeps the original offsets. Cheap here, invisible later.
            raise ValueError(
                f"chunk {self.chunk_id}: offset span {span} does not match "
                f"text length {len(self.text)}"
            )
        return self

    def is_faithful_to(self, document: Document) -> bool:
        """Whether this chunk is a verbatim slice of the given document."""
        return document.slice(self.char_start, self.char_end) == self.text


class ScoredChunk(BaseModel):
    """A chunk with the score from one retriever."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    #: Which retriever produced this: "bm25", "dense", "rrf", "rerank".
    retriever: str


class RetrievalHit(BaseModel):
    """A fused result, carrying enough provenance to explain the ranking.

    The per-retriever ranks are kept because the interesting question about
    hybrid retrieval is not "what came back" but "which retriever found it" --
    that is the ablation story, and reconstructing it after the fact is
    impossible once the lists are merged.
    """

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    #: retriever name -> 1-based rank in that retriever's list. A retriever that
    #: did not return this chunk is absent rather than present with a sentinel.
    ranks: dict[str, int] = Field(default_factory=dict)
    rerank_score: float | None = None
