"""Hybrid retrieval: lexical over Postgres FTS, dense over pgvector, fused with RRF.

**On the name.** ``CLAUDE.md`` calls the lexical arm "BM25 (Postgres FTS)".
Postgres full-text search is not BM25 -- ``ts_rank_cd`` is a cover-density rank
with different length normalization and no ``k1``/``b`` knobs. The retriever is
called ``lexical`` throughout so the results table does not claim an algorithm
this system does not run. RRF consumes ranks rather than scores, so the
substitution costs less than it would in a score-blending scheme.

**Why fuse ranks instead of scores.** BM25 returns unbounded, corpus-dependent
scores; cosine similarity returns [-1, 1]. Combining them numerically means
normalizing first, and every normalization scheme (min-max over the candidate
set, z-scores, softmax) makes the result depend on the shape of the candidate
list rather than on relevance. Reciprocal rank fusion sidesteps it by throwing
the scores away and using only position, which is the one thing both retrievers
agree on the meaning of.

**Why hybrid at all, for contracts specifically.** Legal language is full of
terms of art -- "indemnify", "Force Majeure", a defined term like "Territory" --
where exact lexical match is exactly right and embeddings blur the distinction.
It is equally full of paraphrase, where a governing-law clause never uses the
words "governing law". BM25 handles the first, dense handles the second, and the
ablation in ``docs/RESULTS.md`` is meant to show the gap between each alone and
the two together.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from docintel.ingest.index import TEXT_SEARCH_CONFIG
from docintel.retrieval.embed import Embedder
from docintel.schemas import Chunk, RetrievalHit, ScoredChunk

__all__ = [
    "DENSE",
    "LEXICAL",
    "RRF_K",
    "HybridRetriever",
    "dense_search",
    "lexical_search",
    "or_tsquery",
    "reciprocal_rank_fusion",
]

#: Retriever names. Used as dict keys in fusion and as labels in the ablation,
#: so they are constants rather than inline strings.
LEXICAL: Final = "lexical"
DENSE: Final = "dense"

#: Smoothing constant from Cormack, Clarke & Buettcher (2009). Large enough that
#: the difference between rank 1 and rank 2 is small, so a chunk both retrievers
#: like beats one that a single retriever loves. A tunable, not a law.
RRF_K: Final = 60


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[Chunk]],
    k: int = RRF_K,
    top_k: int | None = None,
) -> list[RetrievalHit]:
    """Fuse ranked candidate lists into one ranking.

    Each chunk scores ``sum(1 / (k + rank))`` over the retrievers that returned
    it, with ``rank`` 1-based. Retrievers that did not return a chunk contribute
    nothing -- they are absent from ``ranks`` rather than present with a sentinel,
    so the ablation can tell "not found" from "found, ranked last".

    Truncation to ``top_k`` happens *after* fusion. Cutting first would discard
    the deep-but-agreed-upon chunks that fusion exists to promote.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    chunks: dict[str, Chunk] = {}
    # Insertion order of first sighting, so ties break the same way every run.
    first_seen: dict[str, int] = {}

    # Sorting the retriever names makes the result independent of dict ordering,
    # which otherwise leaks into tie-breaking.
    for retriever in sorted(ranked_lists):
        seen = ranks.setdefault(retriever, {})
        for position, chunk in enumerate(ranked_lists[retriever], start=1):
            chunk_id = chunk.chunk_id
            # A retriever returning the same chunk twice must not double-count.
            if chunk_id in seen:
                continue
            seen[chunk_id] = position
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
            chunks.setdefault(chunk_id, chunk)
            first_seen.setdefault(chunk_id, len(first_seen))

    ordered = sorted(
        chunks,
        key=lambda chunk_id: (-scores[chunk_id], first_seen[chunk_id]),
    )
    if top_k is not None:
        ordered = ordered[:top_k]

    return [
        RetrievalHit(
            chunk=chunks[chunk_id],
            score=scores[chunk_id],
            ranks={
                retriever: positions[chunk_id]
                for retriever, positions in sorted(ranks.items())
                if chunk_id in positions
            },
        )
        for chunk_id in ordered
    ]


def _row_to_chunk(row: Sequence[Any]) -> Chunk:
    """Build a Chunk from a query row. Column order matches ``_CHUNK_COLUMNS``."""
    return Chunk(
        chunk_id=row[0],
        document_id=row[1],
        ordinal=row[2],
        text=row[3],
        char_start=row[4],
        char_end=row[5],
        heading=row[6],
        token_count=row[7],
    )


#: Column order that ``_row_to_chunk`` expects. Written out with the table alias
#: rather than assembled by string surgery, so a schema change fails loudly at
#: the query instead of silently shifting which column maps to which field.
_CHUNK_COLUMNS: Final = (
    "c.chunk_id, c.document_id, c.ordinal, c.text, "
    "c.char_start, c.char_end, c.heading, c.token_count"
)


#: Query terms. Punctuation is dropped rather than escaped: apostrophes and
#: hyphens in contract prose are not search operators, and passing them through
#: to ``to_tsquery`` is a syntax error.
_TERM = re.compile(r"[A-Za-z0-9]+")


def or_tsquery(query: str) -> str | None:
    """Build an OR-joined ``to_tsquery`` input from free text.

    **Why OR and not AND.** ``websearch_to_tsquery`` and ``plainto_tsquery``
    both conjoin terms, which turns full-text search into a boolean filter:
    every term must appear in the same chunk. That is fine for a two-word
    keyword query and useless for a sentence. Measured on the indexed dev split,
    the question "if a third party receives more favorable pricing the buyer is
    entitled to those terms" matches **0** chunks under AND and 2,148 under OR.

    Zero is the dangerous outcome, because it is silent: the lexical arm
    contributes nothing, fusion has only one list to work with, and "hybrid"
    retrieval quietly degrades to dense-only exactly when the query is phrased
    the way the agent phrases them.

    OR restores the behaviour a lexical ranker is supposed to have -- every
    partial match is a candidate, and ``ts_rank_cd`` sorts by how much of the
    query a chunk covers. Recall is the retriever's job; precision is fusion's
    and the reranker's.

    Returns ``None`` when the query has no usable terms, so the caller can skip
    the round trip instead of sending ``to_tsquery`` an empty string, which
    raises.
    """
    terms = _TERM.findall(query.lower())
    return " | ".join(terms) if terms else None


def lexical_search(
    connection: Any,
    query: str,
    limit: int,
    document_ids: Sequence[str] | None = None,
) -> list[ScoredChunk]:
    """Rank chunks by Postgres full-text relevance.

    Stemming and stop word removal happen inside ``to_tsquery`` with the
    ``english`` configuration, so the OR list built by ``or_tsquery`` does not
    need to do either.

    ``ts_rank_cd`` with normalization 32 divides by ``rank + 1``, bounding the
    score into [0, 1). Chunks here are of comparable size by construction, so
    length normalization is mostly cosmetic -- but an unbounded score would make
    the numbers harder to eyeball next to cosine similarity in a trace.
    """
    tsquery = or_tsquery(query)
    if tsquery is None:
        return []
    filter_sql = "AND c.document_id = ANY(%(document_ids)s)" if document_ids else ""
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {_CHUNK_COLUMNS},
                   ts_rank_cd(c.fts, query, 32) AS score
            FROM chunks c, to_tsquery(%(config)s, %(tsquery)s) AS query
            WHERE c.fts @@ query
            {filter_sql}
            ORDER BY score DESC, c.chunk_id
            LIMIT %(limit)s
            """,
            {
                "config": TEXT_SEARCH_CONFIG,
                "tsquery": tsquery,
                "limit": limit,
                "document_ids": list(document_ids) if document_ids else None,
            },
        )
        return [
            ScoredChunk(chunk=_row_to_chunk(row), score=float(row[8]), retriever=LEXICAL)
            for row in cursor.fetchall()
        ]


def dense_search(
    connection: Any,
    query_embedding: Sequence[float],
    limit: int,
    document_ids: Sequence[str] | None = None,
) -> list[ScoredChunk]:
    """Rank chunks by cosine similarity in pgvector.

    ``<=>`` is cosine *distance*, so it sorts ascending and the reported score
    is ``1 - distance`` to keep "higher is better" consistent across retrievers.
    """
    filter_sql = "AND c.document_id = ANY(%(document_ids)s)" if document_ids else ""
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {_CHUNK_COLUMNS},
                   c.embedding <=> %(embedding)s::vector AS distance
            FROM chunks c
            WHERE c.embedding IS NOT NULL
            {filter_sql}
            ORDER BY distance ASC, c.chunk_id
            LIMIT %(limit)s
            """,
            {
                "embedding": list(query_embedding),
                "limit": limit,
                "document_ids": list(document_ids) if document_ids else None,
            },
        )
        return [
            ScoredChunk(chunk=_row_to_chunk(row), score=1.0 - float(row[8]), retriever=DENSE)
            for row in cursor.fetchall()
        ]


@dataclass(slots=True)
class HybridRetriever:
    """Runs both arms and fuses them.

    ``candidates_per_retriever`` is deliberately much larger than ``top_k``.
    Fusion can only promote a chunk that at least one arm returned, so a narrow
    candidate window throws away the disagreement that fusion exists to
    exploit -- and it is where the ablation's headroom comes from.
    """

    connection: Any
    embedder: Embedder
    candidates_per_retriever: int = 50
    rrf_k: int = RRF_K

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_ids: Sequence[str] | None = None,
        arms: Sequence[str] = (LEXICAL, DENSE),
    ) -> list[RetrievalHit]:
        """Retrieve and fuse. ``arms`` selects which retrievers run, for ablations."""
        if not arms:
            raise ValueError("at least one retrieval arm is required")

        ranked: dict[str, list[Chunk]] = {}

        if LEXICAL in arms:
            ranked[LEXICAL] = [
                scored.chunk
                for scored in lexical_search(
                    self.connection, query, self.candidates_per_retriever, document_ids
                )
            ]

        if DENSE in arms:
            ranked[DENSE] = [
                scored.chunk
                for scored in dense_search(
                    self.connection,
                    self.embedder.embed_query(query),
                    self.candidates_per_retriever,
                    document_ids,
                )
            ]

        return reciprocal_rank_fusion(ranked, k=self.rrf_k, top_k=top_k)
