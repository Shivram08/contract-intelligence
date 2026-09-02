"""Hybrid retrieval: BM25 over Postgres FTS, dense over pgvector, fused with RRF.

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

from collections.abc import Mapping, Sequence
from typing import Final

from docintel.schemas import Chunk, RetrievalHit

__all__ = ["RRF_K", "reciprocal_rank_fusion"]

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
