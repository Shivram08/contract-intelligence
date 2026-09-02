"""Cross-encoder reranking over fused candidates.

The retrievers score a query and a chunk independently and compare the results.
A cross-encoder reads both together, so it can judge whether *this* passage
answers *this* question rather than whether the two are broadly about the same
topic. It is far more accurate and far too slow to run over a corpus, which is
why it sits behind retrieval on a short candidate list.

For contracts the specific win is negation and direction. "Liability is capped
at fees paid" and "liability is not capped" share nearly all their vocabulary
and sit close in embedding space; a cross-encoder separates them. That
distinction is exactly the ``cap_on_liability`` versus ``uncapped_liability``
pair from the clause schema, so the ablation in ``docs/RESULTS.md`` should show
reranking helping Tier 3 more than Tier 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from docintel.schemas import RetrievalHit

__all__ = ["BgeReranker", "IdentityReranker", "Reranker", "rerank"]


class Reranker(Protocol):
    """Scores (query, passage) pairs jointly. Higher is more relevant."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class BgeReranker:
    """``BAAI/bge-reranker-v2-m3`` as a local cross-encoder.

    Runs on CPU if no GPU is present. Scores are raw logits, not probabilities,
    and are comparable only within a single query's candidate list -- which is
    all reranking needs, and why no sigmoid is applied.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 16,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device, max_length=max_length)
        self._batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


class IdentityReranker:
    """Preserves the incoming order. The no-rerank arm of the ablation.

    Returning a descending sequence rather than zeros keeps the contract "higher
    is better" true, so ``rerank`` needs no special case and the two arms differ
    only in which scorer is passed.
    """

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(-position) for position in range(len(passages))]


def rerank(
    reranker: Reranker,
    query: str,
    hits: Sequence[RetrievalHit],
    top_k: int | None = None,
) -> list[RetrievalHit]:
    """Re-order fused hits by cross-encoder score.

    The fusion score and per-retriever ranks are carried through untouched:
    reranking overrides the *ordering*, but throwing away the provenance would
    make it impossible to ask afterwards whether reranking rescued chunks that
    only one arm found.

    Ties fall back to the incoming order, which is the fusion ranking, so a
    reranker that cannot separate two passages leaves retrieval's judgement in
    place rather than shuffling.
    """
    if not hits:
        return []

    scores = reranker.score(query, [hit.chunk.text for hit in hits])
    if len(scores) != len(hits):
        raise ValueError(f"reranker returned {len(scores)} scores for {len(hits)} hits")

    ordered = sorted(
        enumerate(hits),
        key=lambda pair: (-scores[pair[0]], pair[0]),
    )

    reranked = [
        hit.model_copy(update={"rerank_score": scores[position]}) for position, hit in ordered
    ]
    return reranked[:top_k] if top_k is not None else reranked
