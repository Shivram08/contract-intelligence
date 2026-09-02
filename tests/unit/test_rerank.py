"""Unit tests for cross-encoder reranking.

Uses a stub scorer. The cross-encoder itself is a downloaded model; what needs
testing here is the ordering contract around it -- that provenance survives,
that ties fall back to the retrieval ranking rather than shuffling, and that a
scorer returning the wrong number of scores fails loudly instead of silently
mis-pairing scores with passages.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from docintel.retrieval.rerank import IdentityReranker, rerank
from docintel.schemas import Chunk, RetrievalHit


def make_hit(chunk_id: str, text: str, score: float = 0.1, **ranks: int) -> RetrievalHit:
    return RetrievalHit(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="DOC",
            text=text,
            char_start=0,
            char_end=len(text),
            ordinal=0,
            token_count=len(text.split()),
        ),
        score=score,
        ranks=dict(ranks),
    )


class KeywordReranker:
    """Scores by how many times a keyword appears. Deterministic, no model."""

    def __init__(self, keyword: str) -> None:
        self._keyword = keyword

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(passage.lower().count(self._keyword)) for passage in passages]


class ConstantReranker:
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [1.0] * len(passages)


class WrongLengthReranker:
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [1.0]


class TestRerankOrdering:
    def test_reorders_by_score(self) -> None:
        hits = [
            make_hit("a", "nothing relevant here"),
            make_hit("b", "liability liability liability"),
            make_hit("c", "liability once"),
        ]
        result = rerank(KeywordReranker("liability"), "liability", hits)
        assert [hit.chunk.chunk_id for hit in result] == ["b", "c", "a"]

    def test_records_the_rerank_score(self) -> None:
        hits = [make_hit("a", "liability liability")]
        assert rerank(KeywordReranker("liability"), "q", hits)[0].rerank_score == 2.0

    def test_ties_preserve_the_incoming_fusion_order(self) -> None:
        """A reranker that cannot separate passages must not shuffle them."""
        hits = [make_hit(name, "same text") for name in ("a", "b", "c", "d")]
        result = rerank(ConstantReranker(), "q", hits)
        assert [hit.chunk.chunk_id for hit in result] == ["a", "b", "c", "d"]

    def test_top_k_truncates_after_reordering(self) -> None:
        hits = [
            make_hit("a", "no match"),
            make_hit("b", "cap cap cap"),
            make_hit("c", "cap"),
        ]
        result = rerank(KeywordReranker("cap"), "cap", hits, top_k=2)
        assert [hit.chunk.chunk_id for hit in result] == ["b", "c"]

    def test_top_k_larger_than_the_list_is_harmless(self) -> None:
        hits = [make_hit("a", "cap")]
        assert len(rerank(KeywordReranker("cap"), "cap", hits, top_k=99)) == 1


class TestProvenanceSurvives:
    def test_fusion_score_is_preserved(self) -> None:
        hits = [make_hit("a", "cap", score=0.0328)]
        assert rerank(KeywordReranker("cap"), "cap", hits)[0].score == 0.0328

    def test_per_retriever_ranks_are_preserved(self) -> None:
        """Without this, the 'did reranking rescue single-arm finds?' question
        becomes unanswerable after the fact."""
        hits = [make_hit("a", "cap", lexical=3, dense=17)]
        assert rerank(KeywordReranker("cap"), "cap", hits)[0].ranks == {
            "lexical": 3,
            "dense": 17,
        }

    def test_chunk_is_unchanged(self) -> None:
        hits = [make_hit("a", "cap on liability")]
        result = rerank(KeywordReranker("cap"), "cap", hits)[0]
        assert result.chunk == hits[0].chunk


class TestEdgeCases:
    def test_empty_hit_list_returns_empty(self) -> None:
        assert rerank(KeywordReranker("x"), "q", []) == []

    def test_score_length_mismatch_raises(self) -> None:
        """Silently zipping mismatched lists would pair scores with the wrong
        passages -- a wrong ranking that looks entirely plausible."""
        hits = [make_hit("a", "x"), make_hit("b", "y")]
        with pytest.raises(ValueError, match="returned 1 scores for 2 hits"):
            rerank(WrongLengthReranker(), "q", hits)


class TestIdentityReranker:
    def test_preserves_order(self) -> None:
        hits = [make_hit(name, f"text {name}") for name in ("a", "b", "c")]
        result = rerank(IdentityReranker(), "q", hits)
        assert [hit.chunk.chunk_id for hit in result] == ["a", "b", "c"]

    def test_scores_descend_so_higher_is_better_still_holds(self) -> None:
        scores = IdentityReranker().score("q", ["a", "b", "c"])
        assert scores == sorted(scores, reverse=True)

    def test_empty_passages(self) -> None:
        assert IdentityReranker().score("q", []) == []
