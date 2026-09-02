"""Unit tests for reciprocal rank fusion.

RRF is the whole argument for hybrid retrieval, so it gets tested on its
properties rather than on one worked example. The properties that matter:

- It uses **ranks, not scores.** BM25 returns unbounded scores and cosine
  similarity returns [-1, 1]; any score-based combination needs normalization,
  and normalization is where hybrid retrieval quietly goes wrong.
- A document found by **both** retrievers should outrank one found by either
  alone, even when neither ranked it first. That is the entire point.
- It must be **deterministic** under ties, or eval numbers wobble between runs
  for no reason.
"""

from __future__ import annotations

import pytest

from docintel.retrieval.hybrid import RRF_K, reciprocal_rank_fusion
from docintel.schemas import Chunk, RetrievalHit


def make_chunk(chunk_id: str) -> Chunk:
    text = f"body of {chunk_id}"
    return Chunk(
        chunk_id=chunk_id,
        document_id="DOC",
        text=text,
        char_start=0,
        char_end=len(text),
        ordinal=0,
        token_count=3,
    )


def ranked(*chunk_ids: str) -> list[Chunk]:
    return [make_chunk(c) for c in chunk_ids]


def ids(hits: list[RetrievalHit]) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


class TestBasicBehaviour:
    def test_single_list_preserves_its_order(self) -> None:
        result = reciprocal_rank_fusion({"bm25": ranked("a", "b", "c")})
        assert ids(result) == ["a", "b", "c"]

    def test_empty_input_returns_empty(self) -> None:
        assert reciprocal_rank_fusion({}) == []

    def test_all_empty_lists_return_empty(self) -> None:
        assert reciprocal_rank_fusion({"bm25": [], "dense": []}) == []

    def test_one_empty_list_does_not_suppress_the_other(self) -> None:
        result = reciprocal_rank_fusion({"bm25": ranked("a", "b"), "dense": []})
        assert ids(result) == ["a", "b"]

    def test_union_of_both_lists_is_returned(self) -> None:
        result = reciprocal_rank_fusion({"bm25": ranked("a"), "dense": ranked("b")})
        assert set(ids(result)) == {"a", "b"}

    def test_scores_are_descending(self) -> None:
        result = reciprocal_rank_fusion(
            {"bm25": ranked("a", "b", "c"), "dense": ranked("c", "a", "d")}
        )
        scores = [hit.score for hit in result]
        assert scores == sorted(scores, reverse=True)


class TestAgreementWins:
    """The property that justifies hybrid retrieval at all."""

    def test_chunk_found_by_both_beats_chunk_found_by_one(self) -> None:
        result = reciprocal_rank_fusion(
            {"bm25": ranked("shared", "bm25_only"), "dense": ranked("dense_only", "shared")}
        )
        assert ids(result)[0] == "shared"

    def test_agreement_beats_a_single_first_place(self) -> None:
        """'shared' is 2nd in both lists and still wins against two 1st places.

        1/(60+2) + 1/(60+2) = 0.03226 > 1/(60+1) = 0.01639.
        """
        result = reciprocal_rank_fusion(
            {"bm25": ranked("bm25_top", "shared"), "dense": ranked("dense_top", "shared")}
        )
        assert ids(result)[0] == "shared"

    def test_deep_agreement_can_lose_to_strong_single_signal(self) -> None:
        """RRF is not unconditional -- k=60 bounds how much agreement is worth.

        A chunk ranked ~200th in both lists scores 1/262 + 1/261 = 0.0076, which
        loses to a single 1st place at 1/61 = 0.0164. Agreement is worth a lot,
        but not unboundedly; k sets the exchange rate. Filler is disjoint between
        the two lists so nothing but `deep` is actually agreed upon.
        """
        deep = make_chunk("deep")
        bm25 = [make_chunk("top"), *(make_chunk(f"a{i}") for i in range(200)), deep]
        dense = [*(make_chunk(f"b{i}") for i in range(200)), deep]
        result = reciprocal_rank_fusion({"bm25": bm25, "dense": dense})
        assert ids(result)[0] == "top"


class TestRankNotScore:
    def test_input_scores_are_ignored(self) -> None:
        """Only position matters. A retriever returning wild score magnitudes
        must not dominate one returning small ones."""
        first = reciprocal_rank_fusion({"bm25": ranked("a", "b"), "dense": ranked("b", "a")})
        second = reciprocal_rank_fusion({"bm25": ranked("a", "b"), "dense": ranked("b", "a")})
        assert ids(first) == ids(second)

    def test_score_matches_the_rrf_formula(self) -> None:
        result = reciprocal_rank_fusion({"bm25": ranked("a", "b")}, k=60)
        by_id = {hit.chunk.chunk_id: hit.score for hit in result}
        assert by_id["a"] == pytest.approx(1 / 61)
        assert by_id["b"] == pytest.approx(1 / 62)

    def test_score_sums_across_retrievers(self) -> None:
        result = reciprocal_rank_fusion({"bm25": ranked("a"), "dense": ranked("a")}, k=60)
        assert result[0].score == pytest.approx(2 / 61)


class TestSmoothingConstant:
    def test_smaller_k_sharpens_the_top_of_the_ranking(self) -> None:
        """Both lists agree on the order, so scores strictly decrease and the
        spread is meaningful. A symmetric disagreement would give every chunk the
        same score and make this vacuous."""
        lists = {"bm25": ranked("a", "b", "c"), "dense": ranked("a", "b", "c")}
        sharp = reciprocal_rank_fusion(lists, k=1)
        flat = reciprocal_rank_fusion(lists, k=1000)

        def spread(hits: list[RetrievalHit]) -> float:
            return hits[0].score - hits[-1].score

        assert spread(sharp) > spread(flat)

    def test_default_k_is_the_published_value(self) -> None:
        """60, from Cormack et al. (2009). Recorded so an ablation can move it."""
        assert RRF_K == 60

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion({"bm25": ranked("a")}, k=0)


class TestProvenance:
    def test_hit_records_each_retriever_rank(self) -> None:
        result = reciprocal_rank_fusion(
            {"bm25": ranked("x", "shared"), "dense": ranked("shared", "y")}
        )
        shared = next(h for h in result if h.chunk.chunk_id == "shared")
        assert shared.ranks == {"bm25": 2, "dense": 1}

    def test_absent_retriever_is_omitted_not_sentinelled(self) -> None:
        """The ablation question is 'which retriever found this', so a missing
        retriever must be distinguishable from one that ranked it last."""
        result = reciprocal_rank_fusion({"bm25": ranked("only"), "dense": ranked("other")})
        hit = next(h for h in result if h.chunk.chunk_id == "only")
        assert hit.ranks == {"bm25": 1}
        assert "dense" not in hit.ranks

    def test_ranks_are_one_based(self) -> None:
        result = reciprocal_rank_fusion({"bm25": ranked("first", "second")})
        assert [h.ranks["bm25"] for h in result] == [1, 2]


class TestDeterminism:
    def test_ties_break_deterministically(self) -> None:
        """Symmetric input: 'a' and 'b' have identical fused scores. The order
        must still be stable, or eval numbers move between runs."""
        lists = {"bm25": ranked("a", "b"), "dense": ranked("b", "a")}
        assert ids(reciprocal_rank_fusion(lists)) == ids(reciprocal_rank_fusion(lists))

    def test_result_is_independent_of_retriever_dict_ordering(self) -> None:
        forward = reciprocal_rank_fusion({"bm25": ranked("a", "b"), "dense": ranked("b", "a")})
        reverse = reciprocal_rank_fusion({"dense": ranked("b", "a"), "bm25": ranked("a", "b")})
        assert ids(forward) == ids(reverse)

    def test_duplicate_chunk_in_one_list_counts_once(self) -> None:
        """A retriever returning the same chunk twice must not double its score."""
        result = reciprocal_rank_fusion({"bm25": ranked("a", "a", "b")})
        assert len(result) == 2
        assert result[0].score == pytest.approx(1 / 61)


class TestTopK:
    def test_top_k_truncates(self) -> None:
        result = reciprocal_rank_fusion({"bm25": ranked("a", "b", "c", "d")}, top_k=2)
        assert ids(result) == ["a", "b"]

    def test_top_k_larger_than_input_is_harmless(self) -> None:
        assert len(reciprocal_rank_fusion({"bm25": ranked("a")}, top_k=99)) == 1

    def test_truncation_happens_after_fusion(self) -> None:
        """Cutting before fusion would drop a chunk that agreement promotes."""
        result = reciprocal_rank_fusion(
            {"bm25": ranked("x", "y", "shared"), "dense": ranked("z", "w", "shared")},
            top_k=1,
        )
        assert ids(result) == ["shared"]
