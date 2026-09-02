"""Unit tests for the embedding providers.

Only ``HashingEmbedder`` is tested here: ``BgeEmbedder`` needs 1.5 GB of weights
and is exercised by the integration tests. The properties asserted are the ones
the pgvector column and the cosine index actually depend on -- dimension,
unit norm, and determinism. A non-unit vector silently degrades cosine ranking
rather than raising.
"""

from __future__ import annotations

import math

import pytest

from docintel.retrieval.embed import QUERY_PREFIX, HashingEmbedder


class TestHashingEmbedder:
    def test_dimension_is_respected(self) -> None:
        assert HashingEmbedder(dimension=128).dimension == 128

    def test_rejects_non_positive_dimension(self) -> None:
        with pytest.raises(ValueError, match="dimension must be positive"):
            HashingEmbedder(dimension=0)

    def test_vectors_have_the_declared_dimension(self) -> None:
        embedder = HashingEmbedder(dimension=64)
        vectors = embedder.embed_passages(["governing law", "liability cap"])
        assert all(len(vector) == 64 for vector in vectors)
        assert len(embedder.embed_query("notice period")) == 64

    def test_vectors_are_unit_norm(self) -> None:
        """pgvector's cosine ops assume this; a non-unit vector skews ranking."""
        embedder = HashingEmbedder(dimension=64)
        for vector in embedder.embed_passages(["a b c", "the term of this agreement"]):
            assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-9)

    def test_empty_text_still_yields_a_unit_vector(self) -> None:
        """A zero vector makes cosine distance undefined in pgvector."""
        vector = HashingEmbedder(dimension=32).embed_query("")
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-9)

    def test_whitespace_only_text_behaves_like_empty(self) -> None:
        embedder = HashingEmbedder(dimension=32)
        assert embedder.embed_query("   \n\t ") == embedder.embed_query("")

    def test_is_deterministic(self) -> None:
        a, b = HashingEmbedder(dimension=64), HashingEmbedder(dimension=64)
        assert a.embed_query("governing law of delaware") == b.embed_query(
            "governing law of delaware"
        )

    def test_seed_changes_the_projection(self) -> None:
        a = HashingEmbedder(dimension=64, seed=1)
        b = HashingEmbedder(dimension=64, seed=2)
        assert a.embed_query("governing law") != b.embed_query("governing law")

    def test_different_text_gives_different_vectors(self) -> None:
        embedder = HashingEmbedder(dimension=256)
        assert embedder.embed_query("governing law") != embedder.embed_query("liability cap")

    def test_is_case_insensitive(self) -> None:
        embedder = HashingEmbedder(dimension=64)
        assert embedder.embed_query("Governing Law") == embedder.embed_query("governing law")

    def test_embed_passages_of_empty_list_is_empty(self) -> None:
        assert HashingEmbedder(dimension=32).embed_passages([]) == []

    def test_query_and_passage_paths_agree(self) -> None:
        """The fake deliberately applies no query prefix, unlike bge."""
        embedder = HashingEmbedder(dimension=64)
        assert embedder.embed_query("term") == embedder.embed_passages(["term"])[0]


def test_query_prefix_is_the_documented_bge_instruction() -> None:
    """Guards a silent quality regression.

    bge-* models are trained asymmetrically. Dropping or altering this prefix
    costs retrieval quality with no error anywhere, so it is pinned.
    """
    assert QUERY_PREFIX == "Represent this sentence for searching relevant passages: "
