"""Embedding providers.

``Embedder`` is a Protocol with two implementations. ``BgeEmbedder`` is the real
one and needs ``sentence-transformers`` plus torch, about 1.5 GB of weights.
``HashingEmbedder`` is deterministic, dependency-free, and produces vectors of
the right shape with no semantic content.

The fake exists so the retrieval *plumbing* -- SQL, vector round-tripping,
fusion, ranking, the API contract -- is testable in CI without downloading a
model. It is explicitly not a stand-in for measuring retrieval quality: any
accuracy number produced with it is meaningless, which is why its class
docstring says so and why the integration tests that use it assert on structure
rather than on relevance.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final, Protocol

import numpy as np

__all__ = ["QUERY_PREFIX", "BgeEmbedder", "Embedder", "HashingEmbedder"]

#: bge-* models are trained asymmetrically: queries get an instruction prefix,
#: passages do not. Omitting it costs several points of retrieval quality, and
#: the failure is silent -- everything still runs, just worse.
QUERY_PREFIX: Final = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    """Turns text into unit-norm vectors."""

    @property
    def dimension(self) -> int: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BgeEmbedder:
    """``BAAI/bge-base-en-v1.5`` via sentence-transformers.

    Vectors are L2-normalized so that cosine distance in pgvector (``<=>``) is
    equivalent to inner product, and so the HNSW index built with
    ``vector_cosine_ops`` behaves as expected.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._dimension = int(self._model.get_sentence_embedding_dimension() or 0)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return list(vector.tolist())


class HashingEmbedder:
    """Deterministic bag-of-words hashing. **Carries no semantics.**

    For testing plumbing only. Two paraphrases of the same clause get unrelated
    vectors, so any retrieval quality measured with this is noise. It is here
    because the alternative -- skipping every retrieval test unless a 1.5 GB
    model is present -- means the SQL and the fusion never get exercised in CI.
    """

    def __init__(self, dimension: int = 768, seed: int = 42) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._seed = seed

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        accumulator = np.zeros(self._dimension, dtype=np.float64)
        for token in text.lower().split():
            digest = hashlib.blake2b(
                token.encode("utf-8"), digest_size=8, key=str(self._seed).encode()
            ).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            # Signed so that unrelated tokens can cancel instead of only adding.
            sign = 1.0 if digest[4] & 1 else -1.0
            accumulator[bucket] += sign

        norm = float(np.linalg.norm(accumulator))
        if norm == 0.0:
            # An empty or all-whitespace passage. A zero vector makes cosine
            # distance undefined in pgvector, so pick a fixed unit vector.
            accumulator[0] = 1.0
            norm = 1.0
        normalized: list[float] = (accumulator / norm).tolist()
        return normalized

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        # No query prefix: it would only add constant noise to every query.
        return self._vector(text)
