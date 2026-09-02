"""Integration tests for indexing and hybrid retrieval against real Postgres.

Marked ``integration`` and deselected in the default test run. These exist
because the parts of retrieval most likely to break are the parts unit tests
cannot reach: the generated ``tsvector`` column, ``websearch_to_tsquery``
parsing, vector round-tripping through pgvector, and whether the offsets that
came out of the chunker survive a trip through the database.

Bring the stack up with ``docker compose -f docker/docker-compose.yml up -d``.

``HashingEmbedder`` is used throughout, so nothing here asserts on semantic
relevance -- these tests check that the plumbing carries data faithfully. The
lexical arm *is* real, so its assertions are about genuine matching behaviour.

Everything runs against a dedicated **database**, created on demand and never
the one the CLI indexes into. Two earlier attempts got this wrong and both
deleted a developer's real index:

1. Sharing the working database outright -- teardown dropped the live tables.
2. Sandboxing with ``SET search_path TO scratch, public``. This looks airtight
   and is not: ``DROP TABLE IF EXISTS chunks`` resolves through the search path,
   so as soon as one test dropped the scratch copy, the next unqualified drop
   fell through to ``public.chunks``.

A separate database is the only boundary here that does not depend on name
resolution. The fixture also asserts it is not connected to the working database
before it drops anything, because this has now gone wrong twice.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse, urlunparse

import pytest

from docintel.ingest.chunker import ChunkingConfig, WordTokenCounter, chunk_document
from docintel.ingest.index import (
    create_tables,
    create_vector_index,
    drop_tables,
    index_document,
    unindexed_document_ids,
)
from docintel.retrieval.embed import HashingEmbedder
from docintel.retrieval.hybrid import (
    DENSE,
    LEXICAL,
    HybridRetriever,
    dense_search,
    lexical_search,
)
from docintel.schemas import Document

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get(
    "DOCINTEL_TEST_DATABASE_URL",
    "postgresql://docintel:docintel@localhost:5433/docintel",
)
DIMENSION = 64

#: Tests run here. Created on demand; never the database the CLI indexes into.
TEST_DATABASE = "docintel_integration_test"

CONTRACT_ONE = """MASTER SERVICES AGREEMENT

1. TERM. This Agreement commences on the Effective Date and continues for three
years unless terminated earlier.

2. GOVERNING LAW. This Agreement is governed by the laws of the State of
Delaware, without regard to its conflict of laws principles.

3. LIMITATION OF LIABILITY. Each party's aggregate liability is capped at the
fees paid in the twelve months preceding the claim.
"""

CONTRACT_TWO = """DISTRIBUTION AGREEMENT

1. APPOINTMENT. Supplier appoints Distributor as its non-exclusive distributor
in the Territory.

2. GOVERNING LAW. This Agreement shall be construed in accordance with the laws
of the State of New York.

3. INDEMNIFICATION. Supplier shall indemnify Distributor against third party
claims of patent infringement. Liability for such claims is not capped.
"""


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Create the dedicated test database if it does not exist, return its URL."""
    psycopg = pytest.importorskip("psycopg")
    try:
        admin = psycopg.connect(DATABASE_URL, connect_timeout=5, autocommit=True)
    except psycopg.OperationalError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres unavailable at {DATABASE_URL}: {exc}")

    with admin, admin.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DATABASE,))
        if cursor.fetchone() is None:
            # CREATE DATABASE cannot run inside a transaction, hence autocommit.
            # The name is a module constant, not caller input.
            cursor.execute(f'CREATE DATABASE "{TEST_DATABASE}"')

    url = urlunparse(urlparse(DATABASE_URL)._replace(path=f"/{TEST_DATABASE}"))
    with psycopg.connect(url, autocommit=True) as setup, setup.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    return url


@pytest.fixture(scope="module")
def connection(test_database_url: str) -> Iterator[Any]:
    psycopg = pytest.importorskip("psycopg")
    pgvector_psycopg = pytest.importorskip("pgvector.psycopg")

    conn = psycopg.connect(test_database_url, connect_timeout=5)
    pgvector_psycopg.register_vector(conn)

    # Belt and braces. Everything below drops tables, and this has already gone
    # wrong twice; a wrong database here must fail loudly, not quietly.
    with conn.cursor() as cursor:
        cursor.execute("SELECT current_database()")
        (current,) = cursor.fetchone()
    assert current == TEST_DATABASE, (
        f"integration tests would drop tables in {current!r}; expected {TEST_DATABASE!r}"
    )

    try:
        yield conn
    finally:
        drop_tables(conn)
        conn.close()


@pytest.fixture(scope="module")
def indexed(connection: Any) -> Iterator[dict[str, Document]]:
    """A two-document corpus, chunked and embedded."""
    embedder = HashingEmbedder(dimension=DIMENSION)
    counter = WordTokenCounter()
    config = ChunkingConfig(max_tokens=60, min_tokens=0, overlap_tokens=0)

    drop_tables(connection)
    create_tables(connection, embedding_dim=DIMENSION)

    documents = {
        "MSA": Document(document_id="MSA", text=CONTRACT_ONE),
        "DIST": Document(document_id="DIST", text=CONTRACT_TWO),
    }
    for document in documents.values():
        chunks = chunk_document(document, config, counter)
        embeddings = embedder.embed_passages([chunk.text for chunk in chunks])
        index_document(connection, document, chunks, embeddings)

    create_vector_index(connection)
    try:
        yield documents
    finally:
        drop_tables(connection)


class TestSchema:
    def test_create_tables_is_idempotent(self, connection: Any) -> None:
        create_tables(connection, embedding_dim=DIMENSION)
        create_tables(connection, embedding_dim=DIMENSION)
        drop_tables(connection)

    def test_chunk_check_constraint_rejects_inverted_offsets(self, connection: Any) -> None:
        """The database is the last line of defence for the offset invariant."""
        import psycopg

        create_tables(connection, embedding_dim=DIMENSION)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO documents (document_id, text, content_hash) "
                    "VALUES ('X', 'body', 'h')"
                )
                with pytest.raises(psycopg.errors.CheckViolation):
                    cursor.execute(
                        "INSERT INTO chunks (chunk_id, document_id, ordinal, text, "
                        "char_start, char_end, token_count) "
                        "VALUES ('X::0', 'X', 0, 'body', 10, 4, 1)"
                    )
            connection.rollback()
        finally:
            drop_tables(connection)


class TestOffsetsSurviveTheDatabase:
    def test_stored_chunks_still_slice_their_document(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        """The whole evidence chain depends on this holding after a round trip."""
        with connection.cursor() as cursor:
            cursor.execute("SELECT document_id, char_start, char_end, text FROM chunks")
            rows = cursor.fetchall()

        assert rows
        for document_id, char_start, char_end, text in rows:
            document = indexed[document_id]
            assert document.slice(char_start, char_end) == text

    def test_document_text_round_trips_exactly(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        with connection.cursor() as cursor:
            cursor.execute("SELECT document_id, text FROM documents")
            for document_id, text in cursor.fetchall():
                assert text == indexed[document_id].text


class TestLexicalSearch:
    def test_finds_a_clause_by_its_terms(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        hits = lexical_search(connection, "governing law Delaware", limit=10)
        assert hits
        assert "Delaware" in hits[0].chunk.text

    def test_stemming_matches_inflected_forms(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        """'english' config stems, so 'terminate' should reach 'terminated'."""
        hits = lexical_search(connection, "terminate", limit=10)
        assert any("terminated" in hit.chunk.text for hit in hits)

    def test_scores_are_bounded_by_normalization_32(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        hits = lexical_search(connection, "liability", limit=10)
        assert hits
        assert all(0.0 <= hit.score < 1.0 for hit in hits)

    def test_results_are_ordered_by_score(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        hits = lexical_search(connection, "liability agreement", limit=10)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_punctuation_heavy_query_does_not_raise(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        """Why websearch_to_tsquery instead of plainto_tsquery: agent- and
        hand-written queries contain quotes, slashes, and dashes."""
        for query in ['"governing law"', "liability -uncapped", "law/equity", "a & b"]:
            lexical_search(connection, query, limit=5)

    def test_no_match_returns_empty(self, connection: Any, indexed: dict[str, Document]) -> None:
        assert lexical_search(connection, "zzzzunmatchablezzzz", limit=10) == []

    def test_document_filter_restricts_results(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        hits = lexical_search(connection, "governing law", limit=10, document_ids=["MSA"])
        assert hits
        assert {hit.chunk.document_id for hit in hits} == {"MSA"}

    def test_limit_is_respected(self, connection: Any, indexed: dict[str, Document]) -> None:
        assert len(lexical_search(connection, "agreement", limit=1)) <= 1


class TestDenseSearch:
    def test_returns_similarity_not_distance(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        """Scores must read 'higher is better' like the lexical arm."""
        embedder = HashingEmbedder(dimension=DIMENSION)
        hits = dense_search(connection, embedder.embed_query("governing law"), limit=5)
        assert hits
        assert all(-1.0 <= hit.score <= 1.0 for hit in hits)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_exact_chunk_text_retrieves_itself_first(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        """A real check even with a semantics-free embedder: an identical string
        must hash to an identical vector and come back at distance ~0."""
        embedder = HashingEmbedder(dimension=DIMENSION)
        with connection.cursor() as cursor:
            cursor.execute("SELECT text FROM chunks ORDER BY chunk_id LIMIT 1")
            (target,) = cursor.fetchone()

        hits = dense_search(connection, embedder.embed_query(target), limit=1)
        assert hits[0].chunk.text == target
        assert hits[0].score == pytest.approx(1.0, abs=1e-6)

    def test_document_filter_restricts_results(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        embedder = HashingEmbedder(dimension=DIMENSION)
        hits = dense_search(
            connection, embedder.embed_query("indemnify"), limit=10, document_ids=["DIST"]
        )
        assert {hit.chunk.document_id for hit in hits} == {"DIST"}


class TestHybridRetriever:
    def test_returns_fused_hits_with_both_arms_recorded(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        retriever = HybridRetriever(
            connection=connection,
            embedder=HashingEmbedder(dimension=DIMENSION),
            candidates_per_retriever=20,
        )
        hits = retriever.search("governing law of the state", top_k=5)
        assert hits
        assert any(set(hit.ranks) == {LEXICAL, DENSE} for hit in hits), (
            "expected at least one chunk found by both arms"
        )
        assert all(hit.ranks for hit in hits)

    def test_single_arm_ablation_records_only_that_arm(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        retriever = HybridRetriever(
            connection=connection, embedder=HashingEmbedder(dimension=DIMENSION)
        )
        hits = retriever.search("liability", top_k=5, arms=(LEXICAL,))
        assert hits
        assert all(set(hit.ranks) == {LEXICAL} for hit in hits)

    def test_top_k_is_respected(self, connection: Any, indexed: dict[str, Document]) -> None:
        retriever = HybridRetriever(
            connection=connection, embedder=HashingEmbedder(dimension=DIMENSION)
        )
        assert len(retriever.search("agreement", top_k=2)) <= 2

    def test_rejects_an_empty_arm_list(self, connection: Any, indexed: dict[str, Document]) -> None:
        retriever = HybridRetriever(
            connection=connection, embedder=HashingEmbedder(dimension=DIMENSION)
        )
        with pytest.raises(ValueError, match="at least one retrieval arm"):
            retriever.search("x", arms=())


class TestReindexing:
    def test_unchanged_documents_are_reported_as_indexed(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        assert unindexed_document_ids(connection, indexed.values()) == []

    def test_edited_document_is_reported_as_stale(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        edited = Document(document_id="MSA", text=CONTRACT_ONE + "\n4. NOTICES. By email.\n")
        assert unindexed_document_ids(connection, [edited]) == ["MSA"]

    def test_unknown_document_is_reported_as_stale(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        fresh = Document(document_id="NEW", text="1. TERM. One year.")
        assert unindexed_document_ids(connection, [fresh]) == ["NEW"]

    def test_reindexing_replaces_chunks_rather_than_accumulating(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        """Re-chunking with different parameters changes ordinals, so upserting
        by id would leave orphaned chunks from the old configuration behind."""
        embedder = HashingEmbedder(dimension=DIMENSION)
        counter = WordTokenCounter()
        document = indexed["MSA"]

        def chunk_count() -> int:
            with connection.cursor() as cursor:
                cursor.execute("SELECT count(*) FROM chunks WHERE document_id = 'MSA'")
                (count,) = cursor.fetchone()
                return int(count)

        before = chunk_count()
        finer = chunk_document(
            document, ChunkingConfig(max_tokens=15, min_tokens=0, overlap_tokens=0), counter
        )
        index_document(
            connection, document, finer, embedder.embed_passages([c.text for c in finer])
        )
        after = chunk_count()

        assert after == len(finer)
        assert after != before, "expected a different chunk count at a finer max_tokens"


class TestLongQueryRegression:
    """Guards the AND-versus-OR defect in the lexical arm.

    An earlier version used ``websearch_to_tsquery``, which conjoins terms. A
    sentence-length query then matched zero chunks, the lexical list came back
    empty, and fusion silently ran on one arm -- so "hybrid" retrieval degraded
    to dense-only for exactly the queries the agent generates. Nothing raised.
    """

    LONG_QUERY = (
        "which state law governs the interpretation of this agreement "
        "and where must disputes be brought"
    )

    def test_long_natural_language_query_returns_lexical_hits(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        hits = lexical_search(connection, self.LONG_QUERY, limit=10)
        assert hits, "sentence-length query must not return an empty lexical list"

    def test_and_semantics_would_have_returned_nothing(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        """Pins the reason the change was needed, against the same corpus."""
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM chunks WHERE fts @@ websearch_to_tsquery('english', %s)",
                (self.LONG_QUERY,),
            )
            (conjunctive,) = cursor.fetchone()
        assert conjunctive == 0
        assert len(lexical_search(connection, self.LONG_QUERY, limit=10)) > 0

    def test_both_arms_contribute_on_a_long_query(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        retriever = HybridRetriever(
            connection=connection,
            embedder=HashingEmbedder(dimension=DIMENSION),
            candidates_per_retriever=20,
        )
        hits = retriever.search(self.LONG_QUERY, top_k=10)
        arms_seen = {arm for hit in hits for arm in hit.ranks}
        assert arms_seen == {LEXICAL, DENSE}

    def test_partial_match_still_ranks_by_coverage(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        """OR admits weak matches; ts_rank_cd must still put the best on top."""
        hits = lexical_search(connection, "governing law Delaware conflict", limit=10)
        assert hits
        assert "Delaware" in hits[0].chunk.text

    def test_query_of_only_stop_words_does_not_raise(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        """'english' strips these to an empty tsquery, which matches nothing."""
        assert lexical_search(connection, "the a of and to", limit=5) == []

    def test_punctuation_only_query_skips_the_round_trip(
        self, connection: Any, indexed: dict[str, Document]
    ) -> None:
        assert lexical_search(connection, "!!! ???", limit=5) == []


class TestSchemaGuards:
    def test_rejects_a_non_integer_dimension(self, connection: Any) -> None:
        """The dimension is interpolated into DDL, since Postgres cannot bind a
        type modifier. The type check is the thing standing in for binding."""
        with pytest.raises(ValueError, match="embedding_dim must be a positive int"):
            create_tables(connection, embedding_dim="768; DROP TABLE chunks")  # type: ignore[arg-type]

    def test_rejects_a_non_positive_dimension(self, connection: Any) -> None:
        with pytest.raises(ValueError, match="embedding_dim must be a positive int"):
            create_tables(connection, embedding_dim=0)
