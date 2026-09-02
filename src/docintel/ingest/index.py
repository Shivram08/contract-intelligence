"""Postgres schema and index construction for documents, chunks, and vectors.

One database serves both retrieval arms: pgvector for dense search, Postgres
full-text search for lexical. That is a deliberate simplification over running a
separate vector service -- it is one container, it is transactional, and it is
what a bank actually operates.

**A naming correction.** ``CLAUDE.md`` calls the lexical arm "BM25 (Postgres
FTS)". Postgres full-text search does not implement BM25. ``ts_rank_cd`` is a
cover-density ranking over ``tsvector`` positions, with different length
normalization and no ``k1``/``b`` parameters. The code calls this retriever
``lexical`` rather than ``bm25`` so the results table does not claim an
algorithm it does not run. It matters less than it sounds: reciprocal rank
fusion consumes ranks, not scores, so the two agree on ordering far more often
than on magnitude. Where a true BM25 comparison is wanted, ``docs/RESULTS.md``
should say which one was measured.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any, Final

from docintel.schemas import Chunk, Document

__all__ = [
    "SCHEMA_SQL",
    "chunk_content_hash",
    "create_tables",
    "document_content_hash",
    "drop_tables",
    "index_document",
    "upsert_chunks",
    "upsert_document",
]

#: Postgres text search configuration. 'english' applies stemming and a stop
#: word list, which is right for prose. It does mean a defined term like
#: "Territory" stems to "territori" and matches "territories" -- usually helpful
#: in contracts, occasionally not, and worth an ablation if lexical recall
#: disappoints.
TEXT_SEARCH_CONFIG: Final = "english"

SCHEMA_SQL: Final = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    -- The full raw text lives here so the grounding verifier can resolve an
    -- evidence offset without reaching back to the filesystem. The whole corpus
    -- is 27 MB; the convenience is worth the duplication.
    text          TEXT        NOT NULL,
    source_path   TEXT,
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    content_hash  TEXT        NOT NULL,
    indexed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    document_id   TEXT    NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    text          TEXT    NOT NULL,
    -- Offsets into documents.text, not into anything normalized. Enforced here
    -- as well as in Pydantic: the database is the last line of defence for the
    -- invariant the whole evidence chain rests on.
    char_start    INTEGER NOT NULL CHECK (char_start >= 0),
    char_end      INTEGER NOT NULL CHECK (char_end >= char_start),
    heading       TEXT,
    token_count   INTEGER NOT NULL CHECK (token_count >= 0),
    embedding     vector(%(dim)s),
    -- Generated rather than trigger-maintained, so the tsvector cannot drift
    -- out of sync with the text it indexes.
    fts           tsvector GENERATED ALWAYS AS (
                      to_tsvector('%(config)s', text)
                  ) STORED,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_fts_idx      ON chunks USING GIN (fts);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
"""

#: Built separately from the tables: HNSW construction is the slow part, and on
#: a bulk load it is much faster to insert first and index after.
VECTOR_INDEX_SQL: Final = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def document_content_hash(document: Document) -> str:
    """Content hash of a document, so re-indexing can skip unchanged files."""
    return hashlib.sha256(document.text.encode("utf-8")).hexdigest()


def chunk_content_hash(chunks: Sequence[Chunk], embedding_model: str) -> str:
    """Hash identifying a chunking+embedding configuration's output.

    Includes the model name because the same chunks embedded by a different
    model are not interchangeable, and a stale mix of the two produces a vector
    index that returns nonsense without erroring.
    """
    digest = hashlib.sha256()
    digest.update(embedding_model.encode("utf-8"))
    for chunk in chunks:
        digest.update(f"{chunk.chunk_id}:{chunk.char_start}:{chunk.char_end}".encode())
    return digest.hexdigest()


def create_tables(connection: Any, embedding_dim: int = 768) -> None:
    """Create tables and the lexical index. Idempotent.

    The DDL is assembled by string formatting rather than parameter binding
    because Postgres does not accept bind parameters in a type modifier --
    ``vector($1)`` is a syntax error. The two interpolated values are safe by
    construction: ``embedding_dim`` is typed ``int`` and validated ``gt=0`` by
    ``RetrievalSettings``, and ``TEXT_SEARCH_CONFIG`` is a module constant. No
    caller-supplied string reaches this template.
    """
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be a positive int, got {embedding_dim!r}")
    with connection.cursor() as cursor:
        cursor.execute(SCHEMA_SQL % {"dim": embedding_dim, "config": TEXT_SEARCH_CONFIG})
    connection.commit()


def create_vector_index(connection: Any) -> None:
    """Build the HNSW index. Call after bulk loading, not before."""
    with connection.cursor() as cursor:
        cursor.execute(VECTOR_INDEX_SQL)
    connection.commit()


def drop_tables(connection: Any) -> None:
    """Drop both tables, in the connection's current schema.

    Named ``drop_tables`` rather than ``drop_schema`` deliberately. It drops
    tables by unqualified name, which Postgres resolves through ``search_path``
    -- so it is *not* a safe isolation mechanism. An earlier version of the
    integration tests tried to sandbox themselves by prepending a scratch schema
    to ``search_path``; once a test had dropped the scratch copy, the next
    unqualified ``DROP TABLE IF EXISTS chunks`` fell through the path and deleted
    the real one. Isolate by connecting to a separate database instead.
    """
    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS chunks CASCADE")
        cursor.execute("DROP TABLE IF EXISTS documents CASCADE")
    connection.commit()


def upsert_document(connection: Any, document: Document) -> None:
    """Insert or replace a document row."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (document_id, text, source_path, metadata, content_hash)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE SET
                text         = EXCLUDED.text,
                source_path  = EXCLUDED.source_path,
                metadata     = EXCLUDED.metadata,
                content_hash = EXCLUDED.content_hash,
                indexed_at   = now()
            """,
            (
                document.document_id,
                document.text,
                document.source_path,
                json.dumps(document.metadata),
                document_content_hash(document),
            ),
        )


def upsert_chunks(
    connection: Any,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]] | None = None,
) -> None:
    """Replace a document's chunks, with optional embeddings.

    Deletes the document's existing chunks first. Re-chunking with different
    parameters produces different ``ordinal`` values, so upserting by id would
    leave orphaned chunks from the previous configuration in the index --
    invisible until they start showing up in results.
    """
    if not chunks:
        return
    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError(f"got {len(embeddings)} embeddings for {len(chunks)} chunks")

    document_ids = {chunk.document_id for chunk in chunks}
    if len(document_ids) != 1:
        raise ValueError(f"expected chunks from exactly one document, got {len(document_ids)}")

    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM chunks WHERE document_id = %s", (document_ids.pop(),))
        cursor.executemany(
            """
            INSERT INTO chunks (
                chunk_id, document_id, ordinal, text,
                char_start, char_end, heading, token_count, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.ordinal,
                    chunk.text,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.heading,
                    chunk.token_count,
                    list(embeddings[position]) if embeddings is not None else None,
                )
                for position, chunk in enumerate(chunks)
            ],
        )


def index_document(
    connection: Any,
    document: Document,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]] | None = None,
) -> None:
    """Index one document and its chunks in a single transaction.

    Atomic on purpose: a document row with a half-written chunk set is worse
    than no document row, because retrieval would silently return partial
    coverage rather than failing.
    """
    try:
        upsert_document(connection, document)
        upsert_chunks(connection, chunks, embeddings)
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def unindexed_document_ids(connection: Any, documents: Iterable[Document]) -> list[str]:
    """Which of these documents are missing or have changed since last indexed."""
    candidates = {document.document_id: document_content_hash(document) for document in documents}
    if not candidates:
        return []

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT document_id, content_hash FROM documents WHERE document_id = ANY(%s)",
            (list(candidates),),
        )
        stored = dict(cursor.fetchall())

    return sorted(
        document_id
        for document_id, content_hash in candidates.items()
        if stored.get(document_id) != content_hash
    )
