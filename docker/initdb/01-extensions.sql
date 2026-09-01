-- Executed once by the postgres entrypoint against an empty data directory.
CREATE EXTENSION IF NOT EXISTS vector;      -- dense retrieval (pgvector)
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram similarity for fuzzy clause lookup
