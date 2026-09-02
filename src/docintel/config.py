"""Environment-driven settings.

Every tunable that affects a measured number lives here rather than as a default
buried in a function signature, so an eval run can record the configuration that
produced it. Chunk size in particular is a retrieval hyperparameter, not a
constant.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingSettings(BaseSettings):
    """Structure-aware chunking parameters.

    Defaults are set from the corpus profile in ``docs/DATA_AUDIT.md``: the
    median contract is 6,751 tokens with a median of 25 numbered headings, which
    puts a typical section in the low hundreds of tokens.
    """

    model_config = SettingsConfigDict(env_prefix="DOCINTEL_CHUNK_")

    #: Hard ceiling. Sections above this are split further, on paragraph
    #: boundaries where possible.
    max_tokens: int = Field(default=512, gt=0)
    #: Chunks below this are merged forward into their neighbour. Prevents a
    #: heading line from becoming its own useless chunk.
    min_tokens: int = Field(default=64, ge=0)
    #: Overlap applied only when a single paragraph must be split mid-flow, so a
    #: clause straddling the cut is still fully present in one of the pieces.
    overlap_tokens: int = Field(default=64, ge=0)


class RetrievalSettings(BaseSettings):
    """Hybrid retrieval parameters."""

    model_config = SettingsConfigDict(env_prefix="DOCINTEL_RETRIEVAL_")

    #: Candidates drawn from each retriever before fusion.
    candidates_per_retriever: int = Field(default=50, gt=0)
    #: Results returned after fusion and reranking.
    top_k: int = Field(default=10, gt=0)
    #: Reciprocal rank fusion's smoothing constant. 60 is the value from
    #: Cormack et al. (2009); it is a tunable, not a law, and is recorded here so
    #: an ablation can move it.
    rrf_k: int = Field(default=60, gt=0)
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    #: bge-base-en-v1.5 output dimensionality. Must match the pgvector column.
    embedding_dim: int = Field(default=768, gt=0)


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: Port 5433 matches docker-compose, which deliberately avoids 5432 so a
    #: natively installed Postgres cannot shadow the container.
    database_url: PostgresDsn = Field(
        default=PostgresDsn("postgresql://docintel:docintel@localhost:5433/docintel")
    )

    #: Read from plain ``ANTHROPIC_API_KEY`` rather than the ``DOCINTEL_`` prefix,
    #: because that is the name the Anthropic SDK itself documents and the name a
    #: CI secret will already be under.
    #:
    #: This field exists so a key in ``.env`` actually works. pydantic-settings
    #: reads ``.env`` into *this model*, not into ``os.environ``, so the SDK's own
    #: environment lookup never sees it -- a key in ``.env`` silently did nothing
    #: before, while ``.env.example`` advertised it.
    #:
    #: SecretStr so it cannot leak through a repr, a log line, or a serialized
    #: settings object.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read from the environment once."""
    return Settings()


def resolve_anthropic_api_key() -> str | None:
    """The API key, from ``.env`` or the environment, or None.

    None is not an error here: the SDK also resolves ``ANTHROPIC_AUTH_TOKEN`` and
    an ``ant auth login`` profile on its own, so a caller should fall back to a
    bare client rather than refusing.
    """
    key = get_settings().anthropic_api_key
    return key.get_secret_value() if key is not None else None
