"""Command line entry points for ingestion and retrieval.

    uv run python -m docintel.cli index  --split dev --limit 20
    uv run python -m docintel.cli search "who are the parties to this agreement"
    uv run python -m docintel.cli search "liability cap" --arm lexical --no-rerank
    uv run python -m docintel.cli extract --split dev --limit 10

``search`` prints the per-retriever ranks alongside each hit, because the
interesting question about hybrid retrieval is which arm found a chunk, and that
is invisible in a plain ranked list.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from evals.cache import DEFAULT_CACHE_DIR, CacheMode, CachingClient, ResponseCache

from docintel.agent.loop import DEFAULT_MODEL, AgentBudget
from docintel.config import get_settings, resolve_anthropic_api_key
from docintel.extract import extract_document
from docintel.ingest.chunker import ChunkingConfig, TiktokenCounter, chunk_document
from docintel.ingest.index import create_tables, create_vector_index, index_document
from docintel.ingest.loader import iter_documents, load_split
from docintel.retrieval.embed import BgeEmbedder, Embedder, HashingEmbedder
from docintel.retrieval.hybrid import DENSE, LEXICAL, HybridRetriever
from docintel.retrieval.rerank import BgeReranker, rerank

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUAD_DIR = REPO_ROOT / "data" / "raw" / "CUAD_v1"
DEFAULT_SPLIT = REPO_ROOT / "evals" / "golden" / "split.json"


class _Connection(Protocol):
    """The slice of the psycopg connection API this module uses.

    Declared rather than imported so `index.py` and `hybrid.py` stay driver-
    agnostic in their annotations, and so the type checker still catches a typo
    in a method name.
    """

    def cursor(self) -> Any: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...


def _connect(database_url: str) -> _Connection:
    import psycopg
    from pgvector.psycopg import register_vector

    connection = psycopg.connect(database_url)
    # Registers the vector type so lists round-trip without manual casting.
    register_vector(connection)
    return connection


def _build_embedder(fake: bool, dimension: int) -> Embedder:
    if fake:
        print(
            "using HashingEmbedder -- plumbing only, retrieval quality is meaningless", flush=True
        )
        return HashingEmbedder(dimension=dimension)
    return BgeEmbedder()


def command_index(args: argparse.Namespace) -> int:
    settings = get_settings()
    embedder = _build_embedder(args.fake_embeddings, settings.retrieval.embedding_dim)

    if embedder.dimension != settings.retrieval.embedding_dim:
        print(
            f"embedder dimension {embedder.dimension} does not match configured "
            f"{settings.retrieval.embedding_dim}; the pgvector column would reject these",
            file=sys.stderr,
        )
        return 2

    document_ids: set[str] | None = None
    if args.split:
        document_ids = load_split(args.split_file, args.split)
        print(f"restricting to the {args.split} split: {len(document_ids)} contracts", flush=True)

    connection = _connect(str(settings.database_url))
    counter = TiktokenCounter()
    config = ChunkingConfig(
        max_tokens=settings.chunking.max_tokens,
        min_tokens=settings.chunking.min_tokens,
        overlap_tokens=settings.chunking.overlap_tokens,
    )

    try:
        create_tables(connection, embedder.dimension)

        documents = 0
        chunks_written = 0
        started = time.perf_counter()
        for document in iter_documents(args.cuad_dir, document_ids):
            if args.limit and documents >= args.limit:
                break
            chunks = chunk_document(document, config, counter)
            embeddings = embedder.embed_passages([chunk.text for chunk in chunks])
            index_document(connection, document, chunks, embeddings)
            documents += 1
            chunks_written += len(chunks)
            if documents % 10 == 0:
                rate = documents / (time.perf_counter() - started)
                print(
                    f"  {documents} documents, {chunks_written} chunks ({rate:.1f} docs/s)",
                    flush=True,
                )

        # Built last: HNSW construction on a populated table is much faster than
        # maintaining the index across every insert.
        print("building HNSW index...", flush=True)
        create_vector_index(connection)
        elapsed = time.perf_counter() - started
        print(f"indexed {documents} documents / {chunks_written} chunks in {elapsed:.1f}s")
    finally:
        connection.close()

    return 0


def command_search(args: argparse.Namespace) -> int:
    settings = get_settings()
    embedder = _build_embedder(args.fake_embeddings, settings.retrieval.embedding_dim)

    connection = _connect(str(settings.database_url))
    try:
        retriever = HybridRetriever(
            connection=connection,
            embedder=embedder,
            candidates_per_retriever=args.candidates,
            rrf_k=settings.retrieval.rrf_k,
        )

        arms = (LEXICAL, DENSE) if args.arm == "hybrid" else (args.arm,)
        started = time.perf_counter()
        hits = retriever.search(args.query, top_k=args.top_k, arms=arms)
        retrieval_ms = (time.perf_counter() - started) * 1000

        rerank_ms = 0.0
        if args.rerank and hits:
            started = time.perf_counter()
            hits = rerank(BgeReranker(), args.query, hits, top_k=args.top_k)
            rerank_ms = (time.perf_counter() - started) * 1000

        print(f'\nquery: "{args.query}"')
        print(f"arms: {'+'.join(arms)}  |  retrieval {retrieval_ms:.0f}ms", end="")
        print(f"  |  rerank {rerank_ms:.0f}ms" if rerank_ms else "")
        print(f"{len(hits)} hits\n")

        for position, hit in enumerate(hits, start=1):
            ranks = " ".join(f"{name}#{rank}" for name, rank in sorted(hit.ranks.items()))
            score = (
                f"rerank {hit.rerank_score:+.3f}"
                if hit.rerank_score is not None
                else f"rrf {hit.score:.5f}"
            )
            print(f"{position:2d}. {score}  [{ranks}]")
            print(f"    {hit.chunk.document_id[:70]}")
            print(f"    chars {hit.chunk.char_start}-{hit.chunk.char_end}", end="")
            print(f"  heading: {hit.chunk.heading[:50]}" if hit.chunk.heading else "")
            snippet = " ".join(hit.chunk.text.split())[:280]
            print(f"    {snippet}\n")
    finally:
        connection.close()

    return 0


def command_extract(args: argparse.Namespace) -> int:
    """Run the full pipeline over indexed contracts."""
    import anthropic

    settings = get_settings()
    embedder = _build_embedder(args.fake_embeddings, settings.retrieval.embedding_dim)

    # Passed explicitly when present, so a key in `.env` works. Falling back to
    # a bare client lets the SDK resolve ANTHROPIC_AUTH_TOKEN or an
    # `ant auth login` profile by itself.
    api_key = resolve_anthropic_api_key()
    live = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # Construction never fails on missing credentials -- the SDK resolves them
    # lazily and raises on the first request. Without this check the run would
    # iterate every document and produce one identical auth error per contract.
    if not (getattr(live, "api_key", None) or getattr(live, "auth_token", None)):
        print("no Anthropic credentials found.", file=sys.stderr)
        print(
            "Put ANTHROPIC_API_KEY in .env (gitignored), export it, or run "
            "`ant auth login` to store a profile the SDK picks up automatically.",
            file=sys.stderr,
        )
        return 2

    # Every response is cached by content hash, so a re-run after a code change
    # that does not alter the prompt costs nothing. Without this, iterating on
    # the pipeline re-pays for identical requests every time.
    cache = ResponseCache(directory=DEFAULT_CACHE_DIR, mode=CacheMode(args.cache_mode))
    client = CachingClient(client=live, cache=cache)

    document_ids: set[str] | None = None
    if args.split:
        document_ids = load_split(args.split_file, args.split)

    connection = _connect(str(settings.database_url))
    budget = AgentBudget(
        max_turns=args.max_turns,
        max_cost_usd=args.max_cost,
        timeout_s=args.timeout,
    )

    totals = {"documents": 0, "reviewed": 0, "spans": 0, "ungrounded": 0}
    spend = 0.0

    try:
        retriever = HybridRetriever(
            connection=connection,
            embedder=embedder,
            candidates_per_retriever=settings.retrieval.candidates_per_retriever,
            rrf_k=settings.retrieval.rrf_k,
        )

        for document in iter_documents(args.cuad_dir, document_ids):
            if args.limit and totals["documents"] >= args.limit:
                break
            if spend >= args.budget:
                print(f"stopping: spent ${spend:.4f} of ${args.budget:.2f}", flush=True)
                break

            # Retrieval is scoped to this document, so the agent cannot cite a
            # clause from a different contract.
            def search(query: str, top_k: int, _doc_id: str = document.document_id) -> list[Any]:
                return retriever.search(query, top_k=top_k, document_ids=[_doc_id])

            outcome = extract_document(
                client=client,
                document=document,
                search=search,
                budget=budget,
                model=args.model,
                prompt_name=args.prompt,
            )
            result = outcome.result
            spend += result.usage.cost_usd
            totals["documents"] += 1
            totals["reviewed"] += int(result.needs_review)
            totals["spans"] += outcome.grounding.total
            totals["ungrounded"] += len(outcome.grounding.ungrounded)

            present = [c for c in result.clauses if c.present]
            flag = "REVIEW" if result.needs_review else "ok"
            print(
                f"[{flag:>6}] {document.document_id[:58]:<58} "
                f"{len(present):>2}/12 present  "
                f"{result.turns_used:>2} turns  "
                f"${result.usage.cost_usd:.4f}  "
                f"{result.latency_ms.get('total', 0) / 1000:.1f}s",
                flush=True,
            )
            if args.show_turns:
                for n, turn in enumerate(outcome.agent.turn_usage, start=1):
                    print(
                        f"           t{n}: in {turn.input_tokens:>6} "
                        f"cache_r {turn.cache_read_input_tokens:>6} "
                        f"cache_w {turn.cache_creation_input_tokens:>6} "
                        f"out {turn.output_tokens:>5}  ${turn.cost_usd:.4f}",
                        flush=True,
                    )
            if outcome.review is not None:
                for reason in outcome.review.reasons:
                    print(f"           - {reason}", flush=True)
            for violation in result.errors[:3]:
                print(f"           ! {violation.rule_id}: {violation.message}", flush=True)
    finally:
        connection.close()

    if not totals["documents"]:
        print("no documents processed", file=sys.stderr)
        return 1

    rate = totals["ungrounded"] / totals["spans"] if totals["spans"] else 0.0
    print(f"cache: {cache.stats}")
    print(
        f"\n{totals['documents']} documents | "
        f"{totals['reviewed']} routed to review | "
        f"grounding violations {totals['ungrounded']}/{totals['spans']} ({rate:.2%}) | "
        f"total ${spend:.4f} (${spend / totals['documents']:.4f}/doc)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docintel", description=__doc__)
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Use HashingEmbedder instead of downloading bge. Plumbing only.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    indexer = subparsers.add_parser("index", help="Chunk, embed, and index contracts.")
    indexer.add_argument("--cuad-dir", type=Path, default=DEFAULT_CUAD_DIR)
    indexer.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    indexer.add_argument(
        "--split",
        choices=["dev", "golden", "reserve"],
        help="Restrict to one frozen split. Omit to index everything.",
    )
    indexer.add_argument("--limit", type=int, default=0, help="Stop after N documents.")
    indexer.set_defaults(handler=command_index)

    searcher = subparsers.add_parser("search", help="Retrieve chunks for a query.")
    searcher.add_argument("query")
    searcher.add_argument("--top-k", type=int, default=10)
    searcher.add_argument("--candidates", type=int, default=50)
    searcher.add_argument("--arm", choices=["hybrid", LEXICAL, DENSE], default="hybrid")
    searcher.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the cross-encoder. --no-rerank for the ablation arm.",
    )
    searcher.set_defaults(handler=command_search)

    extractor = subparsers.add_parser(
        "extract", help="Run the full extraction pipeline over indexed contracts."
    )
    extractor.add_argument("--cuad-dir", type=Path, default=DEFAULT_CUAD_DIR)
    extractor.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    extractor.add_argument("--split", choices=["dev", "golden", "reserve"], default="dev")
    extractor.add_argument("--limit", type=int, default=10)
    extractor.add_argument("--model", default=DEFAULT_MODEL)
    extractor.add_argument("--max-turns", type=int, default=12)
    extractor.add_argument("--max-cost", type=float, default=0.50, help="Per document.")
    extractor.add_argument("--timeout", type=float, default=180.0, help="Per document.")
    extractor.add_argument(
        "--budget",
        type=float,
        default=2.00,
        help="Hard ceiling for the whole run, in USD. Stops before exceeding it.",
    )
    extractor.add_argument("--prompt", default="extract_v1", help="Prompt version to use.")
    extractor.add_argument(
        "--cache-mode",
        choices=[m.value for m in CacheMode],
        default=CacheMode.READ_WRITE.value,
        help="read_write caches and reuses; read_only fails on a miss; off bypasses.",
    )
    extractor.add_argument(
        "--show-turns",
        action="store_true",
        help="Print per-turn token and cost breakdown. Use when cost surprises you.",
    )
    extractor.set_defaults(handler=command_extract)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.handler
    result: int = handler(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
