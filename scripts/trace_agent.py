"""Replay one extraction from the response cache and print its tool trace.

Costs nothing. Every model response is already recorded, so the loop replays
deterministically and the point is to see *what the agent did*, not to spend
again finding out.

    uv run python scripts/trace_agent.py --document ALLIANCEBANCORP

Prints turn by turn: tool, whether the call errored, the argument that matters
(query text or span), and a running tally of which clause types have been
searched for. That is enough to separate the three ways a loop dies:

1. re-searching the same clause with near-identical queries
2. stuck reading spans, expanding context on one clause
3. silent tool errors treated as retryable
"""

from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

# `evals` is a package at the repo root but is not installed; pytest gets it from
# `pythonpath` in pyproject, a bare script has to add it itself.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.cache import CacheMode, CachingClient, ResponseCache

from docintel.agent.loop import AgentBudget, run_extraction
from docintel.agent.tools import ToolContext
from docintel.config import get_settings
from docintel.ingest.loader import iter_documents
from docintel.schemas import ClauseType

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_CUAD_DIR: Final = REPO_ROOT / "data" / "raw" / "CUAD_v1"

#: Two queries this similar are the same search reworded.
NEAR_DUPLICATE: Final = 0.80


def _clause_hits(query: str) -> set[str]:
    """Which clause types a query plausibly targets, by keyword overlap."""
    lowered = query.lower()
    keywords = {
        ClauseType.GOVERNING_LAW: ("governing", "laws of", "construed", "jurisdiction"),
        ClauseType.PARTIES: ("parties", "between", "entered into by"),
        ClauseType.EFFECTIVE_DATE: ("effective date", "commences", "dated"),
        ClauseType.EXPIRATION_DATE: ("expire", "expiration", "term of this", "initial term"),
        ClauseType.RENEWAL_TERM: ("renew", "extension", "successive"),
        ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL: ("notice", "terminate renewal"),
        ClauseType.CHANGE_OF_CONTROL: ("change of control", "merger", "stock sale"),
        ClauseType.ANTI_ASSIGNMENT: ("assign", "assignment", "consent"),
        ClauseType.CAP_ON_LIABILITY: ("cap", "limitation of liability", "aggregate liability"),
        ClauseType.UNCAPPED_LIABILITY: ("unlimited", "uncapped", "not be limited"),
        ClauseType.NON_COMPETE: ("compete", "non-compete", "restrict"),
        ClauseType.EXCLUSIVITY: ("exclusive", "requirements", "sole supplier"),
    }
    return {
        clause.value for clause, words in keywords.items() if any(word in lowered for word in words)
    }


def trace(document_id: str, cuad_dir: Path, max_turns: int, prompt: str) -> int:
    settings = get_settings()

    import psycopg
    from pgvector.psycopg import register_vector

    from docintel.retrieval.embed import BgeEmbedder
    from docintel.retrieval.hybrid import HybridRetriever
    from docintel.retrieval.rerank import BgeReranker
    from docintel.retrieval.rerank import rerank as apply_rerank

    matches = [
        doc for doc in iter_documents(cuad_dir) if document_id.lower() in doc.document_id.lower()
    ]
    if not matches:
        print(f"no contract matching {document_id!r}")
        return 2
    document = matches[0]
    print(f"contract: {document.document_id}\nchars: {len(document.text):,}\n")

    connection = psycopg.connect(str(settings.database_url))
    register_vector(connection)
    retriever = HybridRetriever(
        connection=connection,
        embedder=BgeEmbedder(),
        candidates_per_retriever=settings.retrieval.candidates_per_retriever,
        rrf_k=settings.retrieval.rrf_k,
    )
    reranker = BgeReranker()

    def search(query: str, top_k: int) -> Sequence[Any]:
        hits = list(retriever.search(query, top_k=top_k, document_ids=[document.document_id]))
        return apply_rerank(reranker, query, hits, top_k=top_k) if hits else hits

    # READ_ONLY: a miss raises rather than costing money. If this run diverges
    # from the recorded one it fails loudly instead of quietly billing.
    cache = ResponseCache(mode=CacheMode.READ_ONLY)
    client = CachingClient(client=None, cache=cache)
    ctx = ToolContext(document=document, search=search)

    try:
        outcome = run_extraction(
            client,
            ctx,
            budget=AgentBudget(max_turns=max_turns),
            prompt_name=prompt,
        )
    finally:
        connection.close()

    print(
        f"stop_reason: {outcome.stop_reason} | turns: {outcome.turns} | "
        f"retries: {outcome.retries} | clauses submitted: {len(outcome.clauses)}"
    )
    print(f"cache: {cache.stats}\n")

    queries: list[str] = []
    covered: set[str] = set()
    reads: list[tuple[int, int]] = []

    print(f"{'turn':>4} {'tool':<18} {'err':<4} detail")
    print("-" * 100)
    for call in ctx.calls:
        turn = call.get("turn", 0)
        tool = str(call.get("tool"))
        errored = "ERR" if call.get("is_error") else ""
        args = call.get("input") or {}
        detail = ""

        if tool == "search_contract":
            query = str(args.get("query", ""))
            near = [
                (round(difflib.SequenceMatcher(None, query, prior).ratio(), 2), index)
                for index, prior in enumerate(queries, start=1)
            ]
            worst = max(near, default=(0.0, 0))
            queries.append(query)
            hits = _clause_hits(query)
            new = hits - covered
            covered |= hits
            flag = (
                f"  <-- {worst[0]:.2f} similar to #{worst[1]}" if worst[0] >= NEAR_DUPLICATE else ""
            )
            detail = f"{query[:62]!r} targets={sorted(hits) or ['?']} new={sorted(new) or []}{flag}"
        elif tool == "read_span":
            span = (int(args.get("char_start", 0)), int(args.get("char_end", 0)))
            repeat = (
                "  <-- OVERLAPS an earlier read"
                if any(span[0] < end and start < span[1] for start, end in reads)
                else ""
            )
            reads.append(span)
            detail = f"chars {span[0]}-{span[1]} ({span[1] - span[0]}){repeat}"
        elif tool == "get_schema":
            detail = f"clause_types={args.get('clause_types') or 'ALL'}"
        elif tool == "submit_extraction":
            detail = f"{len(args.get('clauses') or [])} clauses"
        if call.get("is_error"):
            detail += f"  |  {str(call.get('result'))[:90]}"
        print(f"{turn:>4} {tool:<18} {errored:<4} {detail}")

    print("\n--- summary ---")
    print(f"  search calls        : {sum(1 for c in ctx.calls if c['tool'] == 'search_contract')}")
    print(f"  read_span calls     : {sum(1 for c in ctx.calls if c['tool'] == 'read_span')}")
    print(f"  get_schema calls    : {sum(1 for c in ctx.calls if c['tool'] == 'get_schema')}")
    print(
        f"  submit attempts     : {sum(1 for c in ctx.calls if c['tool'] == 'submit_extraction')}"
    )
    print(f"  errored calls       : {sum(1 for c in ctx.calls if c.get('is_error'))}")
    duplicates = sum(
        1
        for index, query in enumerate(queries)
        if any(
            difflib.SequenceMatcher(None, query, prior).ratio() >= NEAR_DUPLICATE
            for prior in queries[:index]
        )
    )
    print(f"  near-duplicate queries: {duplicates}/{len(queries)}")
    print(f"  clause types targeted : {len(covered)}/12  {sorted(covered)}")
    print(f"  never targeted        : {sorted({c.value for c in ClauseType} - covered)}")
    calls_per_turn = len(ctx.calls) / outcome.turns if outcome.turns else 0
    print(f"  calls per turn        : {calls_per_turn:.1f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", required=True, help="Substring of the contract id.")
    parser.add_argument("--cuad-dir", type=Path, default=DEFAULT_CUAD_DIR)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--prompt", default="extract_v1")
    args = parser.parse_args(argv)
    return trace(args.document, args.cuad_dir, args.max_turns, args.prompt)


if __name__ == "__main__":
    raise SystemExit(main())
