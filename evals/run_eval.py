"""Eval entrypoint: run the baseline ladder, print the table, gate CI.

    uv run python -m evals.run_eval --baselines 1_regex               # free
    uv run python -m evals.run_eval --baselines 4_rag_rerank_agent --limit 20
    uv run python -m evals.run_eval --gate                            # CI, $0

Three modes, and the distinction matters:

**Normal** runs the selected baselines, scores them, prints the table.

**``--freeze``** additionally writes each result to ``evals/baselines/`` as the
committed reference. Only done deliberately, after a run whose numbers are
trusted.

**``--gate``** replays every baseline from cache in read-only mode, compares
against the frozen reference, and exits non-zero on a regression. This is what
runs on a pull request: it costs nothing, needs no API key, and a cache miss is
a hard error rather than a silent live call.

The gate thresholds come from ``CLAUDE.md`` section 6: presence F1 may not drop
more than 2 points against the frozen baseline, and the grounding violation rate
may not exceed 1%.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from docintel.agent.loop import DEFAULT_MODEL, AgentBudget
from docintel.config import get_settings, resolve_anthropic_api_key
from docintel.ingest.loader import iter_documents
from docintel.schemas import ClauseExtraction, Document, RetrievalHit, Tier
from evals.cache import CacheMode, CachingClient, ResponseCache, cache_from_env
from evals.cases import DEFAULT_CASES_PATH, GoldenCase, cases_by_document, load_cases
from evals.metrics import MetricSummary, precision_at_recall, score_cases
from evals.runners import (
    BASELINE_NAMES,
    TRUNCATE_TOKENS,
    AgentBaseline,
    Baseline,
    BaselineOutcome,
    RegexBaseline,
    SingleCallBaseline,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
FROZEN_DIR: Final = REPO_ROOT / "evals" / "baselines"
DEFAULT_CUAD_DIR: Final = REPO_ROOT / "data" / "raw" / "CUAD_v1"

#: CLAUDE.md section 6. A drop larger than this fails the build.
MAX_F1_DROP: Final = 0.02
#: CLAUDE.md sections 6 and 13.
MAX_GROUNDING_VIOLATION_RATE: Final = 0.01

#: Baselines that need no API key and no database.
FREE_BASELINES: Final = frozenset({"1_regex"})
#: Baselines that need Postgres and the embedder.
RETRIEVAL_BASELINES: Final = frozenset({"3_rag_no_rerank", "4_rag_rerank_agent"})


@dataclass(slots=True)
class _Retrieval:
    """Lazily-constructed retrieval stack, shared across agent baselines."""

    connection: Any = None
    retriever: Any = None
    #: Constructed on first use, then reused. Model load dominates reranking
    #: cost -- M2 measured ~47s for three candidates when the model was built
    #: per call -- so one instance has to serve the whole run.
    reranker: Any = None

    def search(
        self, document_id: str, query: str, top_k: int, *, rerank: bool
    ) -> list[RetrievalHit]:
        hits = list(self.retriever.search(query, top_k=top_k, document_ids=[document_id]))
        if not rerank or not hits:
            return hits

        from docintel.retrieval.rerank import BgeReranker
        from docintel.retrieval.rerank import rerank as apply_rerank

        if self.reranker is None:
            self.reranker = BgeReranker()
        return apply_rerank(self.reranker, query, hits, top_k=top_k)

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()


def _open_retrieval(fake_embeddings: bool) -> _Retrieval:
    import psycopg
    from pgvector.psycopg import register_vector

    from docintel.retrieval.embed import BgeEmbedder, HashingEmbedder
    from docintel.retrieval.hybrid import HybridRetriever

    settings = get_settings()
    connection = psycopg.connect(str(settings.database_url))
    register_vector(connection)
    embedder = (
        HashingEmbedder(dimension=settings.retrieval.embedding_dim)
        if fake_embeddings
        else BgeEmbedder()
    )
    return _Retrieval(
        connection=connection,
        retriever=HybridRetriever(
            connection=connection,
            embedder=embedder,
            candidates_per_retriever=settings.retrieval.candidates_per_retriever,
            rrf_k=settings.retrieval.rrf_k,
        ),
    )


def _build_client(cache_mode: CacheMode) -> tuple[CachingClient, ResponseCache]:
    """A caching client. A live client is only constructed if one is needed."""
    cache = ResponseCache(mode=cache_mode) if cache_mode else cache_from_env()
    live: Any = None
    if cache_mode is not CacheMode.READ_ONLY:
        import anthropic

        api_key = resolve_anthropic_api_key()
        live = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    return CachingClient(client=live, cache=cache), cache


def make_baseline(
    name: str,
    client: CachingClient,
    retrieval: _Retrieval | None,
    budget: AgentBudget,
    model: str,
    prompt_name: str,
) -> Baseline:
    """Construct one baseline by name."""
    if name == "1_regex":
        return RegexBaseline()
    if name == "2_zeroshot_truncated":
        return SingleCallBaseline(
            name=name, client=client, truncate_tokens=TRUNCATE_TOKENS, model=model
        )
    if name == "5_full_context":
        return SingleCallBaseline(name=name, client=client, truncate_tokens=None, model=model)
    if retrieval is None:
        raise RuntimeError(f"{name} needs retrieval, which was not opened")
    rerank = name == "4_rag_rerank_agent"
    return AgentBaseline(
        name=name,
        client=client,
        search=lambda doc_id, query, top_k: retrieval.search(doc_id, query, top_k, rerank=rerank),
        budget=budget,
        model=model,
        prompt_name=prompt_name,
    )


def run_baseline(
    baseline: Baseline,
    documents: Sequence[Document],
    cases: Sequence[GoldenCase],
    budget_usd: float,
    verbose: bool = True,
) -> MetricSummary:
    """Run one baseline over the documents and score it against the cases."""
    predictions: dict[str, list[ClauseExtraction]] = {}
    outcomes: list[BaselineOutcome] = []
    spent = 0.0

    for index, document in enumerate(documents, start=1):
        if spent >= budget_usd:
            print(
                f"  stopping {baseline.name}: spent ${spent:.4f} of ${budget_usd:.2f}",
                flush=True,
            )
            break
        outcome = baseline.run(document)
        spent += outcome.usage.cost_usd
        predictions[document.document_id] = outcome.clauses
        outcomes.append(outcome)
        if verbose:
            present = sum(1 for c in outcome.clauses if c.present)
            model_ms = outcome.latency_ms - outcome.retrieval_ms - outcome.validation_ms
            print(
                f"  [{index:>3}/{len(documents)}] {document.document_id[:40]:<40} "
                f"{present:>2}/12 {outcome.turns:>2}t "
                f"${outcome.usage.cost_usd:.4f}  "
                f"tot {outcome.latency_ms / 1000:>5.1f}s "
                f"(model {model_ms / 1000:>5.1f} retr {outcome.retrieval_ms / 1000:>4.1f}"
                f"/{outcome.search_calls}c val {outcome.validation_ms:>4.0f}ms)"
                + ("" if outcome.error is None else f"  ! {outcome.error[:36]}"),
                flush=True,
            )

    scoreable = {o.document_id for o in outcomes if o.is_scoreable}
    summary = score_cases(baseline.name, cases, predictions, scoreable=scoreable)
    summary.attempted = len(outcomes)
    summary.completed = len(scoreable)
    summary.excluded = {o.document_id: o.status.value for o in outcomes if not o.is_scoreable}
    for outcome in outcomes:
        summary.costs_usd.append(outcome.usage.cost_usd)
        summary.latencies_ms.append(outcome.latency_ms)
        summary.retrieval_ms.append(outcome.retrieval_ms)
        summary.validation_ms.append(outcome.validation_ms)
        if not outcome.is_scoreable:
            continue
        summary.schema_attempts += 1
        summary.schema_first_try += int(outcome.schema_ok)
        summary.documents_with_errors += int(outcome.has_errors)
        if outcome.grounding is not None:
            summary.grounding_checked += outcome.grounding.total
            summary.grounding_violations += len(outcome.grounding.ungrounded)
    return summary


def print_table(summaries: Sequence[MetricSummary]) -> None:
    """The results table. Cost and latency sit beside every accuracy figure."""
    if not summaries:
        return
    header = (
        f"{'baseline':<24}{'done':>10}{'F1':>7}{'triv':>7}{'P':>7}{'R':>7}"
        f"{'spanF1':>8}{'ground':>8}{'schema':>8}{'rules':>7}"
        f"{'$/doc':>9}{'$p95':>9}{'p50 ms':>9}{'p95 ms':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for summary in summaries:
        overall = summary.presence.overall
        done = f"{summary.completed}/{summary.attempted}"
        print(
            f"{summary.baseline:<24}{done:>10}"
            f"{overall.f1:>7.3f}{overall.trivial_f1:>7.3f}"
            f"{overall.precision:>7.3f}{overall.recall:>7.3f}"
            f"{summary.spans.mean:>8.3f}"
            f"{summary.grounding_violation_rate:>8.2%}"
            f"{summary.schema_validity_rate:>8.0%}"
            f"{summary.rule_violation_rate:>7.0%}"
            f"{summary.mean_cost_usd:>9.4f}{summary.p95_cost_usd:>9.4f}"
            f"{summary.p50_latency_ms:>9.0f}{summary.p95_latency_ms:>9.0f}"
        )

    print("\nby tier (presence F1 / base rate / n):")
    tiers = [Tier.NEAR_DETERMINISTIC, Tier.REQUIRES_READING, Tier.REQUIRES_JUDGEMENT]
    print(f"  {'baseline':<24}" + "".join(f"{'tier ' + str(int(t)):>22}" for t in tiers))
    for summary in summaries:
        cells = []
        for tier in tiers:
            counts = summary.presence.by_tier.get(tier)
            cells.append(
                f"{counts.f1:>8.3f} /{counts.base_rate:>5.2f} /{counts.total:>4}"
                if counts
                else f"{'-':>22}"
            )
        print(f"  {summary.baseline:<24}" + "".join(cells))

    print("\nprecision at 80% recall (CUAD convention):")
    for summary in summaries:
        result = precision_at_recall(summary.confidence_pairs, 0.80)
        if result is None:
            print(f"  {summary.baseline:<24} 80% recall unreachable at any threshold")
        else:
            precision, threshold = result
            print(f"  {summary.baseline:<24} P={precision:.3f} at confidence >= {threshold:.2f}")


def freeze(summary: MetricSummary, directory: Path = FROZEN_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{summary.baseline}.json"
    path.write_text(json.dumps(summary.to_json(), indent=2) + "\n", encoding="utf-8")
    return path


def gate(summaries: Sequence[MetricSummary], directory: Path = FROZEN_DIR) -> int:
    """Compare against frozen baselines. Non-zero exit means the build fails."""
    failures: list[str] = []
    checked = 0

    for summary in summaries:
        path = directory / f"{summary.baseline}.json"
        if not path.exists():
            print(f"  {summary.baseline}: no frozen baseline, skipping")
            continue
        checked += 1
        frozen = json.loads(path.read_text(encoding="utf-8"))
        was = float(frozen["presence"]["f1"])
        now = summary.presence.overall.f1
        drop = was - now
        status = "ok" if drop <= MAX_F1_DROP else "FAIL"
        print(f"  {summary.baseline:<24} F1 {was:.3f} -> {now:.3f} ({-drop:+.3f})  [{status}]")
        if drop > MAX_F1_DROP:
            failures.append(
                f"{summary.baseline}: presence F1 dropped {drop:.3f}, limit {MAX_F1_DROP:.2f}"
            )

        rate = summary.grounding_violation_rate
        if rate > MAX_GROUNDING_VIOLATION_RATE:
            failures.append(
                f"{summary.baseline}: grounding violation rate {rate:.2%} "
                f"exceeds {MAX_GROUNDING_VIOLATION_RATE:.0%}"
            )

    if not checked:
        print("no frozen baselines to compare against; nothing gated")
        return 0
    if failures:
        print("\nGATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\ngate passed")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="evals.run_eval", description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--cuad-dir", type=Path, default=DEFAULT_CUAD_DIR)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(BASELINE_NAMES),
        choices=list(BASELINE_NAMES),
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap the number of contracts.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="extract_v1")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-cost", type=float, default=1.50, help="Per document.")
    parser.add_argument("--budget", type=float, default=5.00, help="Per baseline, in USD.")
    parser.add_argument("--cache-mode", choices=[m.value for m in CacheMode], default=None)
    parser.add_argument("--fake-embeddings", action="store_true")
    parser.add_argument("--freeze", action="store_true", help="Write results to evals/baselines/.")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Replay from cache read-only and fail on regression. What CI runs.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    cases = load_cases(args.cases)
    grouped = cases_by_document(cases)
    document_ids = set(grouped)

    documents = list(iter_documents(args.cuad_dir, document_ids))

    if args.sample:
        # Sorted before sampling: random.sample is order-sensitive, so an
        # unsorted input would give a different subset per run even at a fixed
        # seed. Same trap as the split builder in scripts/build_split.py.
        ordered = sorted(documents, key=lambda d: d.document_id)
        chosen = random.Random(args.sample_seed).sample(ordered, k=min(args.sample, len(ordered)))
        documents = sorted(chosen, key=lambda d: d.document_id)[args.skip :]

    if args.limit:
        documents = documents[: args.limit]
        kept = {doc.document_id for doc in documents}
        cases = [case for case in cases if case.document_id in kept]

    print(
        f"{len(cases)} cases over {len(documents)} contracts | "
        f"baselines: {', '.join(args.baselines)}"
        + (f" | sample seed {args.sample_seed}, skip {args.skip}" if args.sample else "")
    )

    # The gate never calls the API: read-only cache, and a miss is an error.
    mode = (
        CacheMode.READ_ONLY
        if args.gate
        else (CacheMode(args.cache_mode) if args.cache_mode else None)
    )
    needs_client = any(name not in FREE_BASELINES for name in args.baselines)
    client, cache = (
        _build_client(mode or CacheMode.READ_WRITE)
        if needs_client
        else (CachingClient(client=None, cache=ResponseCache(mode=CacheMode.OFF)), None)
    )

    retrieval: _Retrieval | None = None
    if any(name in RETRIEVAL_BASELINES for name in args.baselines):
        retrieval = _open_retrieval(args.fake_embeddings)

    budget = AgentBudget(max_turns=args.max_turns, max_cost_usd=args.max_cost)
    summaries: list[MetricSummary] = []

    try:
        for name in args.baselines:
            print(f"\n=== {name} ===", flush=True)
            started = time.perf_counter()
            baseline = make_baseline(name, client, retrieval, budget, args.model, args.prompt)
            summary = run_baseline(baseline, documents, cases, args.budget, verbose=not args.quiet)
            summaries.append(summary)
            print(
                f"  {name}: ${summary.total_cost_usd:.4f} in {time.perf_counter() - started:.0f}s",
                flush=True,
            )
    finally:
        if retrieval is not None:
            retrieval.close()

    print_table(summaries)
    if cache is not None:
        print(f"\ncache: {cache.stats}")

    if args.freeze:
        for summary in summaries:
            print(f"froze {freeze(summary)}")

    if args.gate:
        print("\n=== gate ===")
        return gate(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
