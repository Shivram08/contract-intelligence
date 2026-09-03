"""Retrieval ablation, scored against CUAD gold spans. No API calls.

Measures dense-only against hybrid against hybrid+rerank on recall@k, MRR, and
nDCG, using the gold spans as relevance labels. Runs on the **full 150-case
golden set** because it costs nothing -- embeddings and the cross-encoder are
local -- so this is the one comparison in the project that does not trade
statistical power against budget.

**Why measure the rerank effect here rather than end-to-end.** ``CLAUDE.md``
section 6's baseline 3 is "RAG top-k, no rerank", scored through the agent. That
routes the retrieval signal through a second noisy generation stage, which
dilutes it: an improvement in passage ranking has to survive the model's
decisions before it shows up in presence F1. Scoring at the retrieval layer
isolates the thing being ablated, and it is better evidence for the same claim.

**Relevance labels.** A retrieved chunk is relevant if it overlaps any gold span
for that clause type. Binary, not graded, because CUAD supplies spans rather
than relevance judgements -- so nDCG here uses binary gains and is a
rank-position measure rather than a graded-relevance one. Stated because nDCG
usually implies graded labels and it does not here.

**The query.** One query per clause type, taken from CUAD's own category
description rather than invented, so the ablation is not measuring a
prompt-engineering choice made after seeing results.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.config import get_settings
from docintel.schemas import CLAUSE_DEFINITIONS, RetrievalHit
from evals.cases import GoldenCase, cases_by_document, load_cases

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final = REPO_ROOT / "evals" / "baselines" / "retrieval_ablation.json"

#: Cut-offs reported. 5 is the agent's default top_k; 20 is the ceiling the tool
#: schema allows.
K_VALUES: Final[tuple[int, ...]] = (5, 10, 20)

#: Candidates each arm pulls before fusion or reranking. Held constant across
#: arms so the comparison is about ranking, not about how much was retrieved.
CANDIDATES: Final = 50


@dataclass(slots=True)
class ArmScores:
    """Accumulated per-query scores for one retrieval configuration."""

    name: str
    recall_at: dict[int, list[float]] = field(default_factory=dict)
    reciprocal_ranks: list[float] = field(default_factory=list)
    ndcg_at: dict[int, list[float]] = field(default_factory=dict)
    queries: int = 0
    #: Queries where no relevant chunk appeared at any depth. Reported because a
    #: mean over only the queries that found something is a different number.
    misses: int = 0

    def mean_recall(self, k: int) -> float:
        values = self.recall_at.get(k, [])
        return statistics.mean(values) if values else 0.0

    def mean_ndcg(self, k: int) -> float:
        values = self.ndcg_at.get(k, [])
        return statistics.mean(values) if values else 0.0

    @property
    def mrr(self) -> float:
        return statistics.mean(self.reciprocal_ranks) if self.reciprocal_ranks else 0.0


def _overlaps(hit: RetrievalHit, gold: Sequence[tuple[int, int]]) -> bool:
    """Whether a retrieved chunk touches any gold span."""
    start, end = hit.chunk.char_start, hit.chunk.char_end
    return any(start < gold_end and gold_start < end for gold_start, gold_end in gold)


def _dcg(relevances: Iterable[int]) -> float:
    return sum(rel / math.log2(position + 2) for position, rel in enumerate(relevances))


def score_query(
    arm: ArmScores, hits: Sequence[RetrievalHit], gold: Sequence[tuple[int, int]]
) -> None:
    """Score one arm's ranking for one (contract, clause type) query."""
    arm.queries += 1
    relevance = [1 if _overlaps(hit, gold) else 0 for hit in hits]
    total_relevant = sum(relevance)

    if not total_relevant:
        arm.misses += 1

    first = next((index + 1 for index, rel in enumerate(relevance) if rel), None)
    arm.reciprocal_ranks.append(1.0 / first if first else 0.0)

    for k in K_VALUES:
        top = relevance[:k]
        # Recall here is "did a relevant chunk make the top k", not the share of
        # gold spans covered: the agent needs one good passage to answer, and a
        # clause split across three chunks should not be scored as 1/3.
        arm.recall_at.setdefault(k, []).append(1.0 if any(top) else 0.0)

        ideal = _dcg([1] * min(total_relevant, k))
        arm.ndcg_at.setdefault(k, []).append(_dcg(top) / ideal if ideal else 0.0)


def run_ablation(cases: Sequence[GoldenCase], fake_embeddings: bool = False) -> dict[str, Any]:
    """Score every arm on every positive case. Local models only, no API."""
    import psycopg
    from pgvector.psycopg import register_vector

    from docintel.retrieval.embed import BgeEmbedder, HashingEmbedder
    from docintel.retrieval.hybrid import HybridRetriever, dense_search, lexical_search
    from docintel.retrieval.rerank import BgeReranker
    from docintel.retrieval.rerank import rerank as apply_rerank

    settings = get_settings()
    connection = psycopg.connect(str(settings.database_url))
    register_vector(connection)
    embedder = (
        HashingEmbedder(dimension=settings.retrieval.embedding_dim)
        if fake_embeddings
        else BgeEmbedder()
    )
    retriever = HybridRetriever(
        connection=connection,
        embedder=embedder,
        candidates_per_retriever=CANDIDATES,
        rrf_k=settings.retrieval.rrf_k,
    )
    reranker = BgeReranker()

    arms = {
        name: ArmScores(name=name)
        for name in ("lexical_only", "dense_only", "hybrid_rrf", "hybrid_rrf_rerank")
    }

    # Only gold-positive cases: there is no ranking to score when the clause is
    # absent, and including them would make every arm look identical.
    positives = [case for case in cases if case.present and case.gold_spans]
    contracts = len({c.document_id for c in positives})
    print(f"scoring {len(positives)} positive cases over {contracts} contracts")

    indexed = _indexed_documents(connection)
    skipped = 0

    try:
        for index, case in enumerate(positives, start=1):
            if case.document_id not in indexed:
                skipped += 1
                continue
            query = CLAUSE_DEFINITIONS[case.clause_type]
            gold = list(case.gold_spans)
            docs = [case.document_id]

            lexical = [s.chunk for s in lexical_search(connection, query, CANDIDATES, docs)]
            dense = [
                s.chunk
                for s in dense_search(connection, embedder.embed_query(query), CANDIDATES, docs)
            ]
            fused = list(retriever.search(query, top_k=CANDIDATES, document_ids=docs))
            reranked = apply_rerank(reranker, query, fused, top_k=CANDIDATES) if fused else []

            score_query(arms["lexical_only"], _as_hits(lexical), gold)
            score_query(arms["dense_only"], _as_hits(dense), gold)
            score_query(arms["hybrid_rrf"], fused, gold)
            score_query(arms["hybrid_rrf_rerank"], reranked, gold)

            if index % 10 == 0:
                print(f"  {index}/{len(positives)}", flush=True)
    finally:
        connection.close()

    if skipped:
        print(f"skipped {skipped} cases whose contract is not indexed")

    return {
        "cases_scored": arms["hybrid_rrf"].queries,
        "cases_skipped_not_indexed": skipped,
        "candidates_per_arm": CANDIDATES,
        "relevance": "binary: a chunk is relevant if it overlaps any gold span",
        "arms": {
            name: {
                "recall_at": {str(k): round(arm.mean_recall(k), 4) for k in K_VALUES},
                "mrr": round(arm.mrr, 4),
                "ndcg_at": {str(k): round(arm.mean_ndcg(k), 4) for k in K_VALUES},
                "queries": arm.queries,
                "no_relevant_chunk_found": arm.misses,
            }
            for name, arm in arms.items()
        },
    }


def _as_hits(chunks: Sequence[Any]) -> list[RetrievalHit]:
    return [RetrievalHit(chunk=chunk, score=0.0, ranks={}) for chunk in chunks]


def _indexed_documents(connection: Any) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT document_id FROM documents")
        return {row[0] for row in cursor.fetchall()}


def print_report(result: dict[str, Any]) -> None:
    print(f"\n{result['cases_scored']} positive cases | binary relevance vs CUAD gold spans")
    header = f"{'arm':<20}" + "".join(f"{'R@' + str(k):>9}" for k in K_VALUES)
    header += (
        f"{'MRR':>9}" + "".join(f"{'nDCG@' + str(k):>10}" for k in K_VALUES) + f"{'misses':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, arm in result["arms"].items():
        row = f"{name:<20}"
        row += "".join(f"{arm['recall_at'][str(k)]:>9.3f}" for k in K_VALUES)
        row += f"{arm['mrr']:>9.3f}"
        row += "".join(f"{arm['ndcg_at'][str(k)]:>10.3f}" for k in K_VALUES)
        row += f"{arm['no_relevant_chunk_found']:>8}"
        print(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fake-embeddings", action="store_true")
    args = parser.parse_args(argv)

    cases = load_cases()
    if args.limit:
        keep = sorted(cases_by_document(cases))[: args.limit]
        cases = [case for case in cases if case.document_id in set(keep)]

    result = run_ablation(cases, fake_embeddings=args.fake_embeddings)
    print_report(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
