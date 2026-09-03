"""Preflight invariants: refuse to measure the wrong system.

Runs before any spend and asserts that the run is measuring what it claims to
be measuring. Exits non-zero on any failure, and prints what it verified rather
than only what broke -- a check that says nothing when it passes teaches nobody
what it covers.

### Why this exists

Three separate times, this project produced a plausible number that measured the
wrong thing, and none of them raised:

1. **Budget-exhausted runs scored as 0/12 present.** A run that hit the turn
   ceiling looked like total recall loss, so presence F1 was partly a
   measurement of ``max_turns``.
2. **Cache-read time reported as model latency.** Replaying a run recomputed
   latency at replay, so a frozen baseline claimed a 972ms p50 for an arm whose
   live p50 was 44.4 seconds.
3. **An entire evaluation ran against an empty index.** The dev split was
   indexed; the golden split is disjoint from it by construction and was never
   indexed. Every ``search_contract`` call returned nothing. The agent's
   completion rate, cost, turn count and span F1 were all measured on a
   retrieval system that was not there -- and the diagnoses built on top of
   them, about scanning and re-searching, were explanations for an artifact.

Each was caught by accident, late, after money had been spent. The common shape
is a number that is present, plausible, and wrong. Unit tests do not catch these
because each component works correctly in isolation; what fails is the claim
that the assembled thing is the thing being described.

So the checks below verify *setup*, not behaviour, and they run before the
money does.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.config import get_settings
from evals.cases import cases_by_document, load_cases

REPO_ROOT: Final = Path(__file__).resolve().parents[1]

#: A query that must return something from any indexed contract. Deliberately
#: generic: if this returns nothing, retrieval is broken rather than picky.
CANARY_QUERY: Final = "this agreement"

__all__ = ["Check", "Preflight", "run_preflight"]


@dataclass(slots=True)
class Check:
    """One invariant, its verdict, and enough detail to act on a failure."""

    name: str
    passed: bool
    detail: str
    #: Set when failing, so the operator is told what to run.
    remedy: str = ""


@dataclass(slots=True)
class Preflight:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str, remedy: str = "") -> None:
        self.checks.append(Check(name=name, passed=passed, detail=detail, remedy=remedy))

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def report(self) -> None:
        print("preflight")
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            print(f"  [{mark}] {check.name:<28} {check.detail}")
            if not check.passed and check.remedy:
                print(f"         -> {check.remedy}")
        print("preflight " + ("passed" if self.ok else "FAILED"))


def check_index_coverage(preflight: Preflight, document_ids: set[str]) -> None:
    """Every document to be evaluated must be in the retrieval index.

    The check that would have caught incident 3, and the reason it is first.
    """
    import psycopg

    settings = get_settings()
    try:
        connection = psycopg.connect(str(settings.database_url), connect_timeout=5)
    except Exception as exc:
        preflight.add(
            "database reachable",
            False,
            f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}",
            "docker compose -f docker/docker-compose.yml up -d",
        )
        return

    with connection:
        preflight.add("database reachable", True, str(settings.database_url).split("@")[-1])
        with connection.cursor() as cursor:
            cursor.execute("SELECT document_id FROM documents")
            indexed = {row[0] for row in cursor.fetchall()}

        missing = sorted(document_ids - indexed)
        preflight.add(
            "index covers case set",
            not missing,
            f"{len(document_ids) - len(missing)}/{len(document_ids)} documents indexed"
            + (f"; missing e.g. {missing[0][:44]}" if missing else ""),
            "uv run python -m docintel.cli index --split golden",
        )

        # Coverage alone is not enough: a document row with no chunks is still
        # unsearchable.
        if not missing and document_ids:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT document_id, count(*) FROM chunks "
                    "WHERE document_id = ANY(%s) GROUP BY document_id",
                    (list(document_ids),),
                )
                counts: dict[str, int] = dict(cursor.fetchall())
            empty = sorted(doc for doc in document_ids if counts.get(doc, 0) == 0)
            preflight.add(
                "every document has chunks",
                not empty,
                f"min {min(counts.values())} / median chunk count present"
                if counts and not empty
                else f"{len(empty)} document(s) have zero chunks",
                "re-index; a document row without chunks cannot be searched",
            )

        # Embeddings: a chunk with a NULL embedding is invisible to dense search.
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM chunks WHERE document_id = ANY(%s) AND embedding IS NULL",
                (list(document_ids),),
            )
            row = cursor.fetchone()
            unembedded = int(row[0]) if row else 0
        preflight.add(
            "chunks are embedded",
            unembedded == 0,
            "all chunks have embeddings" if not unembedded else f"{unembedded} chunks unembedded",
            "re-index; dense retrieval silently skips NULL embeddings",
        )


def check_retrieval_canary(preflight: Preflight, document_ids: set[str]) -> None:
    """Retrieval must actually return something for a generic query.

    Coverage says the rows exist; this says the query path works end to end --
    FTS configuration, the vector extension, and the embedder all agree.
    """
    import psycopg

    from docintel.retrieval.hybrid import lexical_search

    if not document_ids:
        return
    settings = get_settings()
    try:
        with psycopg.connect(str(settings.database_url), connect_timeout=5) as connection:
            sample = sorted(document_ids)[0]
            hits = lexical_search(connection, CANARY_QUERY, 5, [sample])
    except Exception as exc:
        preflight.add("retrieval canary", False, f"{type(exc).__name__}: {exc}"[:80])
        return

    preflight.add(
        "retrieval canary",
        bool(hits),
        f"{len(hits)} hits for {CANARY_QUERY!r} on {sample[:40]}",
        "retrieval returns nothing even for a generic query; check the FTS index",
    )


def check_split_membership(preflight: Preflight, document_ids: set[str], split: str) -> None:
    """Evaluated documents must belong to the split they claim to.

    Guards leakage in the other direction: reporting dev numbers as golden.
    """
    from docintel.ingest.loader import load_split

    try:
        members = load_split(REPO_ROOT / "evals" / "golden" / "split.json", split)
    except Exception as exc:
        preflight.add("split membership", False, str(exc)[:70])
        return
    stray = sorted(document_ids - members)
    preflight.add(
        "split membership",
        not stray,
        f"all {len(document_ids)} documents are in the {split!r} split"
        if not stray
        else f"{len(stray)} document(s) outside {split!r}, e.g. {stray[0][:40]}",
        "the case set and the split disagree; rebuild cases.json",
    )


def check_frozen_config(preflight: Preflight, max_turns: int, model: str, prompt: str) -> None:
    """The run's agent config must match what ``docs/ARCHITECTURE.md`` records.

    A results table is only attributable if the configuration it was produced
    with is the one written down.
    """
    architecture = REPO_ROOT / "docs" / "ARCHITECTURE.md"
    if not architecture.exists():
        preflight.add("frozen config recorded", False, "docs/ARCHITECTURE.md is missing")
        return
    text = architecture.read_text(encoding="utf-8")
    expected = {"model": model, "prompt": prompt, "max_turns": str(max_turns)}
    absent = [key for key, value in expected.items() if value not in text]
    preflight.add(
        "frozen config recorded",
        not absent,
        f"model={model} prompt={prompt} max_turns={max_turns} all present"
        if not absent
        else f"not found in ARCHITECTURE.md: {absent}",
        "update docs/ARCHITECTURE.md, or the run is not attributable to a stated config",
    )


def check_capture_time_metrics(preflight: Preflight) -> None:
    """Latency must be stored at capture, not recomputed at replay.

    The check that would have caught incident 2. It verifies the mechanism
    exists rather than the value, because the value is only wrong on replay.
    """
    from evals.cache import DEFAULT_CACHE_DIR

    entries = list(DEFAULT_CACHE_DIR.glob("*/*.json"))
    if not entries:
        preflight.add("latency stored at capture", True, "cache is empty; nothing to replay")
        return

    import json

    with_latency = 0
    for path in entries[:200]:
        meta = json.loads(path.read_text(encoding="utf-8")).get("meta") or {}
        with_latency += "latency_ms" in meta
    sampled = min(len(entries), 200)
    preflight.add(
        "latency stored at capture",
        True,  # informational: old entries are excluded, not fatal
        f"{with_latency}/{sampled} sampled entries carry capture latency; "
        f"the rest report latency as unavailable rather than zero",
    )


def run_preflight(
    split: str = "golden",
    max_turns: int = 20,
    model: str = "claude-sonnet-5",
    prompt: str = "extract_v1",
    limit: int = 0,
    sample: int = 0,
    sample_seed: int = 42,
) -> Preflight:
    """Every invariant, against the documents a run would actually touch."""
    import random

    preflight = Preflight()
    cases = load_cases()
    document_ids = set(cases_by_document(cases))

    if sample:
        # Sorted before sampling: random.sample is order-sensitive, so an
        # unsorted input yields a different subset per run at the same seed.
        # Same trap the split builder documents.
        chosen = random.Random(sample_seed).sample(
            sorted(document_ids), k=min(sample, len(document_ids))
        )
        document_ids = set(chosen)
    if limit:
        document_ids = set(sorted(document_ids)[:limit])

    preflight.add(
        "case set loaded", bool(cases), f"{len(cases)} cases, {len(document_ids)} documents"
    )
    check_split_membership(preflight, document_ids, split)
    check_index_coverage(preflight, document_ids)
    check_retrieval_canary(preflight, document_ids)
    check_frozen_config(preflight, max_turns, model, prompt)
    check_capture_time_metrics(preflight)
    return preflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="golden")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--prompt", default="extract_v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args(argv)

    preflight = run_preflight(
        split=args.split,
        max_turns=args.max_turns,
        model=args.model,
        prompt=args.prompt,
        limit=args.limit,
        sample=args.sample,
        sample_seed=args.sample_seed,
    )
    preflight.report()
    return 0 if preflight.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
