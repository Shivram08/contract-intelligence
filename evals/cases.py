"""Golden case construction and stratified sampling.

A **case** is one ``(contract, clause_type)`` pair with a gold label. 150 of
them, drawn from the frozen golden split, committed to
``evals/golden/cases.json``.

### Why 150 and not all 1,200

The golden split holds 100 contracts, so 1,200 pairs exist. Two facts pull in
opposite directions:

- **Cost is per contract, not per case.** One extraction produces all 12 clause
  verdicts for the same $0.2265 (measured in M2). Scoring only 150 pairs
  discards 87% of judgements already paid for.
- **A live baseline over all 100 contracts costs ~$22.65.** Times four
  model-driven baselines, that is real money for a number that is re-derivable.

The resolution: 150 cases is the **gated** set -- what CI replays and what
``docs/RESULTS.md`` reports, matching ``CLAUDE.md`` sections 6 and 13. It draws
on roughly half the golden contracts, so a live baseline is ~$11 rather than
~$23. But every run records all 12 clause verdicts per contract it touches, so
the wider view costs nothing extra and is available for a
higher-powered secondary analysis.

### Why stratify on clause type rather than tier

``CLAUDE.md`` section 6 asks for stratification "across the three difficulty
tiers and across contract length quartiles". Stratifying on ``(clause_type,
length_quartile)`` -- 48 strata -- delivers that *and* more: tiers come out
balanced automatically, because each tier contains exactly four clause types.
Stratifying on tier directly would leave the 12 clause types unevenly covered by
chance, and section 6 also wants per-clause-type presence F1.

### The honest limit on per-clause numbers

150 cases over 12 clause types is ~12.5 cases each, of which roughly 6.7 are
gold-positive. **Per-clause presence F1 on seven positives has error bars wider
than most differences worth measuring.** The trustworthy aggregate is the tier
level, at 50 cases each. ``metrics.py`` reports a Wilson interval beside every
per-clause figure for exactly this reason, and ``evals/README.md`` says so in
prose. Reporting a bare per-clause F1 here would be precision theatre.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from docintel.ingest.loader import (
    GoldSpan,
    load_gold_spans,
    load_split,
    read_contract_text,
)
from docintel.schemas import CLAUSE_TIERS, CUAD_CATEGORIES, ClauseType, Tier
from docintel.text import contract_key

__all__ = [
    "DEFAULT_CASES_PATH",
    "GOLDEN_CASE_COUNT",
    "GoldenCase",
    "build_cases",
    "load_cases",
    "save_cases",
    "stratum_summary",
]

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH: Final = REPO_ROOT / "evals" / "golden" / "cases.json"
DEFAULT_SPLIT_PATH: Final = REPO_ROOT / "evals" / "golden" / "split.json"
DEFAULT_CUAD_DIR: Final = REPO_ROOT / "data" / "raw" / "CUAD_v1"

#: Per CLAUDE.md sections 6 and 13.
GOLDEN_CASE_COUNT: Final = 150

#: Length quartiles, 1 (shortest) to 4 (longest).
QUARTILES: Final = 4


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One gold-labelled ``(contract, clause_type)`` pair."""

    case_id: str
    document_id: str
    clause_type: ClauseType
    #: Gold presence label, from CUAD_v1.json spans. True iff at least one span.
    present: bool
    #: Gold spans as ``(char_start, char_end)`` into the raw document. Empty when
    #: absent. Multi-span is common -- 39% of positive pairs -- and all spans are
    #: kept, because span F1 scores against their union.
    gold_spans: tuple[tuple[int, int], ...] = ()
    length_quartile: int = 1
    #: Contract length in characters, for the cost-versus-length analysis.
    document_chars: int = 0

    @property
    def tier(self) -> Tier:
        return CLAUSE_TIERS[self.clause_type]

    @property
    def stratum(self) -> tuple[str, int]:
        return (self.clause_type.value, self.length_quartile)

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "document_id": self.document_id,
            "clause_type": self.clause_type.value,
            "present": self.present,
            "gold_spans": [list(span) for span in self.gold_spans],
            "length_quartile": self.length_quartile,
            "document_chars": self.document_chars,
            "tier": int(self.tier),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> GoldenCase:
        return cls(
            case_id=str(payload["case_id"]),
            document_id=str(payload["document_id"]),
            clause_type=ClauseType(payload["clause_type"]),
            present=bool(payload["present"]),
            gold_spans=tuple((int(a), int(b)) for a, b in payload.get("gold_spans", [])),
            length_quartile=int(payload.get("length_quartile", 1)),
            document_chars=int(payload.get("document_chars", 0)),
        )


def _length_quartiles(lengths: dict[str, int]) -> dict[str, int]:
    """Assign each document a length quartile, 1 to 4.

    Character count, not tokens: the ordering is identical for stratification
    purposes and it keeps this module free of a tokenizer dependency.
    """
    ordered = sorted(lengths, key=lambda doc_id: (lengths[doc_id], doc_id))
    size = len(ordered)
    return {
        doc_id: min(QUARTILES, (index * QUARTILES) // size + 1)
        for index, doc_id in enumerate(ordered)
    }


def _spread(total: int, buckets: Sequence[Any], rng: random.Random) -> dict[Any, int]:
    """Split ``total`` as evenly as possible, giving the remainder out at random.

    ``total // n`` to everyone, then the remaining ``total % n`` to a seeded
    random subset. Handing the remainder to the alphabetically-first buckets is
    what a naive largest-remainder fill does when every bucket is the same size,
    and it produced a case set where ``anti_assignment`` had 16 cases and
    everything else had 12 -- a third more coverage for one clause type because
    of its name.
    """
    base, extra = divmod(total, len(buckets))
    allocation = dict.fromkeys(buckets, base)
    for bucket in rng.sample(sorted(buckets, key=str), k=extra):
        allocation[bucket] += 1
    return allocation


def _allocate(
    strata: dict[tuple[str, int], list[GoldenCase]], total: int, rng: random.Random
) -> dict[tuple[str, int], int]:
    """Two-level allocation: balance clause types first, then quartiles.

    Flat proportional allocation over 48 equal-sized strata cannot express "every
    clause type gets the same number of cases", which is the property the
    per-clause metrics need. So the split happens twice: ``total`` across the 12
    clause types, then each type's share across its 4 length quartiles.

    Capped at what each stratum actually holds, and any shortfall from a thin
    stratum is redistributed within the same clause type.
    """
    clause_types = sorted({clause for clause, _ in strata})
    per_clause = _spread(total, clause_types, rng)

    allocation: dict[tuple[str, int], int] = {}
    for clause in clause_types:
        quartiles = sorted(q for c, q in strata if c == clause)
        if not quartiles:
            continue
        want = _spread(per_clause[clause], quartiles, rng)

        # Respect the size of each stratum, then push any leftover onto the
        # quartiles that still have room.
        leftover = 0
        for quartile in quartiles:
            available = len(strata[(clause, quartile)])
            take = min(want[quartile], available)
            leftover += want[quartile] - take
            allocation[(clause, quartile)] = take

        for quartile in quartiles:
            if leftover <= 0:
                break
            room = len(strata[(clause, quartile)]) - allocation[(clause, quartile)]
            added = min(room, leftover)
            allocation[(clause, quartile)] += added
            leftover -= added

    return allocation


def build_cases(
    cuad_dir: Path | None = None,
    split_path: Path | None = None,
    split_name: str = "golden",
    size: int = GOLDEN_CASE_COUNT,
    seed: int = 42,
) -> list[GoldenCase]:
    """Build the stratified golden case set.

    Stratified on ``(clause_type, length_quartile)``: 48 strata, so every clause
    type gets ~12 cases and every length quartile ~37. Tier balance follows,
    since each tier holds exactly four clause types.
    """
    cuad = cuad_dir or DEFAULT_CUAD_DIR
    document_ids = load_split(split_path or DEFAULT_SPLIT_PATH, split_name)

    # Read lengths from disk rather than the DB: case construction must work
    # from a clean clone with no Postgres running. `read_contract_text` is used
    # rather than Path.read_text because it disables newline translation -- the
    # length has to match the coordinate system the gold offsets index into.
    lengths: dict[str, int] = {}
    for path in sorted((cuad / "full_contract_txt").glob("*.txt")):
        key = contract_key(path.stem)
        if key in document_ids:
            lengths[key] = len(read_contract_text(path))

    missing = document_ids - set(lengths)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} split contracts have no TXT file: {sorted(missing)[:3]}"
        )

    quartiles = _length_quartiles(lengths)

    # Gold labels. A pair is positive iff CUAD carries at least one span for it.
    spans_by_pair: dict[tuple[str, str], list[GoldSpan]] = defaultdict(list)
    for span in load_gold_spans(cuad, document_ids):
        spans_by_pair[(span.document_id, span.category)].append(span)

    strata: dict[tuple[str, int], list[GoldenCase]] = defaultdict(list)
    for document_id in sorted(lengths):
        for clause_type in ClauseType:
            gold = spans_by_pair.get((document_id, CUAD_CATEGORIES[clause_type]), [])
            case = GoldenCase(
                case_id=f"{document_id}::{clause_type.value}",
                document_id=document_id,
                clause_type=clause_type,
                present=bool(gold),
                gold_spans=tuple(sorted((s.char_start, s.char_end) for s in gold)),
                length_quartile=quartiles[document_id],
                document_chars=lengths[document_id],
            )
            strata[case.stratum].append(case)

    rng = random.Random(seed)
    allocation = _allocate(dict(strata), size, rng)
    selected: list[GoldenCase] = []
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda c: c.case_id)
        take = allocation.get(stratum, 0)
        if take:
            selected.extend(rng.sample(members, k=take))

    return sorted(selected, key=lambda c: c.case_id)


def stratum_summary(cases: Sequence[GoldenCase]) -> dict[str, Any]:
    """Counts by clause type, tier, quartile, and gold label.

    Reported when the case set is built, so the stratification is visible rather
    than asserted.
    """
    by_clause: dict[str, int] = defaultdict(int)
    by_tier: dict[int, int] = defaultdict(int)
    by_quartile: dict[int, int] = defaultdict(int)
    positives_by_clause: dict[str, int] = defaultdict(int)

    for case in cases:
        by_clause[case.clause_type.value] += 1
        by_tier[int(case.tier)] += 1
        by_quartile[case.length_quartile] += 1
        if case.present:
            positives_by_clause[case.clause_type.value] += 1

    return {
        "total": len(cases),
        "documents": len({case.document_id for case in cases}),
        "positives": sum(1 for case in cases if case.present),
        "by_clause_type": dict(sorted(by_clause.items())),
        "positives_by_clause_type": {k: positives_by_clause[k] for k in sorted(by_clause)},
        "by_tier": dict(sorted(by_tier.items())),
        "by_length_quartile": dict(sorted(by_quartile.items())),
    }


def save_cases(
    cases: Sequence[GoldenCase],
    path: Path | None = None,
    seed: int = 42,
    split_name: str = "golden",
) -> Path:
    """Write the case set as committed JSON."""
    target = path or DEFAULT_CASES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "split": split_name,
        "generator": "evals/cases.py",
        "unit": "one (contract, clause_type) pair with a CUAD gold label",
        "stratification": (
            "(clause_type, length_quartile), 48 strata, largest-remainder "
            "allocation. Tier balance follows because each tier holds four "
            "clause types."
        ),
        "caveat": (
            "~12 cases per clause type, of which ~7 are positive. Per-clause "
            "presence F1 therefore carries wide error bars; the tier-level "
            "aggregate (50 cases each) is the figure to trust."
        ),
        "summary": stratum_summary(cases),
        "cases": [case.to_json() for case in cases],
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def load_cases(path: Path | None = None) -> list[GoldenCase]:
    """Read the committed case set."""
    payload = json.loads((path or DEFAULT_CASES_PATH).read_text(encoding="utf-8"))
    return [GoldenCase.from_json(entry) for entry in payload["cases"]]


def cases_by_document(cases: Iterable[GoldenCase]) -> dict[str, list[GoldenCase]]:
    """Group cases by contract.

    The eval runner iterates documents, not cases: one extraction answers every
    case for that contract, so grouping is what keeps the run from paying twice
    for the same document.
    """
    grouped: dict[str, list[GoldenCase]] = defaultdict(list)
    for case in cases:
        grouped[case.document_id].append(case)
    return {doc: sorted(items, key=lambda c: c.clause_type.value) for doc, items in grouped.items()}
