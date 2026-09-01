"""Freeze the dev / golden / reserve contract split, per CLAUDE.md section 2.3.

Run once. The output is committed and must not be regenerated after prompt work
begins -- the whole point is that prompts are developed on `dev` and the reported
numbers come from `golden`, which no prompt has ever seen.

The split is stratified by contract length quartile. Simple random sampling
would leave the expensive long tail unevenly distributed between dev and golden,
which matters here because the headline experiment is a cost curve over length:
a golden set that happens to be short would understate the long-context baseline's
cost and flatter retrieval.

Usage::

    uv run python scripts/build_split.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from audit_data import Contract, load_contracts

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_CUAD_DIR: Final = REPO_ROOT / "data" / "raw" / "CUAD_v1"
DEFAULT_OUTPUT: Final = REPO_ROOT / "evals" / "golden" / "split.json"

DEV_SIZE: Final = 60
GOLDEN_SIZE: Final = 100
QUARTILES: Final = 4


def length_quartiles(contracts: Sequence[Contract]) -> dict[str, int]:
    """Assign each contract to a length quartile, 1 (shortest) to 4 (longest).

    Uses character count rather than tokens: the ordering is what matters for
    stratification, it is monotonic with token count, and it keeps this script
    free of a tokenizer dependency so the split can be rebuilt anywhere.
    """
    ordered = sorted(contracts, key=lambda c: (len(c.text), c.title))
    size = len(ordered)
    return {
        contract.title: min(QUARTILES, (index * QUARTILES) // size + 1)
        for index, contract in enumerate(ordered)
    }


def stratified_sample(
    pool: list[str],
    quartiles: dict[str, int],
    size: int,
    rng: random.Random,
) -> list[str]:
    """Draw ``size`` titles from ``pool``, proportionally across quartiles.

    Remainders are allocated to the quartiles with the largest fractional part,
    so the totals always sum to ``size`` without a final top-up that would skew
    one stratum.
    """
    by_quartile: dict[int, list[str]] = {q: [] for q in range(1, QUARTILES + 1)}
    for title in pool:
        by_quartile[quartiles[title]].append(title)

    exact = {q: len(members) * size / len(pool) for q, members in by_quartile.items()}
    allocation = {q: int(value) for q, value in exact.items()}
    shortfall = size - sum(allocation.values())
    for quartile in sorted(exact, key=lambda q: (-(exact[q] - allocation[q]), q))[:shortfall]:
        allocation[quartile] += 1

    selected: list[str] = []
    for quartile in range(1, QUARTILES + 1):
        members = sorted(by_quartile[quartile])  # sort first: rng.sample is order-sensitive
        selected.extend(rng.sample(members, k=allocation[quartile]))
    return sorted(selected)


def build_split(contracts: Sequence[Contract], seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    quartiles = length_quartiles(contracts)
    remaining = sorted(c.title for c in contracts)

    dev = stratified_sample(remaining, quartiles, DEV_SIZE, rng)
    remaining = sorted(set(remaining) - set(dev))

    golden = stratified_sample(remaining, quartiles, GOLDEN_SIZE, rng)
    reserve = sorted(set(remaining) - set(golden))

    def summarize(titles: Sequence[str]) -> dict[str, object]:
        counts = Counter(quartiles[t] for t in titles)
        return {
            "n": len(titles),
            "by_length_quartile": {str(q): counts.get(q, 0) for q in range(1, QUARTILES + 1)},
            "contract_ids": list(titles),
        }

    return {
        "seed": seed,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "generator": "scripts/build_split.py",
        "corpus": {
            "source": "CUAD v1, full_contract_txt",
            "n_contracts": len(contracts),
        },
        "procedure": (
            "Contracts are ordered by character length and assigned to four "
            "equal-count quartiles. The dev set (60) is drawn first, stratified "
            "proportionally across quartiles; the golden set (100) is drawn from "
            "the remainder under the same stratification; everything left is "
            "reserve. Sampling is seeded and the pool is sorted before each draw, "
            "so the split is reproducible. Contract IDs are CUAD titles, which "
            "are also the TXT filename stems (NFC-normalized)."
        ),
        "invariants": [
            "dev, golden, and reserve are pairwise disjoint",
            "their union is the full corpus",
            "prompts are developed on dev only; golden is read at milestone boundaries",
        ],
        "splits": {
            "dev": summarize(dev),
            "golden": summarize(golden),
            "reserve": summarize(reserve),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuad-dir", type=Path, default=DEFAULT_CUAD_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing split. Refuses by default: regenerating a "
        "frozen split after prompt work has begun invalidates every reported number.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists() and not args.force:
        print(f"{args.output} already exists; refusing to overwrite. Pass --force if you mean it.")
        return 1

    contracts, unmatched = load_contracts(args.cuad_dir)
    if unmatched:
        print(f"warning: {len(unmatched)} contracts had no TXT file and are excluded")

    split = build_split(contracts, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    splits = split["splits"]
    assert isinstance(splits, dict)
    for name, payload in splits.items():
        assert isinstance(payload, dict)
        print(f"{name:8} n={payload['n']:<4} quartiles={payload['by_length_quartile']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
