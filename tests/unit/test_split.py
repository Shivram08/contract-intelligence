"""Tests for the frozen dev / golden / reserve split.

Two jobs. The first half tests the sampling logic on synthetic data. The second
half is a regression guard on the committed `evals/golden/split.json`: once
prompt development begins, any change to that file silently invalidates every
reported number, so the invariants are asserted in CI rather than trusted.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from audit_data import Contract
from build_split import (
    DEV_SIZE,
    GOLDEN_SIZE,
    QUARTILES,
    build_split,
    length_quartiles,
    stratified_sample,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = REPO_ROOT / "evals" / "golden" / "split.json"


def make_contracts(count: int) -> list[Contract]:
    """Synthetic corpus with strictly increasing lengths."""
    return [
        Contract(
            title=f"CONTRACT_{i:04d}",
            text="x" * (100 + i * 10),
            path=Path(f"CONTRACT_{i:04d}.txt"),
            annotations={},
        )
        for i in range(count)
    ]


class TestLengthQuartiles:
    def test_assigns_every_contract_a_quartile_in_range(self) -> None:
        quartiles = length_quartiles(make_contracts(100))
        assert len(quartiles) == 100
        assert set(quartiles.values()) == {1, 2, 3, 4}

    def test_quartiles_are_balanced(self) -> None:
        quartiles = length_quartiles(make_contracts(100))
        counts = [sum(1 for q in quartiles.values() if q == n) for n in range(1, 5)]
        assert counts == [25, 25, 25, 25]

    def test_shortest_contract_lands_in_quartile_one(self) -> None:
        contracts = make_contracts(40)
        quartiles = length_quartiles(contracts)
        shortest = min(contracts, key=lambda c: len(c.text))
        longest = max(contracts, key=lambda c: len(c.text))
        assert quartiles[shortest.title] == 1
        assert quartiles[longest.title] == QUARTILES

    def test_ties_break_deterministically_on_title(self) -> None:
        """Equal-length contracts must not shuffle between runs."""
        contracts = [
            Contract(title=t, text="x" * 500, path=Path(f"{t}.txt"), annotations={})
            for t in ("C", "A", "B", "D")
        ]
        assert length_quartiles(contracts) == length_quartiles(list(reversed(contracts)))


class TestStratifiedSample:
    def test_returns_exactly_the_requested_size(self) -> None:
        contracts = make_contracts(200)
        quartiles = length_quartiles(contracts)
        pool = sorted(c.title for c in contracts)
        for size in (1, 7, 60, 100, 199):
            sample = stratified_sample(pool, quartiles, size, random.Random(42))
            assert len(sample) == size

    def test_spreads_across_quartiles(self) -> None:
        contracts = make_contracts(200)
        quartiles = length_quartiles(contracts)
        pool = sorted(c.title for c in contracts)
        sample = stratified_sample(pool, quartiles, 100, random.Random(42))
        counts = [sum(1 for t in sample if quartiles[t] == q) for q in range(1, 5)]
        assert counts == [25, 25, 25, 25]

    def test_remainder_allocation_never_overdraws_a_quartile(self) -> None:
        """A size that does not divide evenly must not request more than exists."""
        contracts = make_contracts(37)
        quartiles = length_quartiles(contracts)
        pool = sorted(c.title for c in contracts)
        sample = stratified_sample(pool, quartiles, 13, random.Random(1))
        assert len(sample) == len(set(sample)) == 13

    def test_is_deterministic_for_a_given_seed(self) -> None:
        contracts = make_contracts(120)
        quartiles = length_quartiles(contracts)
        pool = sorted(c.title for c in contracts)
        first = stratified_sample(pool, quartiles, 40, random.Random(42))
        second = stratified_sample(pool, quartiles, 40, random.Random(42))
        assert first == second

    def test_different_seeds_give_different_samples(self) -> None:
        contracts = make_contracts(120)
        quartiles = length_quartiles(contracts)
        pool = sorted(c.title for c in contracts)
        assert stratified_sample(pool, quartiles, 40, random.Random(1)) != stratified_sample(
            pool, quartiles, 40, random.Random(2)
        )

    def test_pool_order_does_not_affect_the_result(self) -> None:
        """Guards the subtle bug: `rng.sample` depends on input ordering, so the
        pool is sorted inside the function. Without that, a caller passing a set
        would get a different split on a different Python run."""
        contracts = make_contracts(120)
        quartiles = length_quartiles(contracts)
        pool = sorted(c.title for c in contracts)
        shuffled = list(pool)
        random.Random(7).shuffle(shuffled)
        assert stratified_sample(pool, quartiles, 40, random.Random(42)) == stratified_sample(
            shuffled, quartiles, 40, random.Random(42)
        )


class TestBuildSplit:
    def test_splits_are_disjoint_and_cover_the_corpus(self) -> None:
        contracts = make_contracts(510)
        result = build_split(contracts, seed=42)
        splits = result["splits"]
        assert isinstance(splits, dict)

        sets = {name: set(payload["contract_ids"]) for name, payload in splits.items()}
        assert sets["dev"] & sets["golden"] == set()
        assert sets["dev"] & sets["reserve"] == set()
        assert sets["golden"] & sets["reserve"] == set()
        assert set.union(*sets.values()) == {c.title for c in contracts}

    def test_sizes_match_the_spec(self) -> None:
        result = build_split(make_contracts(510), seed=42)
        splits = result["splits"]
        assert isinstance(splits, dict)
        assert splits["dev"]["n"] == DEV_SIZE
        assert splits["golden"]["n"] == GOLDEN_SIZE
        assert splits["reserve"]["n"] == 510 - DEV_SIZE - GOLDEN_SIZE

    def test_is_reproducible(self) -> None:
        contracts = make_contracts(510)
        assert build_split(contracts, seed=42) == build_split(contracts, seed=42)

    def test_records_the_seed(self) -> None:
        assert build_split(make_contracts(510), seed=7)["seed"] == 7


@pytest.mark.skipif(not SPLIT_PATH.exists(), reason="split.json not yet generated")
class TestCommittedSplit:
    """Regression guard on the frozen artifact itself."""

    @pytest.fixture(scope="class")
    def split(self) -> dict[str, object]:
        payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        return payload

    def test_uses_the_documented_seed(self, split: dict[str, object]) -> None:
        assert split["seed"] == 42

    def test_sizes_are_as_specified(self, split: dict[str, object]) -> None:
        splits = split["splits"]
        assert isinstance(splits, dict)
        assert splits["dev"]["n"] == DEV_SIZE
        assert splits["golden"]["n"] == GOLDEN_SIZE
        assert splits["dev"]["n"] + splits["golden"]["n"] + splits["reserve"]["n"] == 510

    def test_dev_and_golden_are_disjoint(self, split: dict[str, object]) -> None:
        """The single most important property in the repo: reported numbers come
        from contracts no prompt was ever tuned against."""
        splits = split["splits"]
        assert isinstance(splits, dict)
        dev = set(splits["dev"]["contract_ids"])
        golden = set(splits["golden"]["contract_ids"])
        reserve = set(splits["reserve"]["contract_ids"])
        assert dev & golden == set()
        assert dev & reserve == set()
        assert golden & reserve == set()

    def test_no_duplicate_ids_within_a_split(self, split: dict[str, object]) -> None:
        splits = split["splits"]
        assert isinstance(splits, dict)
        for name, payload in splits.items():
            ids = payload["contract_ids"]
            assert len(ids) == len(set(ids)), f"duplicate contract ids in {name}"
            assert len(ids) == payload["n"]

    def test_golden_covers_every_length_quartile(self, split: dict[str, object]) -> None:
        """Stratification is load-bearing for the cost-curve experiment."""
        splits = split["splits"]
        assert isinstance(splits, dict)
        quartiles = splits["golden"]["by_length_quartile"]
        assert set(quartiles) == {"1", "2", "3", "4"}
        assert all(count > 0 for count in quartiles.values())
