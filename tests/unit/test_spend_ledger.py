"""The project-wide spend cap: refuse on worst case, record only live spend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evals.spend import Ledger, SpendCapError


def test_empty_ledger_has_the_whole_cap(tmp_path: Path) -> None:
    ledger = Ledger.load(tmp_path / "ledger.json", cap_usd=25.0)
    assert ledger.spent_usd == 0.0
    assert ledger.remaining_usd == 25.0


def test_refuses_a_run_whose_worst_case_exceeds_what_is_left(tmp_path: Path) -> None:
    ledger = Ledger.load(tmp_path / "ledger.json", cap_usd=10.0)
    ledger.record(8.0, model="m")
    ledger.check(2.0)  # exactly fits
    with pytest.raises(SpendCapError, match="exceeds the remaining"):
        ledger.check(2.01)


def test_records_persist_and_accumulate(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    Ledger.load(path, cap_usd=25.0).record(1.25, model="m", baselines=["5_full_context"])
    Ledger.load(path, cap_usd=25.0).record(0.75, model="m")
    reloaded = Ledger.load(path, cap_usd=25.0)
    assert reloaded.spent_usd == pytest.approx(2.0)
    assert json.loads(path.read_text())["entries"][0]["baselines"] == ["5_full_context"]


def test_zero_cost_runs_leave_no_entry(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    Ledger.load(path).record(0.0, model="m")
    assert not path.exists()
