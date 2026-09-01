"""Structural tests for the hand-authored reference data.

`jurisdictions.yaml` and `clause_schema.yaml` are edited by hand and consumed by
the validation layer, so a typo is a runtime failure in a rule rather than a
parse error at import. These tests run without the CUAD download.

The YAML flow-sequence trap is the reason this file exists: `aliases: [nova
scotia, canada]` is two aliases, not one, and reads perfectly fine to a human.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REFERENCE = Path(__file__).resolve().parents[2] / "data" / "reference"
JURISDICTIONS = REFERENCE / "jurisdictions.yaml"
CLAUSE_SCHEMA = REFERENCE / "clause_schema.yaml"

US_STATE_COUNT = 50
CLAUSE_COUNT = 12


@pytest.fixture(scope="module")
def jurisdictions() -> dict[str, Any]:
    data = yaml.safe_load(JURISDICTIONS.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def clause_schema() -> dict[str, Any]:
    data = yaml.safe_load(CLAUSE_SCHEMA.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


class TestJurisdictions:
    def test_ids_are_unique(self, jurisdictions: dict[str, Any]) -> None:
        ids = [entry["id"] for entry in jurisdictions["jurisdictions"]]
        duplicates = {i for i in ids if ids.count(i) > 1}
        assert not duplicates, f"duplicate jurisdiction ids: {duplicates}"

    def test_every_entry_has_id_canonical_and_type(self, jurisdictions: dict[str, Any]) -> None:
        for entry in jurisdictions["jurisdictions"]:
            assert entry.get("id"), entry
            assert entry.get("canonical"), entry
            assert entry.get("type"), entry

    def test_covers_all_fifty_states_plus_dc(self, jurisdictions: dict[str, Any]) -> None:
        """The closed list has to actually be closed, or the rule is decorative."""
        states = [e for e in jurisdictions["jurisdictions"] if e["type"] == "us_state"]
        assert len(states) == US_STATE_COUNT
        assert any(e["id"] == "US-DC" for e in jurisdictions["jurisdictions"])

    def test_aliases_are_strings_not_accidental_lists(self, jurisdictions: dict[str, Any]) -> None:
        """Catches the YAML flow-sequence trap.

        `aliases: [nova scotia, canada]` parses as TWO aliases, "nova scotia" and
        "canada", not the one string it looks like. That silently fails to match
        the real value and, worse, claims "canada" as an alias of Nova Scotia.
        """
        for entry in jurisdictions["jurisdictions"]:
            for alias in entry.get("aliases") or []:
                assert isinstance(alias, str), f"{entry['id']}: non-string alias {alias!r}"

    def test_no_alias_is_claimed_by_two_jurisdictions(self, jurisdictions: dict[str, Any]) -> None:
        """An ambiguous alias makes resolution order-dependent."""
        owners: dict[str, str] = {}
        collisions: list[str] = []
        for entry in jurisdictions["jurisdictions"]:
            keys = [entry["canonical"], *(entry.get("aliases") or [])]
            for key in keys:
                folded = str(key).casefold()
                if folded in owners and owners[folded] != entry["id"]:
                    collisions.append(f"{folded!r}: {owners[folded]} vs {entry['id']}")
                owners[folded] = entry["id"]
        assert not collisions, collisions

    def test_sentinels_are_disjoint_from_jurisdiction_ids(
        self, jurisdictions: dict[str, Any]
    ) -> None:
        sentinel_ids = {s["id"] for s in jurisdictions["sentinels"]}
        jurisdiction_ids = {j["id"] for j in jurisdictions["jurisdictions"]}
        assert sentinel_ids & jurisdiction_ids == set()

    def test_england_and_wales_absorbs_every_observed_spelling(
        self, jurisdictions: dict[str, Any]
    ) -> None:
        """CUAD spells this five ways; all of them must resolve to one id."""
        entry = next(e for e in jurisdictions["jurisdictions"] if e["id"] == "GB-EAW")
        aliases = {a.casefold() for a in entry["aliases"]}
        for observed in ("england", "wales", "england; wales", "england and wales, uk"):
            assert observed in aliases


class TestClauseSchema:
    def test_has_exactly_twelve_clauses(self, clause_schema: dict[str, Any]) -> None:
        assert len(clause_schema["clauses"]) == CLAUSE_COUNT

    def test_clause_ids_are_unique(self, clause_schema: dict[str, Any]) -> None:
        ids = [c["id"] for c in clause_schema["clauses"]]
        assert len(ids) == len(set(ids))

    def test_tiers_are_one_two_or_three(self, clause_schema: dict[str, Any]) -> None:
        assert {c["tier"] for c in clause_schema["clauses"]} == {1, 2, 3}

    def test_tiers_are_evenly_sized(self, clause_schema: dict[str, Any]) -> None:
        """Four per tier keeps the results table balanced across difficulty."""
        for tier in (1, 2, 3):
            assert sum(1 for c in clause_schema["clauses"] if c["tier"] == tier) == 4

    def test_every_clause_clears_the_minimum_positive_bar(
        self, clause_schema: dict[str, Any]
    ) -> None:
        """The selection rule from CLAUDE.md section 2.2, enforced not just stated."""
        for clause in clause_schema["clauses"]:
            assert clause["positives"] >= 40, f"{clause['id']} has {clause['positives']}"

    def test_base_rate_is_consistent_with_positive_count(
        self, clause_schema: dict[str, Any]
    ) -> None:
        """Guards against a hand-edited count drifting from its rate."""
        for clause in clause_schema["clauses"]:
            assert clause["base_rate"] == pytest.approx(clause["positives"] / 510, abs=0.001)

    def test_every_clause_carries_a_description(self, clause_schema: dict[str, Any]) -> None:
        for clause in clause_schema["clauses"]:
            assert clause.get("description"), clause["id"]

    def test_dropped_clause_is_recorded_with_a_reason(self, clause_schema: dict[str, Any]) -> None:
        """No silent substitutions: anything dropped is documented."""
        substitutions = clause_schema["substitutions"]
        assert substitutions
        for entry in substitutions:
            assert entry["dropped"]
            assert entry["reason"]
            assert entry["replacement"]

    def test_replacement_is_actually_in_the_clause_list(
        self, clause_schema: dict[str, Any]
    ) -> None:
        names = {c["cuad_category"] for c in clause_schema["clauses"]}
        for entry in clause_schema["substitutions"]:
            assert entry["replacement"] in names
            assert entry["dropped"] not in names
