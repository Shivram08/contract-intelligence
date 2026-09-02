"""Asserts the ClauseType enum against data/reference/clause_schema.yaml.

Two sources describe the same 12 clause types: the YAML, which the audit
produced and which records the counts and the substitution rationale, and the
enum, which the code actually uses. If they drift, the code extracts one set of
clauses while the documentation claims another -- and nothing fails.

The definitions matter as much as the names. CLAUDE.md section 2.1 requires
CUAD's own wording verbatim, because extracting against a paraphrased definition
means extracting against a different standard than the ground truth was labelled
with, and the resulting F1 loss looks like a model problem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from docintel.schemas import (
    BOOLEAN_CLAUSES,
    CLAUSE_DEFINITIONS,
    CLAUSE_TIERS,
    CUAD_CATEGORIES,
    ClauseType,
    Tier,
)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "reference" / "clause_schema.yaml"


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    data = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def by_id(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["id"]: entry for entry in schema["clauses"]}


def normalize(text: str) -> str:
    """Collapse whitespace, so YAML line wrapping does not fail a comparison."""
    return " ".join(text.split())


class TestEnumMatchesSchema:
    def test_same_number_of_clause_types(self, schema: dict[str, Any]) -> None:
        assert len(ClauseType) == len(schema["clauses"]) == 12

    def test_same_ids(self, by_id: dict[str, dict[str, Any]]) -> None:
        assert {clause.value for clause in ClauseType} == set(by_id)

    def test_enum_order_matches_schema_order(self, schema: dict[str, Any]) -> None:
        """Order is tier-grouped in both, and the prompt renders them in order."""
        assert [c.value for c in ClauseType] == [e["id"] for e in schema["clauses"]]

    def test_tiers_agree(self, by_id: dict[str, dict[str, Any]]) -> None:
        for clause in ClauseType:
            assert CLAUSE_TIERS[clause].value == by_id[clause.value]["tier"], clause.value

    def test_cuad_category_names_agree(self, by_id: dict[str, dict[str, Any]]) -> None:
        """These strings key into CUAD_v1.json; a typo silently scores zero."""
        for clause in ClauseType:
            assert CUAD_CATEGORIES[clause] == by_id[clause.value]["cuad_category"]


class TestDefinitionsAreVerbatim:
    def test_enum_member_docstrings_are_not_usable_at_runtime(self) -> None:
        """Documents the Python behaviour that caused a real bug.

        The string written after an enum member assignment is discarded; every
        member inherits the class docstring. Relying on it shipped a `get_schema`
        tool that gave the agent the same meta-text as the definition of all
        twelve clauses, with nothing raising.
        """
        assert len({clause.__doc__ for clause in ClauseType}) == 1

    def test_every_clause_type_has_a_definition(self) -> None:
        assert set(CLAUSE_DEFINITIONS) == set(ClauseType)
        for clause in ClauseType:
            assert CLAUSE_DEFINITIONS[clause].strip(), clause.value

    def test_definitions_match_the_schema_descriptions(
        self, by_id: dict[str, dict[str, Any]]
    ) -> None:
        """These strings are what the agent actually sees via `get_schema`."""
        for clause in ClauseType:
            expected = normalize(by_id[clause.value]["description"])
            actual = normalize(CLAUSE_DEFINITIONS[clause])
            assert actual == expected, (
                f"{clause.value}: enum docstring differs from clause_schema.yaml\n"
                f"  enum:   {actual}\n"
                f"  schema: {expected}"
            )

    def test_definitions_are_not_empty_or_placeholder(self) -> None:
        for clause in ClauseType:
            text = normalize(CLAUSE_DEFINITIONS[clause])
            assert len(text) > 20, f"{clause.value} definition looks like a placeholder"
            assert "TODO" not in text


class TestTierStructure:
    def test_all_three_tiers_are_populated(self) -> None:
        assert set(CLAUSE_TIERS.values()) == {
            Tier.NEAR_DETERMINISTIC,
            Tier.REQUIRES_READING,
            Tier.REQUIRES_JUDGEMENT,
        }

    def test_four_clauses_per_tier(self) -> None:
        """Keeps the per-tier results table balanced."""
        for tier in Tier:
            assert sum(1 for t in CLAUSE_TIERS.values() if t is tier) == 4

    def test_every_clause_type_has_a_tier(self) -> None:
        assert set(CLAUSE_TIERS) == set(ClauseType)

    def test_every_clause_type_has_a_cuad_category(self) -> None:
        assert set(CUAD_CATEGORIES) == set(ClauseType)


class TestBooleanClauses:
    def test_boolean_clauses_are_the_ones_with_null_value_type(
        self, by_id: dict[str, dict[str, Any]]
    ) -> None:
        """`value_type: null` in the YAML and BOOLEAN_CLAUSES must agree, or the
        rules will demand a value the prompt tells the model not to produce."""
        from_yaml = {
            clause_id for clause_id, entry in by_id.items() if entry.get("value_type") is None
        }
        assert {c.value for c in BOOLEAN_CLAUSES} == from_yaml

    def test_there_are_six_of_them(self) -> None:
        assert len(BOOLEAN_CLAUSES) == 6

    def test_all_boolean_clauses_are_valid_clause_types(self) -> None:
        assert set(ClauseType) >= BOOLEAN_CLAUSES


class TestSubstitutionIsRecorded:
    def test_the_dropped_clause_is_not_in_the_enum(self, schema: dict[str, Any]) -> None:
        """Most Favored Nation was dropped for having 28 positives."""
        dropped = {entry["dropped"] for entry in schema["substitutions"]}
        assert dropped.isdisjoint(set(CUAD_CATEGORIES.values()))

    def test_the_replacement_is_in_the_enum(self, schema: dict[str, Any]) -> None:
        for entry in schema["substitutions"]:
            assert entry["replacement"] in set(CUAD_CATEGORIES.values())

    def test_every_clause_clears_the_minimum_positive_bar(
        self, by_id: dict[str, dict[str, Any]]
    ) -> None:
        """The selection rule from CLAUDE.md section 2.2, enforced against the
        enum rather than only stated in the YAML."""
        for clause in ClauseType:
            assert by_id[clause.value]["positives"] >= 40, clause.value
