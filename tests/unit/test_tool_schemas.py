"""Tests that the tool schemas are valid for the API, not just for us.

`submit_extraction` runs with `strict: true`. Strict mode has a documented
subset of JSON Schema, and anything outside it is a 400 on *every* request --
not a degraded result, a hard failure on the first live call.

The Python SDK strips unsupported keywords only on the `parse()` /
`output_config` path. A tool passed as a raw dict in `tools=[...]` is sent
exactly as written, so nothing removes them for us.

Unsupported in strict mode (per the API docs):
  - numerical constraints: minimum, maximum, multipleOf
  - string constraints: minLength, maxLength
  - recursive schemas
  - additionalProperties set to anything but false
Type arrays (`{"type": ["string", "null"]}`) are also not among the documented
forms; `anyOf` is.
"""

from __future__ import annotations

from typing import Any

import pytest

from docintel.agent.tools import TOOL_NAMES, build_tool_schemas

UNSUPPORTED_IN_STRICT = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
    }
)


def walk(node: Any) -> list[tuple[str, Any]]:
    """Every (key, value) pair anywhere in a nested schema."""
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append((key, value))
            found.extend(walk(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(walk(item))
    return found


@pytest.fixture(scope="module")
def tools() -> list[dict[str, Any]]:
    return build_tool_schemas()


@pytest.fixture(scope="module")
def strict_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tool for tool in tools if tool.get("strict")]


class TestToolInventory:
    def test_all_declared_tools_are_built(self, tools: list[dict[str, Any]]) -> None:
        assert [tool["name"] for tool in tools] == list(TOOL_NAMES)

    def test_every_tool_has_a_description_and_schema(self, tools: list[dict[str, Any]]) -> None:
        for tool in tools:
            assert tool["description"].strip(), tool["name"]
            assert tool["input_schema"]["type"] == "object", tool["name"]

    def test_submit_extraction_is_the_only_strict_tool(
        self, strict_tools: list[dict[str, Any]]
    ) -> None:
        assert [tool["name"] for tool in strict_tools] == ["submit_extraction"]

    def test_tool_list_is_deterministic(self) -> None:
        """Tools render before system and messages in the cached prefix, so a
        varying tool list invalidates the cache for everything after it."""
        assert build_tool_schemas() == build_tool_schemas()


class TestStrictModeCompliance:
    def test_no_unsupported_keywords_anywhere_in_a_strict_schema(
        self, strict_tools: list[dict[str, Any]]
    ) -> None:
        """The bug this file exists for. `minimum: 0` on an offset field would
        400 on every single request."""
        for tool in strict_tools:
            offenders = sorted(
                {key for key, _ in walk(tool["input_schema"]) if key in UNSUPPORTED_IN_STRICT}
            )
            assert not offenders, f"{tool['name']} uses strict-unsupported keywords: {offenders}"

    def test_nullable_fields_use_anyof_not_a_type_array(
        self, strict_tools: list[dict[str, Any]]
    ) -> None:
        for tool in strict_tools:
            for key, value in walk(tool["input_schema"]):
                if key == "type":
                    assert not isinstance(value, list), (
                        f"{tool['name']} uses a type array {value}; strict mode takes "
                        "a single type or anyOf"
                    )

    def test_every_object_forbids_additional_properties(
        self, strict_tools: list[dict[str, Any]]
    ) -> None:
        for tool in strict_tools:
            for key, value in walk(tool["input_schema"]):
                if key == "type" and value == "object":
                    pass  # checked structurally below
            objects = [
                node
                for _, node in walk(tool["input_schema"])
                if isinstance(node, dict) and node.get("type") == "object"
            ]
            objects.append(tool["input_schema"])
            for obj in objects:
                assert obj.get("additionalProperties") is False, tool["name"]

    def test_every_object_declares_required(self, strict_tools: list[dict[str, Any]]) -> None:
        objects = [
            node
            for tool in strict_tools
            for _, node in walk(tool["input_schema"])
            if isinstance(node, dict) and node.get("type") == "object"
        ]
        for obj in objects:
            assert "required" in obj

    def test_required_lists_every_property(self, strict_tools: list[dict[str, Any]]) -> None:
        """Strict mode requires all properties be listed as required."""
        objects = [
            node
            for tool in strict_tools
            for _, node in walk(tool["input_schema"])
            if isinstance(node, dict) and node.get("type") == "object" and "properties" in node
        ]
        objects.extend(tool["input_schema"] for tool in strict_tools)
        for obj in objects:
            if "properties" not in obj:
                continue
            assert set(obj["required"]) == set(obj["properties"]), obj.get("properties", {}).keys()


class TestClauseSchemaShape:
    def _clause_items(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        submit = next(t for t in tools if t["name"] == "submit_extraction")
        items = submit["input_schema"]["properties"]["clauses"]["items"]
        assert isinstance(items, dict)
        return items

    def test_clause_type_is_a_closed_enum(self, tools: list[dict[str, Any]]) -> None:
        from docintel.schemas import ClauseType

        enum = self._clause_items(tools)["properties"]["clause_type"]["enum"]
        assert enum == [clause.value for clause in ClauseType]

    def test_value_and_raw_text_are_nullable(self, tools: list[dict[str, Any]]) -> None:
        properties = self._clause_items(tools)["properties"]
        for field in ("value", "raw_text"):
            assert properties[field]["anyOf"] == [{"type": "string"}, {"type": "null"}]

    def test_confidence_is_a_number(self, tools: list[dict[str, Any]]) -> None:
        assert self._clause_items(tools)["properties"]["confidence"]["type"] == "number"

    def test_bounds_are_documented_in_prose_since_they_cannot_be_in_the_schema(
        self, tools: list[dict[str, Any]]
    ) -> None:
        """The constraint has to reach the model somehow; strict mode forbids
        expressing it as a keyword, so it lives in the description."""
        confidence = self._clause_items(tools)["properties"]["confidence"]
        assert "between 0 and 1" in confidence["description"]
