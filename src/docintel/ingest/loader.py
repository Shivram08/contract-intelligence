"""Load CUAD contracts into normalized ``Document`` objects.

The loader's one job is to not corrupt offsets. Everything it does is in service
of that: it reads bytes, decodes UTF-8 explicitly, and disables newline
translation. See ``docs/DATA_AUDIT.md`` check 4 -- CUAD's ``answer_start`` values
are byte-exact against the raw file, and Python's default universal-newline
handling would silently collapse CRLF to LF and shift every offset after the
first line break.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docintel.schemas import Document
from docintel.text import contract_key

__all__ = ["GoldSpan", "load_document", "load_gold_spans", "load_split", "read_contract_text"]


@dataclass(frozen=True, slots=True)
class GoldSpan:
    """One CUAD annotation: a clause category and where it appears."""

    document_id: str
    category: str
    text: str
    char_start: int

    @property
    def char_end(self) -> int:
        return self.char_start + len(self.text)


def read_contract_text(path: Path) -> str:
    """Read a contract exactly as stored.

    ``newline=""`` disables universal-newline translation. Without it, a CRLF
    file loses one character per line before any given offset, and every CUAD
    span in that document silently points at the wrong place.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def load_document(path: Path, metadata: dict[str, str] | None = None) -> Document:
    """Load a single contract. The document id is the NFC-folded filename stem."""
    return Document(
        document_id=contract_key(path.stem),
        text=read_contract_text(path),
        source_path=str(path),
        metadata=metadata or {},
    )


def iter_documents(cuad_dir: Path, document_ids: set[str] | None = None) -> Iterator[Document]:
    """Yield contracts from ``full_contract_txt``, optionally filtered.

    Filtering happens on the NFC-folded id so a caller can pass ids straight from
    ``split.json`` without worrying about which Unicode normal form the
    filesystem used.
    """
    wanted = {contract_key(i) for i in document_ids} if document_ids is not None else None
    for path in sorted((cuad_dir / "full_contract_txt").glob("*.txt")):
        key = contract_key(path.stem)
        if wanted is not None and key not in wanted:
            continue
        yield load_document(path)


def load_gold_spans(cuad_dir: Path, document_ids: set[str] | None = None) -> list[GoldSpan]:
    """Load annotations from ``CUAD_v1.json``.

    Only spans are read. ``master_clauses.csv`` is deliberately not consulted for
    presence: the two disagree on two labels, both cases where the CSV ticks
    "Yes" with no supporting span, and a label with no span cannot be verified by
    the grounding check. The JSON is authoritative.
    """
    payload: dict[str, Any] = json.loads((cuad_dir / "CUAD_v1.json").read_text(encoding="utf-8"))
    wanted = {contract_key(i) for i in document_ids} if document_ids is not None else None

    spans: list[GoldSpan] = []
    for entry in payload["data"]:
        document_id = contract_key(str(entry["title"]))
        if wanted is not None and document_id not in wanted:
            continue
        for qa in entry["paragraphs"][0]["qas"]:
            category = str(qa["id"]).split("__", 1)[1]
            spans.extend(
                GoldSpan(
                    document_id=document_id,
                    category=category,
                    text=str(answer["text"]),
                    char_start=int(answer["answer_start"]),
                )
                for answer in qa["answers"]
            )
    return spans


def load_split(split_path: Path, name: str) -> set[str]:
    """Read one split's contract ids from ``evals/golden/split.json``."""
    payload: dict[str, Any] = json.loads(split_path.read_text(encoding="utf-8"))
    splits = payload["splits"]
    if name not in splits:
        raise KeyError(f"unknown split {name!r}; have {sorted(splits)}")
    return {contract_key(i) for i in splits[name]["contract_ids"]}
