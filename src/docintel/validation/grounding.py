"""Verbatim grounding: every quoted evidence span must exist in the source.

If a model quotes text that is not in the document, the extraction is a
fabrication and is rejected **mechanically** -- no second model call, no judge,
no heuristic. This is the cheapest hallucination gate that exists and it catches
the failure mode everyone else hand-waves about.

Three things it deliberately does:

1. **Compares on whitespace-normalized text, not raw text.** A model that
   re-flows a quote across line breaks is quoting correctly; a model that invents
   a sentence is not. Collapsing whitespace on *both sides of the comparison*
   separates those. The stored document is never normalized -- doing that would
   shift every offset (``docs/DATA_AUDIT.md`` check 4).

2. **Repairs offsets rather than only judging them.** If the quote is genuinely
   present but ``char_start`` is wrong, the span is relocated and reported as
   ``RELOCATED``. Models are far better at copying text than at counting
   characters, and throwing away a correct quote over arithmetic would inflate
   the hallucination rate with something that is not hallucination.

3. **Reports why, not just whether.** ``GroundingStatus`` distinguishes
   fabrication from offset drift from a truncated document, because the remedy
   differs and an aggregate "grounding violation rate" that mixes them is not
   actionable.

### What this cannot catch

Worth stating plainly, since the honest limit is more useful than the claim.
A quote can be perfectly verbatim and still support a wrong conclusion:

- **Quote-level correct, inference-level wrong.** Quoting a governing-law clause
  accurately and then reporting the wrong state passes grounding entirely.
- **Negation and scope.** "Liability is not capped" is a verbatim quote that
  grounds a ``cap_on_liability`` extraction perfectly well.
- **Cherry-picking.** A real sentence lifted out of a carve-out or a definitions
  section, presented as the operative clause.
- **Wrong document.** Nothing here checks that the quote came from the document
  the agent was asked about, only that it appears in the text it was given.

Those are what the deterministic rules, the reranker, and the LLM-as-judge are
for. Grounding is a floor, not a ceiling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from docintel.schemas import ClauseExtraction, Document, Evidence
from docintel.text import normalize_whitespace

__all__ = [
    "EvidenceCheck",
    "GroundingReport",
    "GroundingStatus",
    "check_clause",
    "check_evidence",
    "check_extractions",
]


class GroundingStatus(StrEnum):
    """Why an evidence span passed or failed."""

    EXACT = "exact"
    """Byte-identical at the stated offsets. Nothing to do."""

    NORMALIZED = "normalized"
    """Matches at the stated offsets once whitespace is collapsed on both sides.
    A re-flowed quote, which is correct quoting."""

    RELOCATED = "relocated"
    """The quote is in the document, but not where the model said. Offsets are
    repaired; the extraction stands."""

    NOT_FOUND = "not_found"
    """The quote does not appear in the document at all. A fabrication."""

    OUT_OF_BOUNDS = "out_of_bounds"
    """Offsets fall outside the document and the text is absent too."""

    EMPTY = "empty"
    """A blank quote. Vacuously ungrounded -- it supports nothing."""

    @property
    def is_grounded(self) -> bool:
        """Whether the evidence is usable. RELOCATED counts: the quote is real."""
        return self in {
            GroundingStatus.EXACT,
            GroundingStatus.NORMALIZED,
            GroundingStatus.RELOCATED,
        }


@dataclass(frozen=True, slots=True)
class EvidenceCheck:
    """One evidence span's verdict, with a corrected span where possible."""

    status: GroundingStatus
    evidence: Evidence
    #: Offsets that actually contain the quote, when they could be found.
    #: Equal to the input offsets for EXACT and NORMALIZED.
    repaired: Evidence | None = None
    #: How many other places the quote appears. >0 with RELOCATED means the
    #: repair picked one of several candidates and may have picked wrong.
    other_occurrences: int = 0

    @property
    def is_grounded(self) -> bool:
        return self.status.is_grounded


def check_evidence(document: Document, evidence: Evidence) -> EvidenceCheck:
    """Verify one quote against the document."""
    quote = evidence.quote
    if not quote.strip():
        return EvidenceCheck(status=GroundingStatus.EMPTY, evidence=evidence)

    text = document.text
    at_offset = text[evidence.char_start : evidence.char_end]

    if at_offset == quote:
        return EvidenceCheck(status=GroundingStatus.EXACT, evidence=evidence, repaired=evidence)

    if at_offset and normalize_whitespace(at_offset) == normalize_whitespace(quote):
        # A re-flowed quote. The offsets are right; only the whitespace differs.
        return EvidenceCheck(
            status=GroundingStatus.NORMALIZED, evidence=evidence, repaired=evidence
        )

    located = _locate(text, quote)
    if located is None:
        out_of_bounds = evidence.char_end > len(text) or evidence.char_start >= len(text)
        return EvidenceCheck(
            status=(GroundingStatus.OUT_OF_BOUNDS if out_of_bounds else GroundingStatus.NOT_FOUND),
            evidence=evidence,
        )

    start, end, occurrences = located
    return EvidenceCheck(
        status=GroundingStatus.RELOCATED,
        evidence=evidence,
        repaired=Evidence(
            quote=text[start:end],
            char_start=start,
            char_end=end,
            chunk_id=evidence.chunk_id,
        ),
        other_occurrences=occurrences - 1,
    )


def _locate(text: str, quote: str) -> tuple[int, int, int] | None:
    """Find a quote in the document, tolerating whitespace differences.

    Returns ``(char_start, char_end, occurrence_count)`` or None.

    The exact search runs first because it is cheap and covers the common case.
    The whitespace-tolerant fallback builds a regex from the quote's tokens
    separated by ``\\s+``, which finds a quote whose line breaks differ from the
    source without ever normalizing the source itself.
    """
    index = text.find(quote)
    if index >= 0:
        return index, index + len(quote), text.count(quote)

    tokens = quote.split()
    if not tokens:
        return None

    # `\s+` between tokens, so "State of\nDelaware" matches "State of Delaware".
    pattern = re.compile(r"\s+".join(re.escape(token) for token in tokens))
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[0].start(), matches[0].end(), len(matches)


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """Grounding outcome for one document's extractions."""

    checks: list[EvidenceCheck]
    #: Extractions with offsets repaired where a quote was merely misplaced.
    repaired_clauses: list[ClauseExtraction]

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def ungrounded(self) -> list[EvidenceCheck]:
        return [c for c in self.checks if not c.is_grounded]

    @property
    def violation_rate(self) -> float:
        """Share of evidence spans that are not grounded.

        Reported as a first-class metric beside F1. The definition counts spans
        rather than documents: one fabricated quote among ten real ones is a 10%
        violation rate, not a 100% failed document.
        """
        return len(self.ungrounded) / self.total if self.total else 0.0

    def count(self, status: GroundingStatus) -> int:
        return sum(1 for c in self.checks if c.status is status)


def check_clause(
    document: Document, clause: ClauseExtraction
) -> tuple[list[EvidenceCheck], ClauseExtraction]:
    """Check one clause's evidence, returning checks and a repaired clause.

    Ungrounded evidence is **dropped** from the repaired clause. A clause left
    with ``present=True`` and no evidence then trips the ``presence_requires_
    evidence`` rule, so a fabricating extraction fails validation rather than
    passing with a quietly shortened evidence list.
    """
    checks = [check_evidence(document, item) for item in clause.evidence]
    kept = [check.repaired for check in checks if check.repaired is not None]
    if kept == list(clause.evidence):
        return checks, clause
    return checks, clause.model_copy(update={"evidence": kept})


def check_extractions(document: Document, clauses: list[ClauseExtraction]) -> GroundingReport:
    """Check every clause's evidence against the source document."""
    all_checks: list[EvidenceCheck] = []
    repaired: list[ClauseExtraction] = []
    for clause in clauses:
        checks, fixed = check_clause(document, clause)
        all_checks.extend(checks)
        repaired.append(fixed)
    return GroundingReport(checks=all_checks, repaired_clauses=repaired)
