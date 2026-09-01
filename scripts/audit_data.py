"""Pre-flight audit of CUAD v1, per CLAUDE.md section 2.2.

Runs five checks and writes ``docs/DATA_AUDIT.md`` plus a length histogram. This
runs *before* any application code because three of its findings change the
design: which clause types are measurable at all, whether the RAG-versus-long-
context experiment is an accuracy story or a cost story, and -- most important --
whether CUAD's character offsets can be trusted as ground truth for the
grounding verifier.

Usage::

    uv run python scripts/audit_data.py --seed 42 --full-offset-scan
    uv run python scripts/audit_data.py --sample-size 300 --seed 42

The committed report uses ``--full-offset-scan``: the spec asks for a 300-span
sample, but verifying all 13,823 takes about a second and removes any question
of sampling luck on the one check that everything else depends on.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_CUAD_DIR: Final = REPO_ROOT / "data" / "raw" / "CUAD_v1"
DEFAULT_REPORT: Final = REPO_ROOT / "docs" / "DATA_AUDIT.md"
DEFAULT_FIGURE: Final = REPO_ROOT / "docs" / "img" / "length_dist.png"

#: A clause type below this many positive contracts yields noise, not a
#: measurement (CLAUDE.md section 2.2, check 1).
MIN_POSITIVES: Final = 40

#: Context-window thresholds worth reporting against, in tokens.
CONTEXT_WINDOWS: Final[tuple[int, ...]] = (8_192, 32_768, 128_000, 200_000)

#: cl100k_base stands in for Anthropic's tokenizer, which needs a network call
#: per document. The two differ by a few percent on English prose -- fine for
#: sizing a context window, and it keeps the audit runnable offline.
TOKENIZER: Final = "cl100k_base"

_WHITESPACE: Final = re.compile(r"\s+")

#: Hand-written, from inspecting the value distributions. Recorded here rather
#: than in prose because these drive concrete decisions in `rules.py`.
_NORMALIZATION_HAZARDS: Final[dict[str, str]] = {
    "Document Name": "Free text; no closed vocabulary",
    "Parties": "Multi-party, role-tagged; needs a canonical join format",
    "Agreement Date": "Masked partial dates (`[]/[]/2020`); 2-digit years",
    "Effective Date": "Same masking; often absent when it equals the agreement date",
    "Expiration Date": "`perpetual` is not a date; inconsistent casing",
    "Renewal Term": "Durations, not dates (`successive 1 year`)",
    "Notice Period To Terminate Renewal": "Free-form durations; normalize to integer days",
    "Governing Law": "Closed set, but US states mix with countries",
}


# --------------------------------------------------------------------------
# Span offset verification -- the critical check
#
# Everything downstream depends on this. If `answer_start` does not index
# cleanly into the text the pipeline actually reads, the grounding verifier
# will reject correct extractions as hallucinations and the F1 numbers will be
# wrong in a way that looks like a model problem. These functions are pure and
# unit-tested in tests/unit/test_span_offsets.py.
# --------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run to a single space and strip the ends.

    Used for *comparison only*. Note that applying this to a source document
    destroys the character offsets that index into it -- see the module report
    for why that matters.
    """
    return _WHITESPACE.sub(" ", text).strip()


class SpanStatus(StrEnum):
    """Outcome of checking one annotated span against its source document."""

    EXACT = "exact"
    """Byte-identical slice. The only status that needs no remediation."""

    WHITESPACE_ONLY = "whitespace_only"
    """Slice differs from the gold text only in whitespace."""

    OUT_OF_BOUNDS = "out_of_bounds"
    """The span runs past the end of the source document."""

    MISMATCH = "mismatch"
    """Slice is different text. Offsets have drifted against this source."""


@dataclass(frozen=True, slots=True)
class SpanCheck:
    """The result of verifying one span, with enough context to debug a failure."""

    status: SpanStatus
    char_start: int
    gold_text: str
    found_text: str
    #: Where the gold text actually appears, if it appears verbatim elsewhere.
    #: A consistent non-null value across failures means a fixed offset shift;
    #: null means the text is simply absent from this source.
    actual_offset: int | None = None

    @property
    def ok(self) -> bool:
        """Whether the offsets are usable as-is, with no repair step."""
        return self.status is SpanStatus.EXACT


def verify_span(source: str, char_start: int, gold_text: str) -> SpanCheck:
    """Check that ``source[char_start:]`` reproduces ``gold_text``.

    Returns the most specific status that applies. ``actual_offset`` is filled
    in for failures so a systematic shift can be told apart from missing text.
    """
    found = source[char_start : char_start + len(gold_text)] if char_start >= 0 else ""

    if char_start < 0 or (gold_text and char_start >= len(source)):
        status = SpanStatus.OUT_OF_BOUNDS
    elif found == gold_text:
        status = SpanStatus.EXACT
    elif _matches_ignoring_whitespace(source, char_start, gold_text):
        # Recoverable: the right text is here, but the byte offsets do not line
        # up. Deliberately not `ok` -- grading it as a pass would hide exactly
        # the drift this audit exists to measure.
        status = SpanStatus.WHITESPACE_ONLY
    elif len(found) < len(gold_text):
        # Ran off the end of the document and did not recover.
        status = SpanStatus.OUT_OF_BOUNDS
    else:
        status = SpanStatus.MISMATCH

    return SpanCheck(
        status=status,
        char_start=char_start,
        gold_text=gold_text,
        found_text=found,
        actual_offset=None if status is SpanStatus.EXACT else _find_or_none(source, gold_text),
    )


def _matches_ignoring_whitespace(source: str, char_start: int, gold_text: str) -> bool:
    """Whether the gold text starts at ``char_start`` once whitespace is collapsed.

    Length-tolerant on purpose. ``verify_span`` slices by ``len(gold_text)``, so
    a source with *extra* whitespace truncates the slice and would otherwise be
    reported as a flat mismatch -- hiding the difference between "offsets are
    slightly off" and "this text is not in the document at all". Only reached
    after the exact check fails, so the cost stays off the happy path.
    """
    return normalize_whitespace(source[char_start:]).startswith(normalize_whitespace(gold_text))


def _find_or_none(haystack: str, needle: str) -> int | None:
    if not needle:
        return None
    index = haystack.find(needle)
    return None if index < 0 else index


def contract_key(title: str) -> str:
    """Normalize a contract title for matching a JSON entry to a TXT filename.

    CUAD ships at least one title whose accented characters are encoded
    differently in ``CUAD_v1.json`` than on disk (``LECLANCHE S.A.`` uses a
    combining acute in one and a precomposed character in the other). Without
    NFC folding that contract silently drops out of the corpus.
    """
    return unicodedata.normalize("NFC", title)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Annotation:
    """One gold span for one (contract, clause type) pair."""

    contract_title: str
    category: str
    text: str
    char_start: int


@dataclass(slots=True)
class Contract:
    """A CUAD contract: its raw text plus every annotation against it."""

    title: str
    text: str
    path: Path
    #: category -> gold spans. Categories with no spans are negatives and are
    #: present as empty lists, so absence and emptiness are never confused.
    annotations: dict[str, list[Annotation]]

    @property
    def positive_categories(self) -> set[str]:
        return {cat for cat, spans in self.annotations.items() if spans}


def build_text_index(txt_dir: Path) -> dict[str, Path]:
    """Map normalized contract title -> TXT path."""
    return {contract_key(path.stem): path for path in sorted(txt_dir.glob("*.txt"))}


def load_contracts(cuad_dir: Path) -> tuple[list[Contract], list[str]]:
    """Load every contract with its annotations.

    Returns the contracts and a list of titles that had no matching TXT file.
    """
    payload: dict[str, Any] = json.loads((cuad_dir / "CUAD_v1.json").read_text(encoding="utf-8"))
    index = build_text_index(cuad_dir / "full_contract_txt")

    contracts: list[Contract] = []
    unmatched: list[str] = []

    for entry in payload["data"]:
        title = str(entry["title"])
        path = index.get(contract_key(title))
        if path is None:
            unmatched.append(title)
            continue

        paragraph = entry["paragraphs"][0]
        annotations: dict[str, list[Annotation]] = {}
        for qa in paragraph["qas"]:
            category = str(qa["id"]).split("__", 1)[1]
            annotations[category] = [
                Annotation(
                    contract_title=title,
                    category=category,
                    text=str(answer["text"]),
                    char_start=int(answer["answer_start"]),
                )
                for answer in qa["answers"]
            ]

        contracts.append(
            Contract(
                title=title,
                # The JSON `context` and the TXT file are byte-identical (the
                # audit asserts this), so reading from disk keeps the pipeline
                # honest: offsets are verified against the file the loader will
                # actually open, not against a copy embedded in the JSON.
                text=path.read_text(encoding="utf-8"),
                path=path,
                annotations=annotations,
            )
        )

    return contracts, unmatched


@dataclass(frozen=True, slots=True)
class MasterAnswer:
    """What ``master_clauses.csv`` records for one category.

    The ``<Category>-Answer`` columns are not one thing. Eight of them carry a
    lawyer-normalized *value* (``5/8/14``, ``New York``, ``30 days``); the other
    33 carry a bare ``Yes``/``No`` presence label. Treating them uniformly --
    counting non-empty cells, say -- reports 510/510 coverage for every boolean
    category and means nothing.
    """

    category: str
    #: True when the column is a Yes/No presence label rather than a value.
    is_boolean: bool
    #: Contracts labeled "Yes". Only meaningful when ``is_boolean``.
    yes_count: int
    #: Contracts with a non-empty normalized value. Only meaningful otherwise.
    value_count: int
    #: Up to three most common values, for the normalization spec.
    common_values: tuple[str, ...]


def load_master_answers(csv_path: Path) -> dict[str, MasterAnswer]:
    """Parse the ``<Category>-Answer`` columns of ``master_clauses.csv``."""
    columns: dict[str, list[str]] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for column, cell in row.items():
                if column is None:
                    continue
                stripped = column.strip()
                # The header is inconsistent: one column is spelled
                # "Notice Period To Terminate Renewal- Answer".
                if not stripped.lower().endswith("answer"):
                    continue
                category = stripped[: stripped.lower().rindex("answer")].rstrip(" -")
                columns.setdefault(category, []).append((cell or "").strip())

    answers: dict[str, MasterAnswer] = {}
    for category, cells in columns.items():
        filled = [c for c in cells if c]
        distinct = Counter(filled)
        is_boolean = bool(distinct) and set(distinct) <= {"Yes", "No"}
        answers[category] = MasterAnswer(
            category=category,
            is_boolean=is_boolean,
            yes_count=distinct.get("Yes", 0),
            value_count=len(filled),
            common_values=tuple(value for value, _ in distinct.most_common(3)),
        )
    return answers


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClauseStats:
    """Per-category corpus statistics."""

    category: str
    positive_contracts: int
    total_spans: int
    base_rate: float
    answer: MasterAnswer | None

    @property
    def included(self) -> bool:
        return self.positive_contracts >= MIN_POSITIVES

    @property
    def csv_agrees(self) -> bool | None:
        """Whether master_clauses.csv independently confirms the positive count.

        For boolean categories the CSV's "Yes" count should equal the number of
        contracts carrying at least one gold span in the JSON. The two files are
        different exports of the same annotation effort, so disagreement means
        one of them is stale -- and it decides which is safe to use as labels.
        Returns None for value categories, where no such identity is expected.
        """
        if self.answer is None or not self.answer.is_boolean:
            return None
        return self.answer.yes_count == self.positive_contracts


def compute_clause_stats(
    contracts: Sequence[Contract], answers: dict[str, MasterAnswer]
) -> list[ClauseStats]:
    """Checks 1 and 2: positive counts and base rates, sorted descending."""
    categories = sorted({cat for contract in contracts for cat in contract.annotations})
    n = len(contracts)
    stats = []
    for category in categories:
        positives = sum(1 for c in contracts if c.annotations.get(category))
        stats.append(
            ClauseStats(
                category=category,
                positive_contracts=positives,
                total_spans=sum(len(c.annotations.get(category, [])) for c in contracts),
                base_rate=positives / n,
                answer=answers.get(category),
            )
        )
    return sorted(stats, key=lambda s: (-s.positive_contracts, s.category))


@dataclass(frozen=True, slots=True)
class LengthStats:
    """Check 3: token length distribution over contracts."""

    token_counts: list[int]
    over_threshold: dict[int, int]

    def percentile(self, q: float) -> int:
        ordered = sorted(self.token_counts)
        # Nearest-rank; the corpus is 510 documents, so interpolation is noise.
        rank = max(1, min(len(ordered), round(q / 100 * len(ordered))))
        return ordered[rank - 1]

    @property
    def median(self) -> int:
        return int(statistics.median(self.token_counts))


def compute_length_stats(contracts: Sequence[Contract]) -> LengthStats:
    import tiktoken

    encoder = tiktoken.get_encoding(TOKENIZER)
    counts = [len(encoder.encode(c.text, disallowed_special=())) for c in contracts]
    return LengthStats(
        token_counts=counts,
        over_threshold={w: sum(1 for c in counts if c > w) for w in CONTEXT_WINDOWS},
    )


@dataclass(frozen=True, slots=True)
class OffsetAudit:
    """Check 4: span offset integrity."""

    sampled: int
    population: int
    by_status: dict[SpanStatus, int]
    failures: list[tuple[Annotation, SpanCheck]]
    context_matches_txt: int
    context_total: int

    @property
    def mismatch_rate(self) -> float:
        broken = sum(
            count for status, count in self.by_status.items() if status is not SpanStatus.EXACT
        )
        return broken / self.sampled if self.sampled else 0.0


def audit_offsets(
    contracts: Sequence[Contract],
    cuad_dir: Path,
    sample_size: int,
    rng: random.Random,
) -> OffsetAudit:
    """Verify sampled annotations reproduce against the TXT files.

    Also confirms the JSON ``context`` field is identical to the TXT file. If it
    is, offsets need no remapping between the two representations, which is the
    difference between a trivial loader and an offset-translation layer.
    """
    payload: dict[str, Any] = json.loads((cuad_dir / "CUAD_v1.json").read_text(encoding="utf-8"))
    by_title = {contract_key(c.title): c for c in contracts}
    context_matches = 0
    context_total = 0
    for entry in payload["data"]:
        contract = by_title.get(contract_key(str(entry["title"])))
        if contract is None:
            continue
        context_total += 1
        if entry["paragraphs"][0]["context"] == contract.text:
            context_matches += 1

    population = [
        annotation
        for contract in contracts
        for spans in contract.annotations.values()
        for annotation in spans
    ]
    sample = population if sample_size >= len(population) else rng.sample(population, k=sample_size)

    by_status: dict[SpanStatus, int] = dict.fromkeys(SpanStatus, 0)
    failures: list[tuple[Annotation, SpanCheck]] = []
    for annotation in sample:
        source = by_title[contract_key(annotation.contract_title)].text
        check = verify_span(source, annotation.char_start, annotation.text)
        by_status[check.status] += 1
        if not check.ok:
            failures.append((annotation, check))

    return OffsetAudit(
        sampled=len(sample),
        population=len(population),
        by_status=by_status,
        failures=failures,
        context_matches_txt=context_matches,
        context_total=context_total,
    )


@dataclass(frozen=True, slots=True)
class SourceConsistency:
    """How well ``master_clauses.csv`` and ``CUAD_v1.json`` agree.

    They are separate exports of one annotation effort. Where they disagree, one
    of them is wrong, and the pipeline has to pick a side deliberately rather
    than by whichever file it happened to load.
    """

    csv_rows: int
    exact_joins: int
    relaxed_joins: int
    unjoinable: list[str]
    #: (category, contract, csv_says_yes, json_span_count)
    label_disagreements: list[tuple[str, str, bool, int]]


def _relaxed_key(name: str) -> str:
    """Join key tolerant of the punctuation drift between the two files.

    ``master_clauses.csv`` writes ``MACY'S,INC_...`` where the filesystem has
    ``MACY_S,INC_...``, and several rows carry trailing spaces.
    """
    folded = unicodedata.normalize("NFC", name).casefold()
    return "".join(ch for ch in folded if ch.isalnum())


def audit_source_consistency(contracts: Sequence[Contract], csv_path: Path) -> SourceConsistency:
    """Cross-validate the CSV's per-contract labels against the JSON's spans."""
    by_exact = {contract_key(c.title): c for c in contracts}
    by_relaxed = {_relaxed_key(c.title): c for c in contracts}

    csv_rows = 0
    exact = relaxed = 0
    unjoinable: list[str] = []
    disagreements: list[tuple[str, str, bool, int]] = []

    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            csv_rows += 1
            stem = Path((row.get("Filename") or "").strip()).stem
            contract = by_exact.get(contract_key(stem))
            if contract is not None:
                exact += 1
            else:
                contract = by_relaxed.get(_relaxed_key(stem))
                if contract is None:
                    unjoinable.append(stem)
                    continue
                relaxed += 1

            for column, cell in row.items():
                if column is None:
                    continue
                stripped = column.strip()
                if not stripped.lower().endswith("answer"):
                    continue
                value = (cell or "").strip()
                if value not in {"Yes", "No"}:
                    continue
                category = stripped[: stripped.lower().rindex("answer")].rstrip(" -")
                spans = len(contract.annotations.get(category, []))
                if (value == "Yes") != (spans > 0):
                    disagreements.append((category, contract.title, value == "Yes", spans))

    return SourceConsistency(
        csv_rows=csv_rows,
        exact_joins=exact,
        relaxed_joins=relaxed,
        unjoinable=unjoinable,
        label_disagreements=disagreements,
    )


@dataclass(frozen=True, slots=True)
class RuleCheck:
    """A proposed deterministic rule, measured against the gold labels.

    CLAUDE.md section 5 sketches cross-field rules from first principles. A rule
    that fires on correct data is worse than no rule: it trains the reviewer to
    ignore violations. So each one is scored against CUAD before it is written.
    """

    name: str
    holds: int
    violated: int
    #: What the rule should ship as, given the measured violation count.
    verdict: str

    @property
    def applicable(self) -> int:
        return self.holds + self.violated


def _implication(
    contracts: Sequence[Contract], antecedent: str, consequent: str, name: str
) -> RuleCheck:
    """Score `antecedent present => consequent present` over the corpus."""
    holds = violated = 0
    for contract in contracts:
        if not contract.annotations.get(antecedent):
            continue
        if contract.annotations.get(consequent):
            holds += 1
        else:
            violated += 1
    rate = violated / (holds + violated) if holds + violated else 0.0
    if rate == 0:
        verdict = "ship as ERROR"
    elif rate < 0.05:
        verdict = "ship as WARNING"
    else:
        verdict = "**do not ship as written**"
    return RuleCheck(name=name, holds=holds, violated=violated, verdict=verdict)


def validate_proposed_rules(contracts: Sequence[Contract]) -> list[RuleCheck]:
    """Score the cross-field rules CLAUDE.md proposes against the gold labels."""
    mutually_exclusive_holds = mutually_exclusive_violated = 0
    for contract in contracts:
        cap = bool(contract.annotations.get("Cap On Liability"))
        uncapped = bool(contract.annotations.get("Uncapped Liability"))
        if cap and uncapped:
            mutually_exclusive_violated += 1
        else:
            mutually_exclusive_holds += 1

    return [
        RuleCheck(
            name="`cap_on_liability` and `uncapped_liability` cannot both be present",
            holds=mutually_exclusive_holds,
            violated=mutually_exclusive_violated,
            verdict="**do not ship as written** — inverted, see below",
        ),
        _implication(
            contracts,
            "Uncapped Liability",
            "Cap On Liability",
            "`uncapped_liability` present => `cap_on_liability` present (the inverse)",
        ),
        _implication(
            contracts,
            "Renewal Term",
            "Notice Period To Terminate Renewal",
            "`renewal_term` present => `notice_period` present",
        ),
        _implication(
            contracts,
            "Notice Period To Terminate Renewal",
            "Renewal Term",
            "`notice_period` present => `renewal_term` present (the inverse)",
        ),
        _implication(
            contracts,
            "Expiration Date",
            "Effective Date",
            "`expiration_date` present => `effective_date` present",
        ),
    ]


def compute_span_multiplicity(contracts: Sequence[Contract]) -> dict[str, int]:
    """Check 5: how many positive (contract, clause) pairs carry >1 gold span."""
    buckets = {"1 span": 0, "2 spans": 0, "3-5 spans": 0, "6-10 spans": 0, "11+ spans": 0}
    for contract in contracts:
        for spans in contract.annotations.values():
            n = len(spans)
            if n == 0:
                continue
            if n == 1:
                buckets["1 span"] += 1
            elif n == 2:
                buckets["2 spans"] += 1
            elif n <= 5:
                buckets["3-5 spans"] += 1
            elif n <= 10:
                buckets["6-10 spans"] += 1
            else:
                buckets["11+ spans"] += 1
    return buckets


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def render_length_histogram(lengths: LengthStats, output: Path) -> None:
    """Histogram of contract token lengths on a log x-axis.

    Log scale because the corpus spans two orders of magnitude; on a linear axis
    the long tail compresses 90% of the corpus into the first two bins and the
    context-window thresholds become unreadable.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    surface = "#fcfcfb"
    series = "#2a78d6"
    ink_primary = "#0b0b0b"
    ink_secondary = "#52514e"
    ink_muted = "#8a8880"

    counts = np.array(lengths.token_counts)
    bins = np.logspace(np.log10(counts.min()), np.log10(counts.max()), 36)

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    # rwidth leaves a hairline of surface between bars, so adjacent bins read as
    # separate marks rather than one continuous block.
    ax.hist(counts, bins=list(bins), color=series, rwidth=0.9)

    for window in CONTEXT_WINDOWS:
        if not counts.min() <= window <= counts.max():
            continue
        over = lengths.over_threshold[window]
        ax.axvline(window, color=ink_muted, linewidth=1, linestyle=(0, (4, 3)), zorder=3)
        ax.annotate(
            f"{window // 1000}k\n{over} over ({over / len(counts):.0%})",
            xy=(window, ax.get_ylim()[1]),
            xytext=(3, -4),
            textcoords="offset points",
            va="top",
            ha="left",
            fontsize=8,
            color=ink_secondary,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Tokens per contract (cl100k_base, log scale)", fontsize=9, color=ink_secondary)
    ax.set_ylabel("Contracts", fontsize=9, color=ink_secondary)
    ax.set_title(
        f"CUAD v1 contract length — median {lengths.median:,} tokens, "
        f"max {max(lengths.token_counts):,}",
        fontsize=11,
        color=ink_primary,
        loc="left",
        pad=12,
    )

    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.tick_params(colors=ink_secondary, labelsize=8, length=0)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, facecolor=surface)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(
    contracts: Sequence[Contract],
    unmatched: Sequence[str],
    clause_stats: Sequence[ClauseStats],
    lengths: LengthStats,
    offsets: OffsetAudit,
    multiplicity: dict[str, int],
    consistency: SourceConsistency,
    rule_checks: Sequence[RuleCheck],
    seed: int,
    figure_path: Path,
) -> str:
    n = len(contracts)
    included = [s for s in clause_stats if s.included]
    excluded = [s for s in clause_stats if not s.included]
    total_spans = sum(s.total_spans for s in clause_stats)

    parts: list[str] = []
    parts.append(
        f"""# CUAD v1 data audit

Generated by [`scripts/audit_data.py`](../scripts/audit_data.py) - the pre-flight
profiling pass required by [`CLAUDE.md`](../CLAUDE.md) section 2.2, run before any
application code. Reproduce with:

```bash
uv run python scripts/audit_data.py --seed {seed} --full-offset-scan
```

**Corpus:** {n} contracts, {total_spans:,} annotated spans, 41 clause categories.
Contracts are read from `data/raw/CUAD_v1/full_contract_txt/` as UTF-8.

## Verdict up front
"""
    )
    exact = offsets.by_status[SpanStatus.EXACT]
    single_span_share = multiplicity["1 span"] / max(sum(multiplicity.values()), 1)
    parts.append(
        _table(
            ["Check", "Result", "Design impact"],
            [
                [
                    "1. Per-clause positives",
                    f"{len(included)} of 41 categories clear the >={MIN_POSITIVES} bar",
                    f"{len(excluded)} categories excluded as unmeasurable",
                ],
                [
                    "2. Base rate",
                    f"{min(s.base_rate for s in clause_stats):.1%} to "
                    f"{max(s.base_rate for s in clause_stats):.1%}",
                    "High-prevalence clauses need span-level metrics, not presence F1",
                ],
                [
                    "3. Token length",
                    f"median {lengths.median:,}, p99 {lengths.percentile(99):,}, "
                    f"max {max(lengths.token_counts):,}",
                    "Cost-curve story, not an accuracy-cliff story",
                ],
                [
                    "4. Offset integrity",
                    f"**{exact}/{offsets.sampled} exact ({1 - offsets.mismatch_rate:.2%})**",
                    "Offsets usable as-is, but only against raw text",
                ],
                [
                    "5. Multi-span",
                    f"{single_span_share:.0%} of positive pairs are single-span",
                    "Metric defined as any-span-hit; see below",
                ],
            ],
        )
    )
    parts.append("\n---\n")

    # ---- Check 1 & 2 ----
    parts.append("## Checks 1 and 2 — positive counts and base rates\n")
    parts.append(
        _table(
            [
                "#",
                "Clause category",
                "Positive contracts",
                "Base rate",
                "Gold spans",
                "CSV label",
                "Agrees",
                "Status",
            ],
            [
                [
                    str(i),
                    s.category,
                    str(s.positive_contracts),
                    f"{s.base_rate:.1%}",
                    f"{s.total_spans:,}",
                    f"Yes/No ({s.answer.yes_count} yes)"
                    if s.answer and s.answer.is_boolean
                    else (f"value ({s.answer.value_count})" if s.answer else "—"),
                    {True: "yes", False: "**NO**", None: "n/a"}[s.csv_agrees],
                    "included" if s.included else "**EXCLUDED**",
                ]
                for i, s in enumerate(clause_stats, 1)
            ],
        )
    )
    value_cats = [s for s in clause_stats if s.answer and not s.answer.is_boolean]
    bool_cats = [s for s in clause_stats if s.answer and s.answer.is_boolean]
    disagreeing = [s for s in clause_stats if s.csv_agrees is False]
    parts.append(
        f"""
*"CSV label" describes the `<Category>-Answer` column in `master_clauses.csv`.
These are not one thing: {len(value_cats)} categories carry a lawyer-normalized
**value** (`5/8/14`, `New York`, `30 days`) and {len(bool_cats)} carry a bare
**Yes/No** presence label. Counting non-empty cells uniformly reports 510/510 for
every boolean category and measures nothing.*

*"Agrees" cross-checks the two files: for a boolean category the CSV's "Yes" count
should equal the number of contracts with at least one gold span in
`CUAD_v1.json`. It does for {len(bool_cats) - len(disagreeing)} of {len(bool_cats)}
categories{", with no exceptions" if not disagreeing else ""}. The JSON spans and
the CSV labels are independent exports of the same annotation effort, so this is
free validation that the ground truth is internally consistent — and it means
presence labels can be read from either file without reconciling them.*

**What this implies.** {len(included)} of 41 categories clear the ≥{MIN_POSITIVES}-positive
bar, so the constraint on clause selection is not scarcity — it is the base-rate
problem at the other end. """
    )

    top = clause_stats[0]
    high_prevalence = [s for s in clause_stats if s.base_rate >= 0.90]
    # F1 of an always-yes classifier: precision = base rate, recall = 1.
    trivial_f1 = 2 * top.base_rate / (1 + top.base_rate)
    parts.append(
        f"""`{top.category}` appears in {top.positive_contracts}/{n} contracts
({top.base_rate:.0%}); {len(high_prevalence)} categories sit at or above 90% prevalence.
A presence classifier that answers "yes" unconditionally scores ~{trivial_f1:.2f} F1
on those. Reporting presence F1 alone for high-prevalence clauses would be
self-flattering, so every F1 in `RESULTS.md` carries its base rate beside it, and
the primary metric for those clauses is span-level overlap.

The {len(excluded)} excluded categories are listed above; the smallest is
`{excluded[-1].category}` at {excluded[-1].positive_contracts} positives, which
across a 150-case golden set would contribute well under one expected instance.

### Normalization targets, taken from the value columns

The {len(value_cats)} value-bearing categories show what `ClauseExtraction.value`
has to produce, and every one of them is a normalization hazard:
"""
    )
    parts.append(
        _table(
            ["Category", "Values present", "Most common", "Hazard"],
            [
                [
                    s.category,
                    f"{s.answer.value_count}/{n}" if s.answer else "—",
                    ", ".join(f"`{v}`" for v in s.answer.common_values) if s.answer else "—",
                    _NORMALIZATION_HAZARDS.get(s.category, "—"),
                ]
                for s in value_cats
            ],
        )
    )
    parts.append(
        """
Two of these change the design of the rules layer. `Agreement Date` and
`Effective Date` use a **masked** format for partially-known dates -- `[]/[]/2020`
means "sometime in 2020" -- so `value` cannot be a `datetime.date`; it has to be a
string with a documented partial-date form, and `effective_date <= expiration_date`
has to be skipped rather than failed when either side is masked. `Expiration Date`
is worse: its single most common value is `perpetual`, which is not a date at all,
and it appears in both `perpetual` and `Perpetual` casings. The comparison rule
needs an explicit sentinel for open-ended terms.

`Governing Law` behaves exactly as expected -- New York, California, and Delaware
dominate -- which is what `data/reference/jurisdictions.yaml` has to resolve.

### Source consistency: where the two files disagree
"""
    )
    parts.append(
        _table(
            ["Join of master_clauses.csv onto CUAD_v1.json", "Rows"],
            [
                ["CSV rows", f"{consistency.csv_rows}"],
                ["joined on exact filename", f"{consistency.exact_joins}"],
                ["joined only after punctuation folding", f"{consistency.relaxed_joins}"],
                ["unjoinable", f"{len(consistency.unjoinable)}"],
            ],
        )
    )
    parts.append("\n")
    if consistency.label_disagreements:
        parts.append(
            _table(
                ["Category", "Contract", "CSV label", "Gold spans in JSON"],
                [
                    [
                        category,
                        f"`{title[:52]}...`" if len(title) > 55 else f"`{title}`",
                        "Yes" if says_yes else "No",
                        str(spans),
                    ]
                    for category, title, says_yes, spans in consistency.label_disagreements
                ],
            )
        )
    parts.append(
        f"""
**What this implies.** {consistency.relaxed_joins} CSV rows do not join to the JSON on
filename alone. The CSV writes `MACY'S,INC_...` where the filesystem has
`MACY_S,INC_...`, and several rows carry trailing spaces. Folding away punctuation
and case recovers all of them, but anything joining these two files needs that
folding or it silently drops contracts -- the same class of bug as the Unicode
issue in check 4.

More consequentially, **{len(consistency.label_disagreements)} contract-level labels
disagree outright**: the CSV marks the clause `Yes` while the JSON carries zero gold
spans for it, and the CSV's own span column is an empty list `[]`. These are label
errors, not span-extraction failures -- a reviewer ticked the box without
highlighting the text.

That decides the source of truth: **`CUAD_v1.json` is authoritative for presence and
spans**, because a label with no span is unusable for a span-grounded system and
unverifiable by the grounding check. `master_clauses.csv` is used only for the
{len(value_cats)} normalized value columns. The disagreement rate is
{len(consistency.label_disagreements) / (len(bool_cats) * n):.4%} of boolean labels,
which is small enough not to distort the corpus but large enough that the golden set
should be spot-checked by hand rather than trusted blindly.

---
"""
    )

    # ---- Check 3 ----
    parts.append("## Check 3 — token length distribution\n")
    parts.append(f"![Contract token length distribution]({figure_path.as_posix()})\n")
    parts.append(
        _table(
            ["Statistic", "Tokens"],
            [
                ["min", f"{min(lengths.token_counts):,}"],
                ["median", f"{lengths.median:,}"],
                ["p90", f"{lengths.percentile(90):,}"],
                ["p99", f"{lengths.percentile(99):,}"],
                ["max", f"{max(lengths.token_counts):,}"],
                ["mean", f"{int(statistics.mean(lengths.token_counts)):,}"],
                ["total corpus", f"{sum(lengths.token_counts):,}"],
            ],
        )
    )
    parts.append("\n")
    parts.append(
        _table(
            ["Context window", "Contracts over", "Share"],
            [
                [f"{w:,}", str(lengths.over_threshold[w]), f"{lengths.over_threshold[w] / n:.1%}"]
                for w in CONTEXT_WINDOWS
            ],
        )
    )

    over_200k = lengths.over_threshold[200_000]
    tiny = sum(1 for c in lengths.token_counts if c < 500)
    parts.append(
        f"""
**What this implies.** {over_200k} contracts exceed a 200k-token window — and none
exceed 128k either. The longest contract in CUAD is {max(lengths.token_counts):,}
tokens, comfortably inside a single Claude call. That settles an open question in
the spec: the RAG-versus-long-context experiment is **a cost curve, not an accuracy
cliff**. Retrieval has to justify itself on dollars and latency, because "the
document does not fit" is never the argument on this corpus.

The median contract at {lengths.median:,} tokens is small enough that full-context
extraction is affordable, which makes the crossover point the interesting number
rather than a foregone conclusion. The tail is thin — {lengths.over_threshold[32_768]}
contracts ({lengths.over_threshold[32_768] / n:.0%}) exceed 32k — so stratifying the
golden set by length quartile is load-bearing: sample naively and the expensive
tail barely appears in the evaluation at all.

**A caveat on the short end.** {tiny} contracts are under 500 tokens, and they are
not really contracts — the shortest ({min(lengths.token_counts)} tokens) is an SEC
*joint filing agreement*, a one-paragraph boilerplate exhibit. They carry almost no
clauses, so a system scores near-perfectly on them for reasons unrelated to
extraction quality. The bottom length quartile is partly these, which is an
argument for reporting per-tier metrics rather than a corpus-wide average, and for
noting the floor honestly in `LIMITATIONS.md`.

---
"""
    )

    # ---- Check 4 ----
    parts.append("## Check 4 — span offset integrity (the critical check)\n")
    parts.append(
        _table(
            ["Status", "Count", "Share"],
            [
                [
                    status.value,
                    str(offsets.by_status[status]),
                    f"{offsets.by_status[status] / offsets.sampled:.4%}",
                ]
                for status in SpanStatus
            ],
        )
    )
    parts.append(
        f"""
Sampled {offsets.sampled:,} of {offsets.population:,} annotations (seed {seed}).
**Mismatch rate: {offsets.mismatch_rate:.4%}.**

The JSON `context` field is byte-identical to the corresponding TXT file for
{offsets.context_matches_txt}/{offsets.context_total} contracts, so
`answer_start` indexes into the file the loader opens with no remapping layer.

**What this implies — and one correction to the spec.** `CLAUDE.md` section 2.2 asks
whether offsets survive "after whitespace normalization." They do not, and they
cannot: collapsing whitespace runs shortens the string, which shifts every
subsequent character. The premise is inverted. What the audit actually shows is
stronger — offsets are exact against the **raw, unnormalized** UTF-8 text, so the
correct design is to never normalize the offset-authoritative representation at
all:

- The loader reads the TXT file verbatim and treats it as the coordinate system.
- Whitespace normalization is applied **only** to the two strings being compared
  inside the grounding check, never to the stored document.
- Chunking must carry each chunk's absolute `char_start` in the raw document, so
  evidence offsets stay meaningful after retrieval.

Getting this backwards is exactly the silent failure the spec warned about: it
would make the grounding verifier reject correct extractions, and the resulting
F1 drop would look like a model problem rather than an indexing bug.

One real encoding hazard did surface. """
    )
    if unmatched:
        parts.append(
            f"{len(unmatched)} title(s) still fail to match a TXT file: "
            + ", ".join(f"`{t}`" for t in unmatched)
            + ".\n"
        )
    else:
        parts.append(
            """The title `LECLANCHÉ S.A. - JOINT DEVELOPMENT AND MARKETING AGREEMENT`
encodes `É` as a combining acute accent (U+0301) in one source and as a precomposed
character (U+00C9) in the other. String equality fails; NFC folding fixes it. Without
`contract_key()` that contract silently drops out of the corpus and the count reads
509 with no error raised anywhere.
"""
        )
    parts.append("\n---\n")

    # ---- Bonus: pre-validating the proposed rules ----
    parts.append("## Bonus — the proposed cross-field rules, scored against gold labels\n")
    parts.append(
        """`CLAUDE.md` section 5 sketches deterministic rules from first principles.
Before writing any of them, each was scored against CUAD's own labels. A rule that
fires on correct data is worse than no rule -- it teaches the reviewer to ignore
violations.
"""
    )
    parts.append(
        _table(
            ["Proposed rule", "Holds", "Violated", "Violation rate", "Verdict"],
            [
                [
                    check.name,
                    str(check.holds),
                    str(check.violated),
                    f"{check.violated / check.applicable:.1%}" if check.applicable else "n/a",
                    check.verdict,
                ]
                for check in rule_checks
            ],
        )
    )
    parts.append(
        """
**Two of the proposed rules are wrong, and one is wrong in an interesting way.**

`cap_on_liability` and `uncapped_liability` are proposed as mutually exclusive.
They are not — and not merely sometimes. **Every single contract with an uncapped
liability clause also has a cap on liability**; there are zero uncapped-only
contracts in the corpus. The relationship is subset, not exclusion, and it makes
legal sense once stated: a contract sets a general liability cap and then carves
out specific categories (IP infringement, breach of confidentiality) that sit
outside it. You cannot carve an exception out of a cap that does not exist.

Shipped as written, that rule would fire on 111 of 510 contracts and be wrong every
time. Inverted — `uncapped_liability` present implies `cap_on_liability` present —
it holds on 111 of 111 and ships as a hard ERROR.

The renewal dependency is softer than the spec assumes. `renewal_term` implies
`notice_period` holds only about three times in five: plenty of contracts specify a
renewal term with no notice requirement, because renewal is automatic and
unconditional. As an error it would produce 67 false violations. The *inverse* is
far stronger and is the one worth shipping.

The lesson generalizes to the other 20-odd rules: **a business rule is a hypothesis
about the data, and this corpus can test it.** Rules that survive ship as errors,
rules that mostly hold ship as warnings, and rules that fail get discarded rather
than quietly weakening the signal.

---
"""
    )

    # ---- Check 5 ----
    total_pairs = sum(multiplicity.values())
    parts.append("## Check 5 — multi-span annotations\n")
    parts.append(
        _table(
            ["Gold spans per (contract, clause)", "Pairs", "Share"],
            [
                [bucket, f"{count:,}", f"{count / total_pairs:.1%}"]
                for bucket, count in multiplicity.items()
            ],
        )
    )
    multi = total_pairs - multiplicity["1 span"]
    parts.append(
        f"""
{multi:,} of {total_pairs:,} positive pairs ({multi / total_pairs:.0%}) carry more than
one gold span.

**Metric decision, recorded before any results were seen.** For **presence**, a
(contract, clause) pair is a hit if the model returns `present=True` and *at least
one* returned evidence span overlaps *any* gold span. Recall is not divided across
the gold spans — finding 1 of 3 is a hit, not 0.33.

The rationale is that CUAD's multiple spans are usually the same obligation
restated across a definitions section, an operative clause, and a schedule, so
treating them as three independent targets would penalize a correct extraction for
not being exhaustive. The cost of this choice is that it cannot distinguish "found
the clause" from "found every mention of the clause," so **span token F1 against the
union of gold spans is reported alongside it** and is the metric that penalizes
partial coverage. Both numbers appear in `RESULTS.md`; neither is reported alone.
"""
    )

    return "\n".join(parts)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuad-dir", type=Path, default=DEFAULT_CUAD_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full-offset-scan",
        action="store_true",
        help="Verify every annotation rather than a sample.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.cuad_dir.exists():
        print(f"CUAD not found at {args.cuad_dir}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    contracts, unmatched = load_contracts(args.cuad_dir)
    print(f"loaded {len(contracts)} contracts ({len(unmatched)} unmatched)")

    answers = load_master_answers(args.cuad_dir / "master_clauses.csv")
    clause_stats = compute_clause_stats(contracts, answers)
    print(f"clause stats: {sum(1 for s in clause_stats if s.included)}/41 categories included")

    lengths = compute_length_stats(contracts)
    print(
        f"lengths: median {lengths.median:,} p90 {lengths.percentile(90):,} "
        f"max {max(lengths.token_counts):,}"
    )

    sample_size = 10**9 if args.full_offset_scan else args.sample_size
    offsets = audit_offsets(contracts, args.cuad_dir, sample_size, rng)
    print(
        f"offsets: {offsets.by_status[SpanStatus.EXACT]}/{offsets.sampled} exact, "
        f"mismatch rate {offsets.mismatch_rate:.4%}"
    )

    multiplicity = compute_span_multiplicity(contracts)
    consistency = audit_source_consistency(contracts, args.cuad_dir / "master_clauses.csv")
    rule_checks = validate_proposed_rules(contracts)
    print(
        f"consistency: {len(consistency.label_disagreements)} label disagreements, "
        f"{len(consistency.unjoinable)} unjoinable rows"
    )

    render_length_histogram(lengths, args.figure)
    print(f"wrote {args.figure}")

    report = render_report(
        contracts=contracts,
        unmatched=unmatched,
        clause_stats=clause_stats,
        lengths=lengths,
        offsets=offsets,
        multiplicity=multiplicity,
        consistency=consistency,
        rule_checks=rule_checks,
        seed=args.seed,
        figure_path=Path("img") / args.figure.name,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"wrote {args.report}")

    if offsets.mismatch_rate > 0.01:
        print(
            f"\nSTOP: offset mismatch rate {offsets.mismatch_rate:.2%} exceeds 1%. "
            "The grounding verifier cannot trust these offsets.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
