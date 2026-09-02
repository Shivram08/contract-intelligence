"""Typed contracts shared across the pipeline.

The invariant that runs through all of this: **a chunk's text is a verbatim
slice of its document.** ``document.text[chunk.char_start:chunk.char_end] ==
chunk.text``, always. Every offset in the system is an offset into the raw,
unmodified document, so an evidence quote can be traced from a model response
back through a chunk to an exact character range in the source file. Break that
and the grounding verifier silently starts rejecting correct extractions -- see
``docs/DATA_AUDIT.md`` check 4.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "BOOLEAN_CLAUSES",
    "CLAUSE_DEFINITIONS",
    "CLAUSE_TIERS",
    "CUAD_CATEGORIES",
    "PERPETUAL",
    "UNSPECIFIED",
    "Chunk",
    "ClauseExtraction",
    "ClauseType",
    "Document",
    "Evidence",
    "ExtractionResult",
    "RetrievalHit",
    "ReviewItem",
    "RuleViolation",
    "ScoredChunk",
    "Severity",
    "Tier",
    "TokenUsage",
]


# --------------------------------------------------------------------------
# Documents, chunks, retrieval
# --------------------------------------------------------------------------


class Document(BaseModel):
    """A contract in its offset-authoritative form.

    ``text`` is the file exactly as read: no whitespace collapsing, no newline
    translation, no Unicode re-normalization. It is the coordinate system that
    every offset in the pipeline refers to.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    text: str
    #: Where the text came from, for provenance in errors and traces.
    source_path: str | None = None
    #: Free-form; CUAD carries agreement type, split assignment, and so on.
    metadata: dict[str, str] = Field(default_factory=dict)

    def slice(self, char_start: int, char_end: int) -> str:
        """Read a character range out of the raw text."""
        return self.text[char_start:char_end]

    def __len__(self) -> int:
        return len(self.text)


class Chunk(BaseModel):
    """A retrievable passage, anchored to an exact range in its document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    #: Position in the document, 0-based. Retrieval uses this to fetch neighbours.
    ordinal: int = Field(ge=0)
    #: The section heading this chunk sits under, when one was detected.
    #: None for preamble text and for documents with no detectable structure --
    #: 57 of CUAD's 510 contracts have no numbered headings at all.
    heading: str | None = None
    token_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_range(self) -> Chunk:
        if self.char_end < self.char_start:
            raise ValueError(
                f"chunk {self.chunk_id}: char_end {self.char_end} precedes "
                f"char_start {self.char_start}"
            )
        span = self.char_end - self.char_start
        if span != len(self.text):
            # Catches the classic bug where a chunker strips or normalizes text
            # but keeps the original offsets. Cheap here, invisible later.
            raise ValueError(
                f"chunk {self.chunk_id}: offset span {span} does not match "
                f"text length {len(self.text)}"
            )
        return self

    def is_faithful_to(self, document: Document) -> bool:
        """Whether this chunk is a verbatim slice of the given document."""
        return document.slice(self.char_start, self.char_end) == self.text


class ScoredChunk(BaseModel):
    """A chunk with the score from one retriever."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    #: Which retriever produced this: "lexical", "dense", "rrf", "rerank".
    retriever: str


class RetrievalHit(BaseModel):
    """A fused result, carrying enough provenance to explain the ranking.

    The per-retriever ranks are kept because the interesting question about
    hybrid retrieval is not "what came back" but "which retriever found it" --
    that is the ablation story, and reconstructing it after the fact is
    impossible once the lists are merged.
    """

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    #: retriever name -> 1-based rank in that retriever's list. A retriever that
    #: did not return this chunk is absent rather than present with a sentinel.
    ranks: dict[str, int] = Field(default_factory=dict)
    rerank_score: float | None = None


# --------------------------------------------------------------------------
# The clause taxonomy
# --------------------------------------------------------------------------


class Tier(IntEnum):
    """Difficulty tier from ``data/reference/clause_schema.yaml``.

    Metrics are reported per tier because aggregates hide the story: Tier 1 is
    near-deterministic, Tier 3 requires judgement and is where the system is
    expected to fail.
    """

    NEAR_DETERMINISTIC = 1
    REQUIRES_READING = 2
    REQUIRES_JUDGEMENT = 3


class ClauseType(StrEnum):
    """The 12 clause types for v1, locked by the audit counts.

    Definitions live in ``CLAUSE_DEFINITIONS`` below, quoted **verbatim** from
        CUAD's own question text (the string after "Details:" in ``CUAD_v1.json``).
        Per CLAUDE.md section 2.1 these are the definitions the annotators worked
        from; paraphrasing them means extracting against a different standard than
        the ground truth was labelled against, which costs F1 for reasons unrelated
        to the model.

        **The strings below the members are for readers of this file only.** Python
        discards them -- an enum member does not get its own ``__doc__`` and inherits
        the class docstring instead. ``CLAUSE_DEFINITIONS`` below is what the code
        reads. ``tests/unit/test_clause_types.py`` pins both against
        ``data/reference/clause_schema.yaml``.
    """

    GOVERNING_LAW = "governing_law"
    """Which state/country's law governs the interpretation of the contract?"""

    PARTIES = "parties"
    """The two or more parties who signed the contract"""

    EFFECTIVE_DATE = "effective_date"
    """The date when the contract is effective"""

    EXPIRATION_DATE = "expiration_date"
    """On what date will the contract's initial term expire?"""

    RENEWAL_TERM = "renewal_term"
    """What is the renewal term after the initial term expires? This includes
    automatic extensions and unilateral extensions with prior notice."""

    NOTICE_PERIOD_TO_TERMINATE_RENEWAL = "notice_period_to_terminate_renewal"
    """What is the notice period required to terminate renewal?"""

    CHANGE_OF_CONTROL = "change_of_control"
    """Does one party have the right to terminate or is consent or notice
    required of the counterparty if such party undergoes a change of control,
    such as a merger, stock sale, transfer of all or substantially all of its
    assets or business, or assignment by operation of law?"""

    ANTI_ASSIGNMENT = "anti_assignment"
    """Is consent or notice required of a party if the contract is assigned to a
    third party?"""

    CAP_ON_LIABILITY = "cap_on_liability"
    """Does the contract include a cap on liability upon the breach of a party's
    obligation? This includes time limitation for the counterparty to bring
    claims or maximum amount for recovery."""

    UNCAPPED_LIABILITY = "uncapped_liability"
    """Is a party's liability uncapped upon the breach of its obligation in the
    contract? This also includes uncap liability for a particular type of breach
    such as IP infringement or breach of confidentiality obligation."""

    NON_COMPETE = "non_compete"
    """Is there a restriction on the ability of a party to compete with the
    counterparty or operate in a certain geography or business or technology
    sector?"""

    EXCLUSIVITY = "exclusivity"
    """Is there an exclusive dealing commitment with the counterparty? This
    includes a commitment to procure all requirements from one party of certain
    technology, goods, or services or a prohibition on licensing or selling
    technology, goods or services to third parties, or a prohibition on
    collaborating or working with other parties, whether during the contract or
    after the contract ends (or both)."""


#: clause type -> CUAD's own definition, quoted verbatim.
#:
#: This is a dict rather than per-member docstrings because **enum members do
#: not get their own ``__doc__``** -- the string after a member assignment is
#: discarded and every member inherits the class docstring. An earlier version
#: relied on that and shipped a `get_schema` tool that handed the agent the
#: class docstring ("The 12 clause types for v1...") as the definition of all
#: twelve clauses. Nothing raised; the definitions were simply absent from the
#: prompt. The docstrings below the members are kept for readers of the source.
#:
#: Checked against data/reference/clause_schema.yaml by tests/unit/test_clause_types.py.
CLAUSE_DEFINITIONS: Final[dict[ClauseType, str]] = {
    ClauseType.GOVERNING_LAW: (
        "Which state/country's law governs the interpretation of the contract?"
    ),
    ClauseType.PARTIES: ("The two or more parties who signed the contract"),
    ClauseType.EFFECTIVE_DATE: ("The date when the contract is effective"),
    ClauseType.EXPIRATION_DATE: ("On what date will the contract's initial term expire?"),
    ClauseType.RENEWAL_TERM: (
        "What is the renewal term after the initial term expires? This includes automatic "
        "extensions and unilateral extensions with prior notice."
    ),
    ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL: (
        "What is the notice period required to terminate renewal?"
    ),
    ClauseType.CHANGE_OF_CONTROL: (
        "Does one party have the right to terminate or is consent or notice required of "
        "the counterparty if such party undergoes a change of control, such as a merger, "
        "stock sale, transfer of all or substantially all of its assets or business, or "
        "assignment by operation of law?"
    ),
    ClauseType.ANTI_ASSIGNMENT: (
        "Is consent or notice required of a party if the contract is assigned to a third party?"
    ),
    ClauseType.CAP_ON_LIABILITY: (
        "Does the contract include a cap on liability upon the breach of a party's "
        "obligation? This includes time limitation for the counterparty to bring claims "
        "or maximum amount for recovery."
    ),
    ClauseType.UNCAPPED_LIABILITY: (
        "Is a party's liability uncapped upon the breach of its obligation in the "
        "contract? This also includes uncap liability for a particular type of breach "
        "such as IP infringement or breach of confidentiality obligation."
    ),
    ClauseType.NON_COMPETE: (
        "Is there a restriction on the ability of a party to compete with the "
        "counterparty or operate in a certain geography or business or technology sector?"
    ),
    # CUAD's own wording quotes the term "requirements", so this one is a
    # single-quoted literal. The stray ")" after "other parties" is also CUAD's;
    # it is preserved because the definition is quoted verbatim.
    ClauseType.EXCLUSIVITY: (
        "Is there an exclusive dealing  commitment with the counterparty? This includes "
        'a commitment to procure all "requirements" from one party of certain technology, '
        "goods, or services or a prohibition on licensing or selling technology, goods or "
        "services to third parties, or a prohibition on  collaborating or working with "
        "other parties), whether during the contract or  after the contract ends (or both)."
    ),
}

#: clause type -> difficulty tier. Kept beside the enum rather than on it,
#: because StrEnum members cannot carry extra attributes without a metaclass.
CLAUSE_TIERS: Final[dict[ClauseType, Tier]] = {
    ClauseType.GOVERNING_LAW: Tier.NEAR_DETERMINISTIC,
    ClauseType.PARTIES: Tier.NEAR_DETERMINISTIC,
    ClauseType.EFFECTIVE_DATE: Tier.NEAR_DETERMINISTIC,
    ClauseType.EXPIRATION_DATE: Tier.NEAR_DETERMINISTIC,
    ClauseType.RENEWAL_TERM: Tier.REQUIRES_READING,
    ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL: Tier.REQUIRES_READING,
    ClauseType.CHANGE_OF_CONTROL: Tier.REQUIRES_READING,
    ClauseType.ANTI_ASSIGNMENT: Tier.REQUIRES_READING,
    ClauseType.CAP_ON_LIABILITY: Tier.REQUIRES_JUDGEMENT,
    ClauseType.UNCAPPED_LIABILITY: Tier.REQUIRES_JUDGEMENT,
    ClauseType.NON_COMPETE: Tier.REQUIRES_JUDGEMENT,
    ClauseType.EXCLUSIVITY: Tier.REQUIRES_JUDGEMENT,
}

#: CUAD's category name for each clause type, for scoring against gold spans.
CUAD_CATEGORIES: Final[dict[ClauseType, str]] = {
    ClauseType.GOVERNING_LAW: "Governing Law",
    ClauseType.PARTIES: "Parties",
    ClauseType.EFFECTIVE_DATE: "Effective Date",
    ClauseType.EXPIRATION_DATE: "Expiration Date",
    ClauseType.RENEWAL_TERM: "Renewal Term",
    ClauseType.NOTICE_PERIOD_TO_TERMINATE_RENEWAL: "Notice Period To Terminate Renewal",
    ClauseType.CHANGE_OF_CONTROL: "Change Of Control",
    ClauseType.ANTI_ASSIGNMENT: "Anti-Assignment",
    ClauseType.CAP_ON_LIABILITY: "Cap On Liability",
    ClauseType.UNCAPPED_LIABILITY: "Uncapped Liability",
    ClauseType.NON_COMPETE: "Non-Compete",
    ClauseType.EXCLUSIVITY: "Exclusivity",
}

#: Clause types whose answer is presence plus evidence, with no normalized value.
#: Asking for a value here invites paraphrase, and a paraphrase cannot be checked
#: by the grounding verifier. See data/reference/normalization.md section 5.
BOOLEAN_CLAUSES: Final[frozenset[ClauseType]] = frozenset(
    {
        ClauseType.CHANGE_OF_CONTROL,
        ClauseType.ANTI_ASSIGNMENT,
        ClauseType.CAP_ON_LIABILITY,
        ClauseType.UNCAPPED_LIABILITY,
        ClauseType.NON_COMPETE,
        ClauseType.EXCLUSIVITY,
    }
)

#: The open-ended-term sentinel for ``expiration_date``. ``perpetual`` is the
#: single most common gold value for that clause (66 of 329) and is not a date --
#: see data/reference/normalization.md section 1.
PERPETUAL: Final = "PERPETUAL"

#: Clause present but naming no resolvable value.
UNSPECIFIED: Final = "UNSPECIFIED"


# --------------------------------------------------------------------------
# Extraction output
# --------------------------------------------------------------------------


class Severity(StrEnum):
    """How much a rule violation matters.

    Three levels rather than two because the audit measured a middle ground: the
    ``renewal_term => notice_period`` dependency holds only 62% of the time on
    gold data, so firing it as an ERROR would be wrong 67 times out of 176. Rules
    like that ship as WARNING. See docs/DATA_AUDIT.md.
    """

    ERROR = "error"
    """Internally inconsistent or unusable output. Always routes to review."""

    WARNING = "warning"
    """Suspicious but legitimately possible. Surfaced; does not force review."""

    INFO = "info"
    """Recorded for analysis; never routes to review."""


class Evidence(BaseModel):
    """A verbatim quote from the source, with its exact location.

    ``char_start``/``char_end`` index into the **raw** document text -- the same
    coordinate system the chunker and CUAD's own offsets use.
    """

    model_config = ConfigDict(frozen=True)

    quote: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    #: Which chunk this came from, for tracing a citation back through retrieval.
    chunk_id: str | None = None

    @model_validator(mode="after")
    def _check_range(self) -> Evidence:
        if self.char_end <= self.char_start:
            raise ValueError(
                f"evidence char_end {self.char_end} must exceed char_start {self.char_start}"
            )
        return self


class ClauseExtraction(BaseModel):
    """One clause type's verdict for one document."""

    model_config = ConfigDict(frozen=True)

    clause_type: ClauseType
    present: bool
    #: Normalized per data/reference/normalization.md. None for the six boolean
    #: clause types, where presence plus evidence is the whole answer.
    value: str | None = None
    #: The clause text as it appears, verbatim. Never a paraphrase.
    raw_text: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def tier(self) -> Tier:
        return CLAUSE_TIERS[self.clause_type]


class RuleViolation(BaseModel):
    """A deterministic check that failed."""

    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(min_length=1)
    severity: Severity
    message: str = Field(min_length=1)
    #: Clause types the rule looked at. Empty for document-level rules.
    clause_types: list[ClauseType] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Token counts and dollar cost for one extraction.

    Cost comes from ``response.usage``, which is authoritative, rather than being
    estimated from a local tokenizer. Nobody reports cost per document; this
    project does.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Accumulate usage across turns of the agent loop."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class ExtractionResult(BaseModel):
    """Everything one document's extraction produced, including how it went."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    clauses: list[ClauseExtraction] = Field(default_factory=list)
    violations: list[RuleViolation] = Field(default_factory=list)
    needs_review: bool = False
    #: Hash of the prompt text, so a regression can be attributed to a prompt
    #: change rather than a code change.
    prompt_version: str = ""
    model: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)
    #: Per-stage wall-clock in milliseconds: retrieval, agent, validation.
    latency_ms: dict[str, float] = Field(default_factory=dict)
    #: Turns the agent loop actually used, for budget analysis.
    turns_used: int = 0
    #: True when the loop hit a turn, token, or time limit rather than finishing.
    stopped_on_budget: bool = False

    def clause(self, clause_type: ClauseType) -> ClauseExtraction | None:
        return next((c for c in self.clauses if c.clause_type is clause_type), None)

    @property
    def errors(self) -> list[RuleViolation]:
        return [v for v in self.violations if v.severity is Severity.ERROR]


class ReviewItem(BaseModel):
    """A result routed to a human, with the reason it was routed."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    reasons: list[str] = Field(min_length=1)
    result: ExtractionResult
