# Contract Intelligence Service — Project Spec

**Owner:** Shivram Nekkanti
**Purpose:** Public, shippable analog of production GenAI document-understanding work. Targets Capital One (AI Foundations), JPMorgan Chase (Applied AI/ML, AI for Operations), and American Express (AI/ML Engineer).
**Estimated build time:** 2–3 focused days with Claude Code.

---

## 1. What this project proves

Three things, in priority order:

1. **Software engineering practice** — CI/CD, typed interfaces, tests, containers, observability. This is the single largest gap in the current portfolio and is listed explicitly in nearly every AI/ML posting.
2. **Agentic LLM systems with enforced structure** — tool-calling loops, Pydantic-validated outputs, deterministic business rules, mechanical hallucination gates.
3. **Evaluation rigor** — labeled ground truth, a baseline ladder, regression gates in CI, cost and latency budgets.

Explicit non-goal: beating state of the art on a benchmark. The deliverable is a system with defensible numbers, not a leaderboard entry.

---

## 2. Domain and data

**Primary dataset: CUAD** (Contract Understanding Atticus Dataset) — 510 commercial legal contracts spanning 25 agreement types, with 13,101 expert-labeled clauses across 41 categories, annotated by lawyers and law students over roughly a year. Source contracts come from SEC EDGAR.

**License: CC BY 4.0.** Free for commercial and non-commercial use, attribution required. Two obligations for this repo:

- Include the Hendrycks et al. (2021) NeurIPS citation and a CC BY 4.0 attribution line in `README.md`.
- Note in `LIMITATIONS.md` that the Atticus Project makes no representations about the license status of the underlying contracts themselves, which are public EDGAR filings.

Why CUAD over SEC EDGAR raw: **it has ground truth.** Expert span-level annotations mean the evaluation is measurable rather than vibes-based, which is the whole point of the project. Commercial contract review is also directly analogous to what banks do with loan agreements, vendor contracts, and ISDAs.

**Optional extension: SEC EDGAR 10-K risk factors.** Unlabeled, so it demonstrates LLM-as-judge where ground truth doesn't exist — a nice contrast to the labeled CUAD path. Add only after the core system is done.

### 2.1 What to download

| Artifact | Source | Purpose |
|---|---|---|
| CUAD v1 full package | atticusprojectai.org/cuad | Raw contracts (PDF + **TXT**), `master_clauses.csv`, `CUAD_v1.json` with character offsets |
| `category_descriptions.csv` | github.com/The-Atticus-Project/cuad | Official lawyer-written definitions of all 41 categories |
| `data.zip` | same repo | Pre-split train/test in SQuAD format, if not doing a custom split |
| `theatticusproject/cuad-qa` | HuggingFace | Convenience loader for the QA-format version |
| `BAAI/bge-base-en-v1.5`, `BAAI/bge-reranker-v2-m3` | HuggingFace | Embeddings + reranker, ~1.5 GB, cache to a mounted volume |

**Use the plain-text contracts, not the PDFs.** PDF parsing adds a day and proves nothing this project is trying to prove.

**`category_descriptions.csv` is the piece people miss.** Use those definitions verbatim as clause descriptions in prompts and as docstrings on the `ClauseType` enum. Writing your own definitions means extracting against a different standard than the annotators used, which produces bad F1 for reasons that have nothing to do with the model.

### 2.2 Pre-flight data audit — run before any application code

One throwaway profiling script. Three of these five findings will change the design, so do not skip it.

1. **Per-clause positive counts.** Several of the 41 categories are rare. A clause with 15 positives across 510 contracts yields noise, not a measurement. Require **≥40 positive examples corpus-wide** to include a clause type.
2. **Base rate per clause.** The inverse problem. A clause present in 495/510 contracts has trivially high presence F1. Report base rate beside every F1, and lean on span-level metrics for high-prevalence clauses.
3. **Token length distribution.** Median, p90, max, and counts above each common context-window threshold. Determines whether the RAG-versus-long-context experiment is an accuracy-cliff story or a cost-curve story.
4. **Span offset integrity.** Verify `answer_start` offsets index correctly into the TXT files after whitespace normalization. **This is the most likely silent failure in the project** — drift here makes the grounding check flag correct extractions as hallucinations.
5. **Multi-span annotations.** Many clauses have several gold spans per contract. Decide the metric now: is 1 of 3 gold spans a hit, partial credit, or a miss? Write it down before seeing results, so the definition isn't chosen to flatter the numbers.

**Clause scope for v1:** 12 clause types, not all 41, spanning three difficulty tiers. The list below is a provisional guess — **the audit counts overrule it.**

- *Tier 1, near-deterministic:* Governing Law, Effective Date, Expiration Date, Parties
- *Tier 2, requires reading:* Auto-Renewal, Notice Period to Terminate, Change of Control, Assignment
- *Tier 3, requires judgment:* Cap on Liability, Uncapped Liability, Most Favored Nation, Non-Compete

The tiering gives the results table structure and gives you something real to say about where the system fails and why.

### 2.3 Split protocol — freeze before writing a single prompt

Iterating prompts against the evaluation set is test-set leakage. This is the most common flaw in LLM portfolio projects and the easiest to avoid.

- **Dev set — ~60 contracts.** Debug here, iterate prompts here, look as much as you want.
- **Held-out golden set — ~150 cases from a disjoint ~100 contracts.** Run at milestone boundaries only. These are the reported numbers.
- **Reserve — remainder.** For the nightly full run.

Seed the split, commit contract IDs to `evals/golden/split.json`, and describe the procedure in the README. One sentence — "prompts were developed on a disjoint dev split and never on the reported evaluation set" — signals more than most of the code will.

### 2.4 Reference data to hand-author

- **`data/jurisdictions.yaml`** — ~80 entries: 50 states, DC, England and Wales, and other common governing-law choices. Delaware and New York will dominate. Feeds the governing-law validation rule.
- **`data/normalization.md`** — spec for canonical forms: dates to ISO 8601, party names to a canonical string, notice periods to integer days. "30 days," "thirty (30) days," and "one month" must land in the same place or the cross-field rules won't work.

### 2.5 What is NOT needed

No training data (no fine-tuning in v1). No EDGAR scraping. No OCR. No labeling — except ~60 hand-labeled cases for judge calibration, and that is M6.

### 2.6 Cost planning

Storage and compute are trivial; the corpus is small and pgvector handles the chunk count easily. Local embedding is 15–30 min on a laptop GPU, a few hours on CPU — do it once and cache.

API spend is the real constraint. A 150-case run at ~10k tokens of context is roughly 1.5M input tokens per baseline; the full-context baseline is several times that. Multiply by five baselines and multiple iterations. Compute the actual figure against current pricing, set a hard budget cap in the nightly workflow, and get the response cache working during M3 rather than deferring it.

---

## 3. Architecture

```
Client
  │
  ▼
FastAPI  ──► /extract  (sync, single doc)
             /extract/batch (async, job id)
             /healthz, /metrics
  │
  ▼
Extraction Orchestrator
  │
  ├─► Retrieval Layer ──► BM25 (Postgres FTS) ─┐
  │                       Dense (pgvector)  ───┤─► RRF fusion ──► cross-encoder rerank
  │
  ├─► Agent Loop (tool-calling, max N turns, token + cost budget)
  │       tools: search_contract, read_span, get_schema, validate_extraction
  │
  ├─► Structured Output (Pydantic v2, strict mode, retry-on-parse-failure)
  │
  ├─► Validation Layer
  │       grounding check   → quoted evidence must appear verbatim in source
  │       field rules       → types, enums, date parsing
  │       cross-field rules → internal consistency
  │       confidence gate   → route low-confidence to review queue
  │
  └─► Telemetry (OTel spans → Jaeger, Prometheus metrics → Grafana)
```

### Stack — decided, do not re-litigate

| Layer | Choice | Rationale |
|---|---|---|
| Runtime | Python 3.11+, `uv` | Fast installs, lockfile, modern default |
| API | FastAPI + Pydantic v2 | Typed contracts, auto OpenAPI |
| LLM | Anthropic API (Claude Sonnet) | Native tool-calling, cheap, fast |
| Embeddings | `BAAI/bge-base-en-v1.5` | Free, local, strong for the size |
| Vector store | **pgvector** on Postgres | One container, what banks actually run |
| Lexical | Postgres full-text search | Same database, no extra service |
| Reranker | `BAAI/bge-reranker-v2-m3` | Local cross-encoder, no API cost |
| Agent | **Hand-rolled tool loop** | See note below |
| Tests | pytest + pytest-asyncio | Standard |
| Lint/types | ruff + mypy | Standard |
| Observability | OpenTelemetry + Prometheus | Mirrors production experience |
| CI | GitHub Actions | Visible on the repo page |
| Containers | Multi-stage Dockerfile + compose | Deployability signal |

**On the agent framework:** write the tool-calling loop by hand. A LangChain quickstart in a portfolio repo reads as "I followed a tutorial." A hand-rolled loop with retry logic, token budgets, and turn limits reads as "I understand the mechanics." It is also about 120 lines of code. If you want orchestration signal specifically, LangGraph is acceptable — LangChain's legacy chains are not.

---

## 4. Repository layout

```
contract-intelligence/
├── CLAUDE.md                     # this spec
├── README.md                     # written LAST, as a memo
├── pyproject.toml                # pinned deps via uv.lock
├── .github/workflows/
│   ├── ci.yml                    # lint, types, unit tests, cached eval gate
│   └── eval-nightly.yml          # full live eval, budget-capped
├── docker/
│   ├── Dockerfile                # multi-stage, non-root user
│   └── docker-compose.yml        # api + postgres/pgvector + jaeger + prometheus + grafana
├── src/docintel/
│   ├── config.py                 # pydantic-settings, env-driven
│   ├── schemas.py                # ClauseExtraction, ExtractionResult, ReviewItem
│   ├── ingest/
│   │   ├── loader.py             # CUAD → normalized Document
│   │   ├── chunker.py            # structure-aware (section headers, not fixed-size)
│   │   └── index.py              # build pgvector + FTS indexes
│   ├── retrieval/
│   │   ├── hybrid.py             # BM25 + dense, reciprocal rank fusion
│   │   └── rerank.py             # cross-encoder top-k
│   ├── agent/
│   │   ├── tools.py              # tool schemas + implementations
│   │   ├── loop.py               # turn limit, token budget, retry, timeout
│   │   └── prompts/              # versioned .md files, hashed at runtime
│   ├── validation/
│   │   ├── grounding.py          # verbatim span verification
│   │   └── rules.py              # field + cross-field checks
│   ├── review/
│   │   └── queue.py              # HITL routing (Postgres table)
│   └── api/
│       ├── main.py
│       ├── routes.py
│       └── telemetry.py
├── evals/
│   ├── golden/                   # 150-case curated set, stratified by difficulty tier
│   ├── cases.py                  # case loader + stratification
│   ├── metrics.py                # P/R/F1, span F1, grounding rate, schema validity
│   ├── judge.py                  # LLM-as-judge + human-label calibration
│   ├── cache.py                  # content-hash cache so CI replays cost $0
│   ├── run_eval.py               # CLI entrypoint
│   └── baselines/                # frozen JSON results, committed
├── experiments/
│   └── rag_vs_longcontext.py     # the headline finding
├── tests/
│   ├── unit/                     # chunker, rules, grounding, fusion — no network
│   └── integration/              # API + DB via testcontainers
└── docs/
    ├── ARCHITECTURE.md
    ├── RESULTS.md
    └── LIMITATIONS.md
```

---

## 5. Core contracts

### Output schema (`schemas.py`)

```python
class Evidence(BaseModel):
    quote: str                    # must appear verbatim in source
    char_start: int
    char_end: int
    chunk_id: str

class ClauseExtraction(BaseModel):
    clause_type: ClauseType       # Enum, closed set
    present: bool
    value: str | None             # normalized (ISO dates, canonical jurisdiction names)
    raw_text: str | None
    evidence: list[Evidence]      # >= 1 when present is True
    confidence: float = Field(ge=0.0, le=1.0)

class ExtractionResult(BaseModel):
    document_id: str
    clauses: list[ClauseExtraction]
    violations: list[RuleViolation]
    needs_review: bool
    prompt_version: str           # hash — attributes regressions to prompt changes
    model: str
    usage: TokenUsage             # in, out, cost_usd
    latency_ms: dict[str, float]  # per-stage breakdown
```

### The grounding check — the most important 30 lines in the repo

Every `Evidence.quote` must appear **verbatim** in the source document (after whitespace normalization). If it does not, the extraction is a hallucination and gets rejected mechanically, with no model call required.

This is cheap, deterministic, and catches the failure mode everyone else hand-waves about. Report **grounding violation rate** as a first-class metric alongside F1.

### Deterministic rules (`rules.py`)

Ship at least 25 rules. Examples:

- `effective_date <= expiration_date`
- `governing_law` resolves to a known jurisdiction from a closed list
- If `auto_renewal.present` then `notice_period` must be present (cross-field dependency)
- `cap_on_liability` and `uncapped_liability` cannot both be `present=True`
- Any clause with `present=True` and `evidence == []` is invalid
- `confidence < 0.7` on a Tier-3 clause routes to review

Each rule emits a `RuleViolation` with a severity. This is the public analog of the 120+ underwriting validation checks — same pattern, shareable code.

---

## 6. Evaluation design

### Metrics

| Metric | Definition | Why |
|---|---|---|
| Presence F1 | Per clause type, did we correctly detect presence | Primary quality metric |
| Span token F1 | Overlap with CUAD gold spans | Extraction precision |
| Precision @ 80% recall | CUAD's own convention | Comparable to published numbers |
| Grounding violation rate | % outputs with non-verbatim evidence | Hallucination gate |
| Schema validity rate | % parsed into Pydantic on first attempt | Structured-output reliability |
| Rule violation rate | % results tripping ≥1 deterministic rule | Business-logic correctness |
| Cost per document | USD, mean and p95 | Nobody reports this; you should |
| Latency | p50 / p95, per stage | Ties to production experience |

Report all metrics **broken out by difficulty tier.** Aggregate numbers hide the interesting story.

### Baseline ladder — report all five

1. **Regex/keyword** — for Governing Law and dates only. Establishes the floor.
2. **Zero-shot, no retrieval, truncated document** — the naive LLM approach.
3. **RAG top-k, no rerank** — the standard approach.
4. **RAG + rerank + agent loop** — the system.
5. **Full long-context, no retrieval** — the expensive ceiling.

A results table with all five, plus cost and latency columns, is what separates this from a tutorial. It is also the most valuable interview artifact in the repo.

### Golden set

150 cases, stratified across the three difficulty tiers and across contract length quartiles. Committed to the repo as JSON. Document the sampling procedure in `evals/README.md` — stratified sampling with a stated rationale is a specific skill and worth making visible.

### Judge calibration

For clause types where CUAD labels are ambiguous, add an LLM-as-judge. Then do the thing almost nobody does: **hand-label 60 cases yourself, compute Cohen's kappa between your labels and the judge, and report it.** A judge with κ = 0.71 against human labels is a measured instrument. A judge with no calibration number is decoration.

### CI gate

- **On every PR:** run the 150-case eval against the response cache (content-hash of prompt + model + params). Costs $0, takes under 60 seconds. Fail the build if presence F1 drops more than 2 points against the frozen baseline, or grounding violation rate exceeds 1%.
- **Nightly on main:** live eval with real API calls, hard budget cap (e.g. $3). Writes results to `evals/baselines/` on success.
- Every eval run records the prompt version hash, so a regression can be attributed to a prompt change versus a code change.

This caching design is the detail that makes LLM evals in CI actually viable rather than theoretical. Say so in the README.

---

## 7. The headline experiment

`experiments/rag_vs_longcontext.py`

Contract length in CUAD spans roughly 3k to 100k+ tokens. Measure accuracy, cost, and latency across the length distribution for: full-context, retrieval top-k (k ∈ {5, 10, 20}), and retrieval + rerank.

The expected finding — that there is a crossover point where retrieval starts winning on cost without losing accuracy, and a second point where it starts losing accuracy — is a **quantified engineering tradeoff**, not an opinion. Put the chart at the top of the README.

This is the public version of the RAG-versus-long-context evaluation already on the resume. Now it has a chart and a repo link behind it.

---

## 8. Milestones

Work these strictly in order. Do not let Claude Code scaffold everything at once — it will generate a plausible-looking repo with no working parts.

**M0 — Scaffold (45 min)**
`uv init`, pyproject with pinned deps, ruff + mypy config, CI workflow that runs lint and an empty test suite, docker-compose with Postgres/pgvector.
*Done when:* CI is green on an empty repo and `docker compose up` gives you a working Postgres with the vector extension.

**M0.5 — Data audit and split (1–1.5 hrs)**
Download CUAD, run the five audit checks from §2.2, select the final 12 clause types from the counts, freeze the dev/golden/reserve split, hand-author `jurisdictions.yaml` and `normalization.md`.
*Done when:* `docs/DATA_AUDIT.md` contains the clause count table, base rates, token length distribution, and the offset-integrity verdict; `evals/golden/split.json` is committed; the 12 clause types are locked.

**M1 — Ingestion and retrieval (2–3 hrs)**
CUAD loader, structure-aware chunker (split on section headers with token-count fallback, not fixed windows), pgvector + FTS index build, hybrid retrieval with RRF, cross-encoder rerank.
*Done when:* unit tests for the chunker and fusion pass, and a CLI command retrieves sensible chunks for a hand-written query.

**M2 — Agent and validation (3–4 hrs)**
Tool definitions, tool-calling loop with turn limit / token budget / timeout / retry-on-parse-failure, versioned prompts, Pydantic strict outputs, grounding verifier, 25+ deterministic rules.
*Done when:* end-to-end extraction runs on 10 contracts, grounding check catches at least one real hallucination, and rule violations are surfaced in the output.

**M3 — Evaluation (3–4 hrs)**
Golden set construction, all metrics, response cache, all five baselines, CI eval gate wired up.
*Done when:* `uv run python -m evals.run_eval` prints the full results table, and a deliberately worsened prompt causes CI to fail.

**M4 — Service and observability (2–3 hrs)**
FastAPI routes, OTel instrumentation with per-stage spans, Prometheus metrics, multi-stage Dockerfile with a non-root user, integration tests via testcontainers.
*Done when:* `docker compose up` brings up the full stack, a request produces a trace visible in Jaeger, and `/metrics` exposes latency histograms.

**M5 — Results and writeup (2–3 hrs)**
RAG-versus-long-context experiment, charts, `RESULTS.md`, `ARCHITECTURE.md`, `LIMITATIONS.md`, README as a memo.
*Done when:* someone who has never seen the repo can read the README in three minutes and know what was built, what the numbers are, and what does not work.

**M6 — Optional, only if time allows**
Human-in-the-loop review queue with a minimal UI; judge calibration with hand-labeled data; local vLLM serving path with a LoRA adapter (echoes the FM Global work but costs a full day — skip unless the GenAI-forward framing needs reinforcing).

---

## 9. Working with Claude Code

- **Commit this file as `CLAUDE.md` at the repo root** before starting. It becomes persistent context for every session.
- **One milestone per session.** Start each with: "Read CLAUDE.md. Implement M2 only. Do not touch files outside `src/docintel/agent/` and `src/docintel/validation/`."
- **Require tests before advancing.** "Write the unit tests for the chunker first, then the implementation" produces meaningfully better code than the reverse.
- **Push back on scaffolding.** If it starts generating twelve files of stubs, stop it and narrow the scope. Stub files that never get filled in are the most common failure mode.
- **Review the retrieval and validation code yourself, line by line.** You will be asked about these in interviews. Generated code you cannot explain is a liability, not an asset.
- **Make commits granular and messages real.** The commit history is visible and is itself a signal about engineering practice.

---

## 10. Anti-patterns — these will sink the project

- A Jupyter notebook as the primary deliverable
- `requirements.txt` with unpinned versions
- Chroma or FAISS in-memory instead of a real database
- No tests, or tests that only assert `is not None`
- A README that is a tutorial ("first, install the dependencies...") instead of a memo
- Metrics with no baseline comparison
- `main` as the only branch with no PRs — PRs against your own repo demonstrate the CI gate actually gates
- Claiming production readiness in the README. Say what it is: a rigorously evaluated prototype.

---

## 11. Resume bullets this unlocks

Draft — replace bracketed values with real numbers once measured.

> Built a **contract-intelligence service** (FastAPI, Postgres/pgvector, Docker) that extracts 12 clause types from commercial contracts via a **tool-calling agent with Pydantic-enforced structured outputs**, achieving **[X] presence F1** against expert annotations while holding **p95 latency under [Y] ms**.

> Designed a **verbatim-grounding verifier and 25 deterministic business rules** that mechanically reject unsupported model outputs, reducing hallucinated extractions from **[X]% to [Y]%** with no additional inference cost.

> Shipped an **evaluation harness gating CI on a 150-case stratified golden set**, with content-hash response caching enabling zero-cost regression checks on every pull request and **[X]-point F1 drops failing the build**.

> Quantified the **retrieval-versus-long-context tradeoff** across contract lengths, identifying a **[X]k-token crossover** where retrieval reduced cost per document by **[Y]%** at equivalent accuracy.

---

## 12. Interview talking points this creates

- Why hybrid retrieval with RRF instead of dense-only, and what the ablation showed
- How the grounding check works, why it is cheap, and what class of hallucination it cannot catch
- Why evals belong in CI, and how the response cache makes that economically possible
- What the judge's kappa against your own labels was, and what disagreement patterns you found
- The crossover point in the RAG-versus-long-context data, and how you would choose in a real deployment
- Where the system fails: which Tier-3 clauses, and your read on why

---

## 13. Definition of done

- [ ] CI green, badge in README
- [ ] `docker compose up` produces a working stack from a clean clone
- [ ] Results table with all five baselines, broken out by difficulty tier
- [ ] Cost and latency reported alongside every accuracy number
- [ ] Grounding violation rate below 1%
- [ ] Eval gate demonstrably fails on a deliberately regressed prompt
- [ ] README readable in three minutes, with the tradeoff chart above the fold
- [ ] `LIMITATIONS.md` written honestly — this reads as senior, not weak
- [ ] At least three merged PRs showing the CI gate working
- [ ] `docs/DATA_AUDIT.md` present, with clause counts, base rates, and length distribution
- [ ] Dev/golden split frozen, committed, and described in the README
- [ ] CC BY 4.0 attribution and the CUAD citation present in the README