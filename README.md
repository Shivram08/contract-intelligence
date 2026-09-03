# Contract Intelligence Service

[![CI](https://github.com/Shivram08/contract-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Shivram08/contract-intelligence/actions/workflows/ci.yml)

Extracting 12 clause types from commercial contracts — governing law, liability
caps, assignment restrictions and nine others — with every claim backed by a
verbatim quote from the source, validated by 30 deterministic rules, and
measured against expert annotations. Built on [CUAD](https://www.atticusprojectai.org/cuad),
510 SEC-filed contracts with 13,101 lawyer-labelled clauses.

---

## What this found

> **On a corpus that fits in a model's context window, the retrieval-plus-agent
> pipeline and a single full-context call made byte-identical extraction
> decisions on every case in the affordable sample — while the agent cost ~4×
> more.** The quality comparison could not separate them at any n this project
> could pay for; the operational differences are decisive without it.
>
> This holds **because no CUAD contract exceeds 128k tokens** (median 6,751,
> max 80,234 — [`DATA_AUDIT.md`](docs/DATA_AUDIT.md)). On a corpus that does not
> fit, retrieval stops being optional and this result would not transfer. The
> boundary condition is as much the finding as the comparison is.

The results that carry real statistical weight are not the head-to-head. They
are the retrieval ablation (84 cases), a metric-validity error that invalidated
my own conclusion, and the finding that grounding is load-bearing rather than a
safety net.

**Three-minute version:** the box above, plus the reranker table below. Stop
there and you have the finding. Everything after
[Headline results](#headline-results) is supporting depth — the paired
comparison and why it is not presented as a result, a power analysis of what
telling would have cost, and a record of five occasions when this harness
produced a plausible number that measured the wrong thing.

---

## Headline results

### Reranking earns its recall and cannot justify its cost

84 positive cases, 59 contracts, scored against CUAD gold spans. **$0 — no model
calls.**

| arm | R@5 | R@10 | R@20 | MRR | nDCG@5 | p50 | p95 |
|---|---|---|---|---|---|---|---|
| lexical only | 0.774 | 0.905 | **0.964** | 0.583 | 0.589 | **13ms** | 45ms |
| dense only | 0.679 | 0.833 | 0.905 | 0.610 | 0.578 | 117ms | 192ms |
| hybrid RRF | 0.798 | 0.917 | 0.917 | 0.639 | 0.628 | 102ms | 173ms |
| hybrid + cross-encoder | **0.917** | 0.929 | 0.952 | **0.758** | **0.755** | 37,035ms | 67,545ms |

Reranking buys a real, large gain — +0.119 recall@5, +0.119 MRR — for **361× the
latency**. And it is redundant:

```
hybrid RRF        recall@20 = 0.917  at    102ms
hybrid + rerank   recall@5  = 0.917  at 37,035ms
```

**Identical recall.** Retrieving 20 unreranked candidates gets what reranking 50
down to 5 gets, at 1/361 the latency, for ~2,500 extra context tokens (about
half a cent).

Confirmed independently by a live trace through the deployed stack — a different
measurement path entirely:

```
docintel.extract              7371.2ms
  docintel.retrieval          2558.1ms  hits=5
  docintel.rerank             4808.3ms  candidates=5
  docintel.validation            0.4ms  clauses=12
```

And it explains a pathology: one run consumed its entire 300-second budget
inside the cross-encoder on a **two-chunk** contract — five sequential rerank
calls at p95 latency, not candidate count and not document length.

Two secondary findings, same run: **lexical retrieval beats dense at every k**
(0.774/0.905/0.964 vs 0.679/0.833/0.905) and is 9× faster. At R@20 lexical alone
beats every other arm, reranked included.

### recall@k could not see the thing that mattered

This is the most transferable result here, and it came from being wrong.

The ablation above says hybrid@20 and hybrid+rerank@5 have identical recall, so I
dropped the reranker and re-baselined. I **pre-registered four predictions**
before running ([`ARCHITECTURE.md`](docs/ARCHITECTURE.md)):

| # | predicted | actual | |
|---|---|---|---|
| 1 | latency falls by the rerank share | retrieval 972s → 1.6s | ✅ |
| 2 | cost rises $0.03–0.08 | rose **$0.1018** | ❌ 13–34× off |
| 3 | completion improves to 10/10 | **8/10**, worse | ❌ |
| 4 | span F1 moves little | **−0.074** | ❌ |

**Recall parity at the retrieval layer did not transfer end to end.** `recall@20`
says the right chunk is *somewhere* in the 20. It says nothing about it being
near the top — and the agent consumes rank order, not sets. The reranker's real
contribution was ordering, which recall@k is blind to by construction.

MRR and nDCG@5 *did* show the gap (0.639→0.758, 0.628→0.755). I read them as
corroborating recall rather than as measuring something recall could not see.
That is a metric-selection error, not a modelling one, and it is the kind that
survives peer review because every individual number is correct.

### Grounding is load-bearing, not a safety net

**100% of evidence spans required relocation.** The models essentially never
compute correct character offsets; the grounding verifier recovers position by
searching for the quoted text.

So a 0-violation rate is **not** evidence that the model does not fabricate
quotes. It is evidence that every quote it produced was locatable. Those are
different claims. The defensible version:

> A mechanically enforced invariant that makes structured extraction viable when
> the model cannot self-report position — recovering 100% of spans from quoted
> text at **0.4ms** and zero inference cost.

The negative case is unit-tested: a quote absent from the source is still
rejected even with unknown offsets. That test is the only thing separating
"invariant" from "escape hatch."

---

## The paired comparison, and why it is not presented as a result

**n = 18 paired cases over 10 contracts. 7 gold positives. Base rate 0.389.
Trivial always-yes F1 = 0.560.**

| metric | arm 4 (RAG+rerank+agent) | arm 5 (full context) | separates? |
|---|---|---|---|
| presence F1 | 1.000 (trivial 0.560) | 1.000 (trivial 0.560) | **no** — identical |
| confusion matrix | tp=7 fp=0 tn=11 fn=0 | tp=7 fp=0 tn=11 fn=0 | **no** — byte-identical |
| recall CI (Wilson, 7 pos) | [0.65, 1.00] | [0.65, 1.00] | no |
| span token F1 | 0.716 | 0.691 | no — Δ0.025 on 7 positives |
| grounding violations | 0 of 46 spans | **1** of 46 spans, CI [0.00, 0.11] | no — one observation |
| schema validity | 80% (2 retries of 10) | 100% | no |
| rule violation rate | 40% | 40% | no |
| completion (paired) | 10/10 | 10/10 | no |
| **cost per document** | **~4× arm 5** | — | **yes, order of magnitude** |
| end-to-end latency | **unmeasured** | 9.8s p50 / 15.7s p95 | see below |

**The arms did not merely fail to differ significantly — they made exactly the
same decision on every case.** Byte-identical confusion matrices, zero
discordant pairs, identical rule-violation rates. On a corpus that fits in
context, the retrieval-plus-agent apparatus changed nothing about what got
extracted. That is a substantive finding, not a null result.

**McNemar's test is inapplicable**, not non-significant: it operates on
discordant pairs and there are none. Reporting a p-value would be inventing one.

### What n would have been needed

Measured per-case span F1 standard deviation is 0.366. To detect the observed
Δ0.025 at 80% power, α=0.05:

| σ of paired difference | cases needed | contracts | cost |
|---|---|---|---|
| 0.091 (arms highly correlated) | 105 | 58 | **$16** |
| 0.183 | 419 | 233 | $63 |
| 0.366 (per-arm σ) | 1,678 | 932 | $252 |

The **most optimistic** figure equals the entire project budget. The middle case
needs 233 contracts — more than the 100-contract golden split contains. The
pessimistic case exceeds all 510 contracts in CUAD. This is not "we couldn't
tell"; it is a quantified boundary on what telling would have cost.

### Why the small-n figures are not presented as results

The same regex detector, scored on two sets:

| set | n | base rate | trivial F1 | regex F1 | verdict |
|---|---|---|---|---|---|
| full golden set | 150 | 0.560 | **0.718** | 0.260 | loses badly to trivial |
| paired subset | 8 | 0.375 | **0.545** | **0.800** | beats trivial |

**Opposite conclusions from an identical, deterministic detector.** If n=8 can
flip the floor from "worse than answering yes to everything" to "comfortably
above trivial," it can flip anything. That is the cleanest evidence available
for why the arm-4/arm-5 quality numbers above are reported with intervals and a
"does not separate" column rather than as findings.

### How 150 cases became 18

```
150 golden cases      (contract, clause_type) pairs, stratified, seed 42
  →  81 contracts     the contracts those cases belong to
  →  10 attempted     budget: $16 total for the whole measurement phase
  →  10 both complete  after a schema fix; was 5/10 before it
  →  18 paired cases  the cases belonging to those 10 contracts
```

Seed 42, `--sample 43 --sample-seed 42 --limit 10`. The reduction is budget and
completion rate. Nothing was selected after seeing results.

**Live versus replay, stated plainly.** Observed live, pre-schema-fix: arm 4
completed 7/10 attempts, arm 5 6/10. The 10/10 in the table comes from a cache
replay under identical current validation rules — **it was never observed live.**
That makes it a fairer *comparison* (both arms scored by the same rules) of
outputs generated under the old regime; the model responses were produced by a
system that was rejecting arm-5 submissions at construction. The scoring changed;
the generation did not.

### Latency

Arm 4's end-to-end p50/p95 is **unmeasured**. Capture-time latency landed in the
response cache after that run, so those entries carry none, and the harness
reports *unavailable* rather than substituting zero. Closing it costs ~$2 for one
10-contract run; judged not worth it against a gap already decisive at the
retrieval layer:

**Arm 4's retrieval stage alone (10.8–972.4s, live-measured) exceeds arm 5's
complete end-to-end p95 (15.7s) in nearly every run — by ~62× at the top end.**
Arm 5 never retrieves: it is one call with the document inline and no
`search_contract` tool, so the retrieval bug described below could not have
touched its figures.

Validation costs **0.4ms**, measured in a live trace.

---

## Methodology: how often a plausible number measured the wrong thing

Five times this harness produced a number that was present, plausible, and
wrong. None raised an error. Each was caught by accident, and the record is kept
because the failure mode is the transferable part.

1. **Budget-exhausted runs scored as 0/12 present.** A run that hit the turn
   ceiling looked like total recall loss, so presence F1 was partly a
   measurement of `max_turns`. Runs now carry a terminal state and incomplete
   ones are excluded from accuracy entirely.
2. **Cache-read time reported as model latency.** Replay recomputed latency at
   replay time, so a frozen baseline claimed a 972 ms p50 for an arm whose live
   p50 was 44 seconds. Latency is now stored at capture; a replayed entry without
   it reports *unavailable*, never zero.
3. **An entire agent evaluation ran against an empty index.** The dev split was
   indexed; the golden split is disjoint from it *by construction* and never was.
   Every `search_contract` call returned nothing. Cost, completion, turn count
   and span F1 were all measured on a retrieval system that was not there. Cost:
   $1.66 and three misdiagnoses.
4. **Three interventions were built to fix a failure that did not exist.** A
   prompt rewrite and a five-guard set, both targeting a "scanning and
   re-searching" stall that was entirely an artifact of #3. Measured afterwards,
   two of the three guards never bind: the read cap used 45 of 60 available, and
   `max_turns=20` was never approached (max used: 8).
5. **A voided figure came back as evidence.** A stage split of "97.5% model /
   2.5% retrieval" from the smoke test — whose "retrieval" was the cost of
   returning zero rows — was struck when #3 surfaced, then cited in a later
   report as free latency evidence. It is recorded as void rather than deleted
   in [`ARCHITECTURE.md`](docs/ARCHITECTURE.md).

The common shape: every component was individually correct, and what failed was
the claim that the assembled system was the thing being described. **Unit tests
passed throughout all five.**

### Four milestone-completion claims falsified by execution

Every one would have shipped:

- An **invented base-image digest** in the Dockerfile — would have failed on the
  first build.
- **`uv sync --extra` on one layer but not the next.** `uv sync` prunes to the
  requested set, so the second sync silently removed `torch`. The container
  raised `ModuleNotFoundError: sentence_transformers` behind a swallowed
  exception, surfacing only as `/healthz` reporting `extractor: absent` with no
  hint why — an image that could not do dense retrieval at all.
- **`setup_tracing` returned a tracer without installing it**, so every span
  would have been a silent no-op and traces would have been empty.
- **`stage_span("model")` and `stage_span("validation")` were never called**, and
  `telemetry.py` sat under `api/` where the core pipeline could not import it
  without inverting the layering. The first verified trace had **one** span; it
  now has six.

Four of four. The argument is for acceptance criteria that **run** rather than
criteria that describe.

### Preflight invariants

The response to #3 was not another fix but a check that refuses to run.
[`evals/preflight.py`](evals/preflight.py) asserts, before any spend: index
coverage over the exact case set, chunks present, embeddings present, a canary
query returning hits, split membership, agent config matching the frozen record,
and latency stored at capture. `run_eval` exits 2 rather than running.

It earned its place immediately, catching a `max_turns` default of 12 in
`run_eval` disagreeing with the 20 recorded in `ARCHITECTURE.md` — which would
have made the run unattributable to any stated configuration.

### Other measured decisions

- **Prompt instructions did not change behaviour; removing a tool did.** Asking
  the agent to "validate at most once" made things *worse* ($0.6851 vs $0.6370);
  it wrote the payload five times anyway. Deleting the dry-run tool cut the same
  two contracts to $0.3968.
- **That dry-run tool had been masking hallucinations.** Grounding violations
  went from 0/50 to 1/20 once the agent could no longer fix its own fabrications
  before submission.
- **Caching only the system prompt made input cost quadratic in turn count.**
  Caching the conversation cut one contract from $0.5212 to $0.2721.
- **Postgres FTS is not BM25.** `ts_rank_cd` is a cover-density rank with no
  `k1`/`b`. The code calls that arm `lexical` so the results table does not claim
  an algorithm it does not run.
- **`websearch_to_tsquery` conjoins terms**, so a sentence-length query matched
  0 chunks and "hybrid" retrieval silently degraded to dense-only for exactly
  the queries the agent generates. OR semantics: 2,148 matches.

---

## Architecture

```
FastAPI  →  /extract  /extract/batch  /healthz  /metrics
              ↓
        Orchestrator (docintel.extract) — shared with the eval harness
              ↓
   ┌──────────┴───────────┐
   Retrieval              Agent loop (hand-rolled)
   Postgres FTS +         turn / cost / timeout / retry budgets
   pgvector, RRF          tools: search_contract, read_span,
   + cross-encoder        get_schema, submit_extraction
              ↓
        Validation — verbatim grounding, then 30 deterministic rules
              ↓
        OTel spans → Jaeger    Prometheus → Grafana
```

Routes call the same orchestrator the eval harness does, so `RESULTS.md` cannot
describe a system the service does not run. Details, the frozen agent
configuration, and the rejected interventions are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quickstart

```bash
docker compose -f docker/docker-compose.yml up -d
```

Brings up the API, Postgres/pgvector, Jaeger, Prometheus and Grafana. Runs in
**stub mode** by default — a deterministic canned model client, so the whole
request path, traces and metrics are exercisable at zero cost. `/healthz`
reports `extractor: stub`, so a demo can never be mistaken for a measurement.

```bash
curl -s localhost:8000/extract -H 'content-type: application/json' -d '{
  "document_id": "DEMO",
  "text": "AGENCY AGREEMENT\n\n12. GOVERNING LAW. This Agreement shall be governed by the laws of the State of Delaware."
}'
```

Trace: <http://localhost:16686> · Metrics: <http://localhost:8000/metrics> ·
Grafana: <http://localhost:3000>

For real extraction, put `ANTHROPIC_API_KEY=sk-ant-...` in `.env` (gitignored)
and unset `DOCINTEL_LLM_STUB`. Expect ~$0.20 per contract.

```bash
# Evaluation. The regex floor and the retrieval ablation cost nothing.
uv run python -m evals.run_eval --baselines 1_regex
uv run python -m evals.retrieval_ablation
uv run python -m evals.run_eval --gate          # CI gate, replays from cache, $0
```

CUAD is not vendored (~165 MB); see [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md)
for the expected layout under `data/raw/`.

---

## What this is not

A rigorously evaluated prototype, not a production service. No auth, no rate
limiting, no multi-tenancy, and batch jobs live in process memory. The honest
limits — including what the paired comparison cannot support — are in
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

---

## Data and attribution

Built on the **Contract Understanding Atticus Dataset (CUAD) v1**.

> Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). *CUAD: An
> Expert-Annotated NLP Dataset for Legal Contract Review.* Proceedings of the
> Neural Information Processing Systems Track on Datasets and Benchmarks.

CUAD is distributed by [The Atticus Project](https://www.atticusprojectai.org/cuad)
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The Atticus
Project makes no representations about the licence status of the underlying
contracts, which are public SEC EDGAR filings.
