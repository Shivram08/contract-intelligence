# Results

Every number here carries its n, its base rate, and an interval. Where a
comparison cannot support a conclusion, that is stated rather than dressed up.

**The subset is budget-determined, not selected.** A fixed budget of $16 covered
the whole measurement phase. Contracts were drawn from the frozen golden split
by seeded sample (`--sample 43 --sample-seed 42`), and the reported comparison
uses the first 10 of that sample. Nothing was chosen after seeing results.

---

## Headline

> **On a corpus that fits in context, the retrieval agent is indistinguishable
> from a single full-context call on presence detection, while costing 3.9x more
> and completing less often.**
>
> This holds *because* no CUAD contract exceeds 128k tokens
> ([`DATA_AUDIT.md`](DATA_AUDIT.md) check 3: median 6,751, max 80,234). It would
> not transfer to a corpus that does not fit, where retrieval stops being
> optional. The boundary condition is the finding as much as the comparison is.

---

## The paired comparison

Arms 4 (RAG + rerank + agent) and 5 (full context, single call), scored on the
**same** cases. A case counts only where both arms completed; an incomplete run
is excluded and reported in the completion rate, never scored as zeros.

**n = 11 paired cases over 9 contracts. 4 gold positives. Base rate 0.364.**

| | arm 4 (agent) | arm 5 (full context) |
|---|---|---|
| completion | 9/10 | **10/10** |
| presence F1 | 1.000 | 1.000 |
| *trivial always-yes F1* | *0.545* | *0.545* |
| recall CI (Wilson) | [0.51, 1.00] | [0.51, 1.00] |
| **span token F1** | **0.776** | 0.747 |
| **grounding violation** | **0.00%** | 2.33% |
| schema validity | 78% | **100%** |
| rule violations | 40% | 40% |
| **cost / document** | $0.1956 | **$0.0506** |
| turns | 1–8 (median 6) | 1 |

### Presence detection is saturated, and McNemar's test is inapplicable

Both arms produced **identical confusion matrices**: tp=4, fp=0, tn=7, fn=0.
There are **zero discordant pairs**, so McNemar's test has no data to operate on.

That is not "the difference was not significant". It is that the test cannot be
computed: McNemar works on the pairs where the arms disagree, and there are
none. Reporting a p-value here would be inventing one.

The recall interval of [0.51, 1.00] on 4 positives spans half the possible
range. Presence F1 is therefore **not** the primary metric for this comparison,
and scaling n would only help if the arms ever disagreed.

The trivial always-yes classifier scores **0.545** on this set. An F1 of 1.000
means something only against that comparator, which is why it appears in every
row.

### What does separate the arms

- **Span token F1: 0.776 vs 0.747.** The agent's evidence spans overlap the gold
  spans more tightly. This is the metric that penalises partial coverage, and
  the one presence F1 deliberately forgives.
- **Grounding: 0.00% vs 2.33%.** Arm 5 produced a genuinely ungrounded evidence
  span; arm 4 did not. **One violation on n=11 is a signal, not a rate** — it
  should not be quoted as "2.33% hallucination rate" without that caveat. It is
  nonetheless the first real grounding violation from a valid run, and it came
  from the arm without retrieval.
- **Schema validity: 78% vs 100%.** The agent needed retries; the single call
  did not.
- **Cost: 3.9x.** $0.1956 against $0.0506 per document.

---

## Baseline 1: the regex floor

Run on the full 150-case set, $0.

| | value |
|---|---|
| presence F1 | 0.188 |
| trivial always-yes F1 | **0.718** |
| precision / recall | 0.750 / 0.107 |
| tier 1 / 2 / 3 F1 | 0.321 / 0.000 / 0.000 |

**Pattern matching scores worse than answering "yes" to everything.** Base rates
are high enough that a trivial classifier reaches 0.718 while the regex reaches
0.188. Reported because a floor that loses to a constant is more informative
than one that beats it — and because an F1 of 0.188 with no comparator beside it
would have looked like a working baseline.

---

## Methodology: four times a number measured the wrong thing

This is the most transferable result in the project. Four times, the harness
produced a number that was present, plausible, and wrong. None raised an error.
Each was caught by accident.

**1. Budget-exhausted runs scored as 0/12 present.** A run that hit the turn
ceiling looked like total recall loss, so presence F1 was partly a measurement
of `max_turns`. Fixed by giving runs a terminal state and excluding incomplete
ones from accuracy scoring entirely.

**2. Cache-read time reported as model latency.** Replaying a run recomputed
latency at replay time, so a frozen baseline claimed a 972ms p50 for an arm
whose live p50 was 44 seconds. Fixed by storing latency at capture; a replayed
entry without it now reports *unavailable* rather than zero.

**3. An entire agent evaluation ran against an empty index.** The dev split was
indexed in M1; the golden split is disjoint from it *by construction* and never
was. Every `search_contract` call returned nothing. Arm 4's completion rate,
cost, turn count and span F1 were all measured on a retrieval system that was
not present. Cost: $1.66 and three misdiagnoses.

**4. Three interventions were built to fix a failure that did not exist.**
This is the one worth reading twice. The sequence:

- Arm 4 stalled: 60% of runs hit the turn ceiling without submitting, searching
  13–33 times per run.
- **Prompt v2** asked the agent to batch searches and validate once. It made
  things *worse* — $0.6851 against v1's $0.6370, and the payload was written
  five times anyway.
- **A guard set** followed: a read cap of 6, near-duplicate query rejection at
  0.90, client-side read clamping, widened search context, `max_turns` raised to
  20. Completion went 2/5 to 4/5 but cost *rose*, from $0.1194 to $0.1809.
- Then the empty index was found. With real retrieval, turns halved (1–8, median
  6), searches more than halved (6–16), and span F1 went 0.636 to 0.776.

Measured after the fix, the guards are largely inert:

| guard | binding? |
|---|---|
| read cap of 6 | **never** — 45 calls against 60 available |
| `max_turns` = 20 | **never approached** — max used was 8 |
| duplicate rejection ≥0.90 | fires: 48 of 177 queries |

Two of three interventions were treating an artifact of broken instrumentation.
The generalisable lesson is not "check your index" — it is that **a component
working correctly in isolation tells you nothing about whether the assembled
system is measuring what you claim.** Unit tests passed throughout all four
incidents.

### Preflight invariants

The response to incident 4 was not another fix but a check that refuses to run.
[`evals/preflight.py`](../evals/preflight.py) asserts, before any spend:

- every document in the case set is in the retrieval index
- every one of those documents has chunks, and the chunks have embeddings
- a canary query returns hits
- the documents belong to the split they claim to
- the agent config matches the frozen record in [`ARCHITECTURE.md`](ARCHITECTURE.md)
- latency is stored at capture rather than derived at replay

`run_eval` exits 2 rather than running if any fail. It earned its place
immediately by catching a `max_turns` default of 12 in `run_eval` disagreeing
with the 20 recorded in `ARCHITECTURE.md` — which would have made the run
unattributable to any stated configuration.

---

## Cost and the eval cache

Measured, not estimated:

| | $/document |
|---|---|
| baseline 1 (regex) | $0.0000 |
| baseline 5 (full context) | $0.0506 |
| baseline 4 (agent) | $0.1956 |

A full live run of the 150-case golden set costs roughly $11 per model-driven
arm. Nobody runs that on a pull request. The content-hash response cache makes
replay cost nothing and take seconds, which is what makes an eval gate in CI
viable rather than theoretical — and the cache is read-only in CI, so a miss is
a hard error instead of a surprise bill.

One measured cost lesson: prompt caching covered only the system block at first,
so the growing conversation was re-priced at full rate every turn and input cost
was quadratic in turn count. Caching the conversation cut a single contract from
$0.5212 to $0.2721.

---

## Latency

*Pending: per-stage p50/p95 with the reranker separated from search. The current
single "retrieval" figure conflates ~1.5s of search with 122–972s of
cross-encoder reranking, so arm 4's end-to-end latency is not yet reportable.
The retrieval ablation below quantifies whether reranking earns that.*

What is established:

- **Model time dominates.** Median of five contracts: model 97.5%, retrieval
  2.5%, validation 0.01%. End-to-end latency is governed by turn count.
- **Validation costs 4 milliseconds.** The grounding gate and 30 deterministic
  rules add no inference cost — measured, not asserted.
- One arm-4 run consumed its entire 300s budget inside the reranker on a
  **2-chunk** contract, so the cause is not candidate count or document length.
  Unattributed.

---

## Retrieval ablation

*Pending — running on all 150 cases at $0.*
