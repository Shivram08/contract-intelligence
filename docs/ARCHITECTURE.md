# Architecture

Written during the build rather than after it, so the configuration recorded
here is the one the numbers in [`RESULTS.md`](RESULTS.md) were measured with.

## The extraction agent — frozen configuration

**This config is final as of the M3 measurement run.** It is recorded because
the reported numbers are properties of *this* configuration, and because the
temptation after seeing a result is to tune and re-run, which turns a
measurement into a search.

| Parameter | Value | Why |
|---|---|---|
| model | `claude-sonnet-5` | CLAUDE.md §3. 2.5x cheaper than Opus per token |
| prompt | `extract_v1` | `extract_v2` was measured and was worse; see below |
| `max_turns` | 20 | Safety net. Not the fix for anything |
| `max_cost_usd` | 1.50 per document | $0.50 cut a run off mid-extraction |
| `timeout_s` | 300 | 180 killed a 2,958-token contract at 190s |
| `max_retries` | 3 | Submissions rejected before one is accepted |
| `max_tokens` per turn | 8,000 | |
| read budget | 6 `read_span` calls per run | |
| duplicate-query rejection | 0.90 similarity | |
| oversized reads | clamped to 8,000 chars, not rejected | Correcting an argument the caller can fix should not cost a turn |
| search context padding | ±400 characters per hit | |
| retrieval | hybrid lexical+dense, RRF k=60, 50 candidates/arm, cross-encoder rerank | |

## What was measured about the agent, and left alone

These are **data**, not outstanding defects. They are reported rather than fixed
because the configuration was frozen before the measurement run, and tuning
after seeing results would make the numbers meaningless.

- **Turns-to-submit spans 6 to 18** on five contracts. Two landed at 16 and 18
  of a 20 ceiling, so the ceiling is doing real work rather than sitting idle.
- **Completion rate is 80%** on that sample. A run that does not finish is
  excluded from accuracy scoring and reported in the completion rate; it is
  never scored as zeros, which would measure `max_turns` rather than the model.
- **Present-count variance is wide** — 2/12 to 10/12 across contracts, and the
  *same* contract returned 9/12 and 10/12 on consecutive runs. Run-to-run
  variance is real and the paired comparison in `RESULTS.md` is designed around
  it.
- **One regression.** `AgapeAtpCorp` completed in 9 turns before search-context
  widening and afterwards ended without submitting at turn 10, spending $0.28
  for no output. n=1, but it is a real cost of that change.

## Interventions that were measured and rejected

Recorded because a negative result is only useful if it is written down.

1. **Prompt v2** — asked the agent to batch searches and validate at most once.
   $0.6851 against v1's $0.6370 on the same two contracts, and it wrote the
   payload five times anyway. Kept as `extract_v2.md` so the result is
   reproducible.
2. **The `validate_extraction` dry-run tool** — removed. It was the most
   expensive thing in the loop, and removing it cut the same two contracts from
   $0.6370 to $0.3968. It was also *masking* hallucinations: grounding
   violations went from 0/50 to 1/20 once the agent could no longer fix its own
   fabrications before submission.
3. **Four mechanical guards** (read cap, duplicate rejection, read clamping,
   search widening) — raised completion from 2/5 to 4/5 but *increased* cost
   from $0.1194 to $0.1809 per contract. Attribution from the cached bodies: the
   read cap never bound (25 calls against 30 available), duplicate rejection
   fired twice, and batching did not recover (23 of 45 responses still made a
   single tool call). None of the four did the work; the turn ceiling did.

The pattern across all three: **mechanical changes worked, prompt instructions
did not.** Removing a tool changed behaviour reliably; asking the model to use
it less did not.

## Latency, measured

Median of five contracts, end to end:

| stage | time | share |
|---|---|---|
| model | 57.9s | **97.5%** |
| retrieval | 1.5s | 2.5% |
| validation | 0.004s | 0.01% |

End-to-end latency is governed by **turn count**, and nothing else. Retrieval
optimisation would have been wasted effort — worth stating because measuring
first is what established it.

Validation at 4 milliseconds also substantiates the claim that the grounding
gate adds no inference cost: it is a string search, and it is now measured
rather than asserted.

## Amendment 1 — the reranker is dropped (arm 4 v2)

**The freeze is amended once, on the record.** Any further configuration change
goes through review before it runs.

### What changed

| | arm 4 v1 | arm 4 v2 |
|---|---|---|
| retrieval | hybrid RRF + cross-encoder rerank | hybrid RRF only |
| candidates per search | top_k = 5 (reranked from 50) | top_k = 20 |
| reranker | `BAAI/bge-reranker-v2-m3` | none |

Both remain runnable (`4_rag_rerank_agent`, `4b_rag_top20_agent`) and both are
reported. v1 is not superseded; it is an ablation point.

### Why this is not tuning toward a favourable result

The change was driven by **retrieval-layer evidence measured independently of
the paired comparison, and before any v2 accuracy number existed.** The
retrieval ablation scores rankings against CUAD gold spans — it never calls the
model and never touches presence F1 or span F1.

It found the v1 configuration strictly dominated:

```
hybrid RRF        recall@20 = 0.917  at    102ms p50
hybrid + rerank   recall@5  = 0.917  at 37,035ms p50
```

Identical recall, 1/361 the latency. The cost of the substitution is 2,535 extra
context tokens per search, roughly half a cent. Reranking does earn its recall
(+0.119 recall@5, +0.119 MRR, +0.127 nDCG@5 over hybrid at matched k) and cannot
justify its cost when the same recall is purchasable that cheaply.

It also explains a failure mode: one v1 run consumed its entire 300-second
budget inside the cross-encoder on a **2-chunk** contract — five sequential
rerank calls at p95 latency, not candidate count and not document length.

**Direction disclosure:** this change helps arm 4 — the arm currently losing on
cost and latency, and the arm I built. That is stated here rather than left for
a reader to notice.

### Prediction, recorded before the v2 run

Written down so the run can contradict it:

1. **Latency falls by roughly the rerank share.** v1 spent 10.8–972.4s in
   retrieval; v2 should spend ~0.1s per search.
2. **Cost rises $0.03–$0.08 per document** from the wider context, so roughly
   $0.23–$0.28 against v1's $0.1956.
3. **Completion improves to 10/10.** The single v1 timeout was caused by the
   reranker, and the cause is removed.
4. **Span F1 moves little** — recall is matched at the retrieval layer, so the
   agent should see equivalent evidence.

If span F1 drops materially, that **contradicts the substitution argument** and
needs explaining rather than accepting: it would mean rank order within the
top-20 matters to the agent in a way recall@20 does not capture.

## Amendment 2 — the provider changes to Gemini 3.5 Flash-Lite

**Why:** money, stated plainly. The Sonnet measurement phase spent its $16 and
stopped at 10 contracts (n = 18 paired cases), which the power analysis in
`RESULTS.md` shows cannot separate the arms. The remaining budget is $25 of
Gemini credit. At Sonnet's measured $0.17 per agent run that buys ~150 agent
runs; at Flash-Lite's it buys well over a thousand, which is the difference
between a subset and the full golden set on every arm.

The Sonnet results are not discarded. Their responses are in the committed
cache, so they replay at $0 and stay reportable as a model-comparison row.

### What changed in the code

- `docintel.llm.gemini` adapts Gemini's `generateContent` REST API to the
  Anthropic Messages shape, so the loop, both single-call baselines, the cache
  and the API are unchanged. Provider follows from the model id.
- **Thought signatures** ride on content blocks. Gemini 3 rejects an echoed
  function call without one (HTTP 400, verified live), so they survive the
  cache's JSON round trip and are part of the cache key.
- **No strict tool schemas.** `strict: true` is dropped; Gemini constrains
  generation with `parametersJsonSchema` but does not validate server-side.
  Retry-on-parse-failure is now the only structural guarantee.
- A committed **spend ledger** (`evals/spend_ledger.json`) caps all live runs
  together at $25, checked against each run's worst case before any call.

### Model selection — dev split only, 3 contracts each

Same three dev contracts, `extract_v1`, live:

| model | $/contract | turns | completed | ungrounded spans (pre-fix verifier) |
|---|---|---|---|---|
| `gemini-3.5-flash-lite` | $0.0173 | 7–11 | 3/3 | 2/24 |
| `gemini-3.1-flash-lite` | $0.0114 | 5–11 | 3/3 | 3/32, all 3 routed to review |
| `gemini-3.8-flash` | $0.0680 | 12 (cap) | **0/3** | — |

3.8 Flash exhausted the CLI's 12-turn ceiling on every contract and, at 4x the
cost, could not run the golden set on every arm. 3.1 Flash-Lite is cheaper per
token but paraphrased more. **3.5 Flash-Lite is selected.** Its default
thinking level is `minimal` and it is left there.

### The "hallucinations" were page footers

Of the first six ungrounded Gemini quotes, **five were real sentences with an
SEC page footer silently dropped** — `14 Source: AIMMUNE THERAPEUTICS, INC.,
8-K, 2/5/2020` sits mid-sentence in the source text, and the model quoted the
sentence without it. The sixth was a genuine fabrication: two separate clauses
stitched together with a colon.

The verifier now skips a footer-shaped gap between two words — anchored on the
literal `Source:` and a trailing date, confined to one line — and reports the
match as `ARTIFACT_ELIDED`. It skips nothing else; a dropped word is still
`NOT_FOUND` (unit-tested). Both rates are reported: the **lenient** rate counts
elided matches as grounded, the **strict** rate counts them as violations.

This change was made after seeing model output, on the dev split, before any
golden-set Gemini run. That ordering is the reason it is defensible, and it is
why the strict rate stays in every table.

### Rejected: prompt v3

`extract_v3` added explicit rules for exactly the failures observed — no
ellipses, leave filing artifacts in place, `raw_text` is contract text not the
normalized value. On 10 dev contracts:

| prompt | ungrounded, strict verifier | after footer tolerance | $/contract |
|---|---|---|---|
| `extract_v1` | 2/54 | **0/54** | $0.0138 |
| `extract_v3` | 4/57 | 1/57 | $0.0133 |

No improvement, the third time in this project that a prompt instruction failed
to change behaviour where a mechanical change succeeded. `extract_v1` stays,
which also keeps the Sonnet comparison on an identical prompt.

### Frozen Gemini configuration

| Parameter | Value |
|---|---|
| model | `gemini-3.5-flash-lite`, thinking at its `minimal` default |
| prompt | `extract_v1` |
| `max_turns` | 20 |
| everything else | unchanged from the table at the top |

### Predictions, recorded before the golden-set run

1. **Arm 4 costs $0.01–$0.03 per contract** (dev: $0.0138); arm 5 under $0.01.
2. **Completion is at least 95%** for arm 4 at a 20-turn ceiling.
3. **Lenient grounding violation rate is under 2%; strict is 3–10%,** since
   35% of contracts carry footers.
4. **Arms 4 and 5 make the same presence decision on at least 90% of cases**,
   continuing the Sonnet result on a corpus that fits in context.
5. **Gemini's presence F1 is within 0.10 of Sonnet's** on the 10 contracts both
   have run.
