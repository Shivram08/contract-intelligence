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
