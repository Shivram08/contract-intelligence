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
