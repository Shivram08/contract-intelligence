# Limitations

What this system does not establish, and what it would take to establish it.
Written as a list of things that would be fair to attack in review.

---

## The evaluation is small, and the reason is money

A fixed $16 budget covered the whole measurement phase. The paired comparison
rests on **11 cases over 9 contracts, with 4 gold positives**.

Contracts were drawn by seeded sample (`--sample 43 --sample-seed 42`) from the
frozen golden split, and the reported set is the first 10 of that sample. The
subset is budget-determined, not selected after seeing results — but it is small
enough that most per-clause figures are not interpretable, and the interval on
recall ([0.51, 1.00]) spans half the possible range.

Closing this costs roughly **$0.25 per additional paired contract**. Reaching
~35 paired cases would cost about $4; the full 150-case set across both arms
would cost about $27.

## Presence detection is not measured, it is saturated

Both arms produced identical confusion matrices and **zero discordant pairs**.
McNemar's test is *inapplicable*, not non-significant. Any claim that one arm
detects clauses better than the other is unsupported in either direction.

Increasing n only helps if the arms ever disagree. On this evidence they do not.

## Long-context accuracy IS measured; long-context *scaling* is not

Arm 5 was run end to end, so its accuracy is measured rather than assumed. What
is **not** measured is how either arm behaves on documents that do not fit in
context — because no CUAD contract does. Median 6,751 tokens, max 80,234, and
nothing above 128k.

Every comparative claim here is conditional on that. On a corpus of 500k-token
agreements, retrieval stops being optional and the result would likely invert.
This is the single most important boundary on the headline.

## Grounding: 0.00% is conditional on relocation working

The grounding violation rate rests on a fact that surprised me: **100% of
evidence spans required relocation.** `relocated=32` and `relocated=29` — every
span. The models essentially never compute correct character offsets; grounding
recovers the position by searching for the quoted text.

So grounding is not a gate that occasionally repairs an offset. It is a
load-bearing component without which **neither arm produces valid output at
all**.

Two consequences worth stating plainly:

- A 0.00% violation rate is **not** evidence that the model does not fabricate
  quotes. It is evidence that every quote it produced was locatable in the
  source. Those are different claims.
- The defensible claim is a mechanically-enforced invariant that makes
  structured extraction viable when the model cannot self-report position,
  recovering 100% of spans from quoted text at ~4ms and zero inference cost.
  Not a hallucination-reduction percentage.

The negative case is unit-tested: a quote absent from the source is still
rejected even with unknown offsets. That test is the only thing separating
"invariant" from "escape hatch".

Arm 5's single grounding violation is **one observation on n=11** — a signal,
not a rate. It should not be quoted as "2.33%" without that qualifier.

## The arms are not perfectly matched

**Retry budget: 3 versus 1.** The agent loop retries a failed submission three
times; the single-call arm retries once. Zero-versus-three was a genuine
handicap and was corrected; matching all three would make "single call" a
misnomer. One is a judgement call that *favours the arm expected to win*, and it
is recorded rather than buried.

In the event the asymmetry never mattered: **the retry never fired.** Arm 5
reached 10/10 completion on the schema fix alone (`retries_used = 0`). The
asymmetry exists in code and is currently untriggered.

## Two guards are inert, and stayed in anyway

Measured against real retrieval, the read cap never binds (45 of 60 available)
and `max_turns=20` is never approached (max used: 8). Both were built to fix a
stall that turned out to be caused by an empty index.

They remain in the configuration because it was frozen before the measurement
run and changing it afterwards would turn a measurement into a search. They
should be removed before any future run, and the run re-baselined.

## Reranker cost is not yet accounted for

Arm 4 spent 122s, and in one case 972s, inside the cross-encoder against roughly
1.5s of actual search. The 972s case was a **2-chunk** contract, so it is not
explained by candidate count or document length, and it is currently
unattributed. Until the per-stage split lands, arm 4's end-to-end latency is not
reportable.

## Not production-ready

This is a rigorously evaluated prototype. It has no authentication, no rate
limiting, no multi-tenancy, no retry/backoff policy tuned for a real SLA, and no
human review workflow beyond a routing flag. The review queue routes items; it
does not manage them.

## Dataset caveats

CUAD is CC BY 4.0, and the Atticus Project makes no representations about the
licence status of the underlying contracts themselves — they are public EDGAR
filings, used here as distributed.

Two ground-truth defects were found and are documented in
[`DATA_AUDIT.md`](DATA_AUDIT.md): two contracts carry a `Yes` label in
`master_clauses.csv` with no supporting span in `CUAD_v1.json`. `CUAD_v1.json`
is treated as authoritative for presence, since a label with no span cannot be
verified by a grounding check.

The corpus also contains 15 contracts under 500 tokens — mostly one-paragraph
SEC joint filing agreements. They carry almost no clauses, so a system scores
well on them for reasons unrelated to extraction quality.

## Clause scope

12 of CUAD's 41 categories. Most Favored Nation was dropped for having 28
positives corpus-wide, below the ≥40 bar, and replaced with Exclusivity. The
substitution and the alternatives considered are recorded in
[`clause_schema.yaml`](../data/reference/clause_schema.yaml) rather than made
silently.
