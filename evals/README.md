# Evaluation

## The split — frozen, seed 42

`golden/split.json` partitions all 510 CUAD contracts into three disjoint sets.
It was generated once by [`scripts/build_split.py`](../scripts/build_split.py) and
is committed. **Do not regenerate it.** Every reported number depends on the
golden set having been untouched during prompt development.

| Split | Contracts | Purpose |
|---|---|---|
| `dev` | 60 | Prompt iteration, debugging, error analysis. Look freely. |
| `golden` | 100 | Reported numbers. Read at milestone boundaries only. |
| `reserve` | 350 | Nightly full run. |

Prompts are developed on `dev` and never on the reported evaluation set. This is
the single easiest flaw to avoid in an LLM evaluation and the most common one to
find, which is why the split is frozen in version control rather than computed at
runtime.

### Sampling procedure

Contracts are ordered by character length and assigned to four equal-count
quartiles. `dev` is drawn first, proportionally across quartiles; `golden` is
drawn from the remainder under the same stratification; the rest is `reserve`.
The pool is sorted before each draw, because `random.sample` is sensitive to input
ordering — without that sort the split would differ between runs that build the
pool from a set.

**Why stratify by length rather than sample uniformly.** The headline experiment
is a cost curve over contract length. Contract lengths in CUAD span two orders of
magnitude (185 to 80,234 tokens, median 6,751), and uniform sampling would leave
the long tail unevenly distributed between `dev` and `golden` by chance. A golden
set that happened to skew short would understate the long-context baseline's cost
and flatter retrieval — exactly the conclusion the experiment is supposed to test.

Resulting distribution:

| Split | Q1 (shortest) | Q2 | Q3 | Q4 (longest) |
|---|---|---|---|---|
| `dev` | 15 | 15 | 15 | 15 |
| `golden` | 25 | 25 | 25 | 25 |
| `reserve` | 88 | 87 | 88 | 87 |

Invariants are asserted in [`tests/unit/test_split.py`](../tests/unit/test_split.py)
against the committed file, so an accidental regeneration fails CI rather than
quietly invalidating the results.

### From 100 contracts to 150 cases

The golden *set* is 100 contracts. The golden *cases* — 150 (contract, clause type)
pairs stratified across the three difficulty tiers — are built from those contracts
in M3. Contract-level disjointness is what prevents leakage; case selection happens
downstream of it.

## Metric conventions, fixed before results were seen

Recorded in [`docs/DATA_AUDIT.md`](../docs/DATA_AUDIT.md) and repeated here because
they are choices, not defaults:

- **Presence** is scored any-span-hit: a hit requires `present=True` and at least
  one returned evidence span overlapping *any* gold span. Finding 1 of 3 gold spans
  is a hit, not 0.33. CUAD's multiple spans are usually one obligation restated in
  several places, so treating them as independent targets would penalize a correct
  extraction for not being exhaustive.
- **Span token F1** is reported alongside, against the union of gold spans. It is
  the metric that penalizes partial coverage, and it is the primary metric for
  high-prevalence clauses where presence F1 is trivially high.
- **Every F1 is reported with its base rate.** `parties` appears in 99.8% of
  contracts; an always-yes classifier scores 0.999 presence F1 there.
- **Metrics are broken out by difficulty tier.** Aggregates hide the story.

## Source of truth

`CUAD_v1.json` is authoritative for presence and spans. `master_clauses.csv` is
used only for its eight normalized value columns. The two disagree on 2 of 16,830
boolean labels — cases where the CSV marks a clause `Yes` with no supporting span —
and a label with no span cannot be verified by the grounding check. See
`docs/DATA_AUDIT.md`.
