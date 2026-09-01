# Normalization spec

Canonical forms for `ClauseExtraction.value`. Written before the extractor, so
the cross-field rules in `src/docintel/validation/rules.py` have a fixed target
rather than whatever the model happened to emit.

The governing principle: **`value` is normalized, `raw_text` is verbatim, and
`evidence.quote` must appear byte-for-byte in the source.** Normalization never
touches the evidence path — see [`DATA_AUDIT.md`](../../docs/DATA_AUDIT.md) check 4
for why that separation is load-bearing.

Every rule below is grounded in a value distribution actually observed in CUAD's
`master_clauses.csv`, not in what a well-formed contract *ought* to say.

---

## 1. Dates — `partial_date`

**Do not use `datetime.date`.** CUAD's own gold values are frequently partial:
`[]/[]/2020` is the most common `Agreement Date` value in the corpus and means
"sometime in 2020, day and month unknown". A date type that cannot represent that
either drops the value or invents a precision the contract does not have.

Canonical form is an **ISO 8601 prefix**, truncated at the known precision:

| Source text | Canonical `value` | Precision |
|---|---|---|
| `March 15, 2019` | `2019-03-15` | day |
| `15th day of March, 2019` | `2019-03-15` | day |
| `3/15/19` | `2019-03-15` | day |
| `March 2019` | `2019-03` | month |
| `[]/[]/2020` (CUAD mask) | `2020` | year |
| `5/8/14` | `2014-05-08` | day |

Rules:

- **Two-digit years** resolve to 1950–2049 (`14` → `2014`, `97` → `1997`). CUAD
  contracts are SEC filings from roughly 1996 onward, so this window covers the
  corpus without ambiguity.
- **US month/day order.** These are US SEC filings; `3/15/19` is March 15. Where a
  contract is governed by non-US law and the ordering is ambiguous
  (`5/8/14`), prefer the US reading and let the confidence score carry the doubt.
- **Comparison with mixed precision** uses prefix semantics: `2019` and `2019-03-15`
  are comparable only where the prefix differs. `effective_date <= expiration_date`
  is **skipped, not failed**, when the two cannot be ordered at their shared
  precision. A rule that cannot evaluate must not report a violation.

### The `perpetual` problem — `partial_date_or_perpetual`

`Expiration Date` is not a date field. Its single most common gold value is
`perpetual` (66 of 329 non-empty values, in both `perpetual` and `Perpetual`
casings). Contracts also express open-ended terms as "until terminated" or
"indefinite".

`expiration_date.value` is therefore either a `partial_date` **or** one of two
sentinels:

| Sentinel | Meaning | Source phrasings |
|---|---|---|
| `PERPETUAL` | No end date; runs until terminated | `perpetual`, `Perpetual`, `indefinite`, `until terminated` |
| `UNSPECIFIED` | Clause present but names no date | `[]`, `[* * *]` (redacted in filing) |

Date-ordering rules treat `PERPETUAL` as `+infinity` and skip on `UNSPECIFIED`.

---

## 2. Durations — `duration_days` and `duration`

Notice periods must land on a single integer or the cross-field rules cannot
compare them. `"30 days"`, `"thirty (30) days"`, and `"one month"` must produce
the same value.

**`duration_days`** (used by `notice_period_to_terminate_renewal`) is an
**integer count of days**:

| Source text | `value` |
|---|---|
| `30 days` | `30` |
| `thirty (30) days` | `30` |
| `ninety (90) days'` | `90` |
| `one month` | `30` |
| `3 months` | `90` |
| `one year` | `365` |
| `6 weeks` | `42` |

Conversion constants — deliberately fixed, not calendar-aware:

- 1 week = 7 days
- 1 month = 30 days
- 1 quarter = 90 days
- 1 year = 365 days

A calendar-aware conversion would be more correct and less useful: notice periods
are compared against each other, not resolved against a calendar, and a stable
integer beats a precise one that varies by start date. **Where the contract says
"months", the normalized value is an approximation and the field records that** via
`raw_text`, which always holds the original phrasing.

Business-day qualifiers (`30 business days`) are **not** converted to calendar days.
They normalize to the integer with the qualifier preserved in `raw_text`, and the
discrepancy is a known limitation rather than a silent 1.4x error.

**`duration`** (used by `renewal_term`) keeps more structure, because CUAD's gold
values distinguish a one-off extension from a rolling one — `successive 1 year` is
the single most common value:

```
{ "length_days": 365, "recurring": true }   # "successive one (1) year periods"
{ "length_days": 365, "recurring": false }  # "may be extended for one year"
```

---

## 3. Jurisdictions — `jurisdiction_id`

`governing_law.value` is an `id` from
[`jurisdictions.yaml`](jurisdictions.yaml), never free text. Matching is
case-insensitive with whitespace collapsed.

Three findings from the audit shape this:

1. **Spelling variation dominates the long tail.** England appears five ways in
   CUAD (`England`, `England; Wales`, `England, United Kingdom; Wales, United
   Kingdom`, `England and Wales, UK`, `England and Wales`). All collapse to
   `GB-EAW`. The aliases in `jurisdictions.yaml` marked `(observed)` are literal
   CUAD strings.
2. **Values can be multi-valued.** CUAD separates multiple forums with `;`
   (`Illinois; New York`) and occasionally `,` (`Virginia, Texas`). The field is a
   **list** of `jurisdiction_id`, not a scalar. Single-jurisdiction contracts get a
   one-element list.
3. **Not every value is a jurisdiction.** `the state in which the breach occurs`
   and `THE UNITED STATES TRADEMARK ACT OF 1946` appear in the gold data. These map
   to the sentinels in `jurisdictions.yaml` (`DEFERRED`,
   `NON_JURISDICTIONAL_BODY_OF_LAW`) so that "cannot be normalized" is
   distinguishable from "normalized wrongly". A `DEFERRED` value routes to review.

With those aliases, 446 of 448 semicolon-separated components in CUAD resolve
(99.6%). The two that do not are comma-separated multi-jurisdiction values, handled
by rule 2 above.

---

## 4. Parties — `party_list`

A list of objects, ordered as they appear in the contract:

```
[
  { "name": "Birch First Global Investments Inc.", "role": "Company" },
  { "name": "Mount Knowledge Holdings Inc.",       "role": "Marketing Affiliate" }
]
```

- **`name`** preserves the legal entity name as written, including the suffix
  (`Inc.`, `LLC`, `S.A.`). Suffixes are **not** stripped or standardized — they are
  legally significant and distinguish related entities.
- **`role`** is the defined term the contract assigns (`"Company"`, `"Distributor"`,
  `"Licensee"`), taken from the quotation-marked definition. Null when the contract
  defines no short name.
- **Unicode is NFC-normalized.** CUAD contains at least one entity name where the
  same character is encoded two ways across files; see `DATA_AUDIT.md` check 4.

CUAD's own `Parties-Answer` uses `Name ("Role"); Name ("Role")`. The structured
form above is what the API returns; the CUAD string form is reconstructed only when
scoring against gold.

Note that `parties` is the most multi-span category in the corpus — 2,554 gold spans
across 509 contracts — because each party is annotated at every mention. Scoring uses
the any-span-hit convention from `DATA_AUDIT.md` check 5.

---

## 5. Boolean clauses

Six of the twelve clause types — `change_of_control`, `anti_assignment`,
`cap_on_liability`, `uncapped_liability`, `non_compete`, `exclusivity` — carry no
normalized value at all: `present` plus `evidence` is the entire answer, and
`value` is `null`.

This is deliberate. Inventing a summary string for these invites the model to
paraphrase, and a paraphrase cannot be verified by the grounding check. The
verbatim clause text lives in `raw_text` and the quote lives in `evidence`.

---

## 6. Invariants

Applied to every extraction regardless of clause type:

1. `value` is `null` whenever `present` is `false`. A normalized value for an
   absent clause is a contradiction.
2. `raw_text` is always a verbatim substring of the source when `present` is
   `true` — never a paraphrase, never reflowed.
3. Normalization is **idempotent**: normalizing an already-normalized value
   returns it unchanged. This is unit-tested, because a non-idempotent normalizer
   silently corrupts values on retry.
4. Normalization never mutates the source document, and never shifts a character
   offset. Offsets index into raw text, always.
