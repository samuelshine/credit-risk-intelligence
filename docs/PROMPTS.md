# Prompt Engineering, Token Optimisation and Hallucination Control

Every number in this document was measured, not estimated. Token counts come
from one of two sources, and each table says which:

* **Measured (API)** — the Gemini API's own `usage_metadata` on a real call, or
  `client.models.count_tokens(model="gemini-3.7-flash", ...)`.
* **Estimated (chars/4)** — `SchemaCard.approx_tokens`, the offline heuristic
  the code uses at selection time when no API round trip is available.

Where both exist for the same text they are shown side by side, so the
heuristic's error is visible rather than hidden. It runs about 5-11% low.

Measurements in this document were taken on 2026-09-07 against the real
307,511-row Home Credit database (`data/credit_risk.duckdb`, 218 chatbot-
visible columns across 7 tables) and the live Gemini API on a free-tier key.
Prompt version at time of measurement: **`v1.0.0`** (`PROMPT_VERSION` in
`src/talk_to_data/prompt_templates.py`, recorded on every `Answer`).

---

## 1. The pipeline

```
  question
     │
     ▼
  render_schema_card(catalog, question)        src/talk_to_data/schema_card.py
     │   tier 1: all 7 tables, row counts, join keys        (always)
     │   tier 2: full column detail for the 3 top-scoring tables,
     │           ≤28 columns each, chosen by domain prior + question match
     ▼
  build_sql_prompt(question, card)             src/talk_to_data/prompt_templates.py
     │   system = rules + refusal protocol + schema card + dataset notes
     │   user   = 3 similarity-selected worked examples + the question
     ▼
  Gemini  role="sql"  gemini-3.7-flash  temp=0  seed=42  max_output=600
     │
     ├── text starts with "CANNOT_ANSWER:" ──► refuse, stop. 1 API call total.
     ▼
  validate_sql(text, catalog)                  src/talk_to_data/sql_validator.py
     │   sqlglot parse (duckdb dialect) → AST checks → LIMIT injected/clamped
     │
     ├── rejected ──► build_repair_prompt(bad_sql, errors)
     │                  └─► Gemini role="sql" ONCE more, then re-validate.
     │                      Second failure is reported as a failure.
     ▼
  run_query(sql)                               src/talk_to_data/query_runner.py
     │   read-only DuckDB cursor, enable_external_access=false,
     │   30 s watchdog → cursor.interrupt(), fetchmany(limit+1)
     ▼
  result.to_markdown(max_rows=50)              rows only — no SQL, no schema
     │
     ▼
  Gemini  role="summary"  gemini-3.5-flash-lite  max_output=400
     │
     ▼
  Answer(answer, sql, rows, usage[], repaired, refused, prompt_version, ...)
```

Two API calls on the happy path, three when a repair fires, one on a refusal.
Every stage is recorded on the `Answer` object and rendered in the UI, which is
what makes an answer auditable rather than merely asserted.

The pipeline is **stateless between questions** by design (`TalkToData`'s
docstring): no conversation memory, so the prompt cannot grow monotonically and
an earlier wrong answer cannot contaminate a later one.

---

## 2. Prompt templates

All prompts live in `src/talk_to_data/prompt_templates.py` and are versioned.
Changing one is a reviewable diff, and `PROMPT_VERSION` is attached to every
answer, so any transcript below can be traced back to the prompt that produced
it.

### 2.1 SQL generation — system prompt

```
You are a SQL analyst for a bank's credit risk team. You translate questions
about the Home Credit loan portfolio into a single DuckDB SQL query.

RULES
1. Output raw SQL only. No prose, no explanation, no markdown fences.
2. Write exactly one SELECT statement. Never write INSERT, UPDATE, DELETE,
CREATE, DROP, COPY, ATTACH or PRAGMA.
3. Use only the tables and columns given in the schema below. Never invent a
column name. If you need a column that is not listed, the question cannot be
answered.
4. Never call functions that read outside the database, such as read_csv or
glob.
5. Prefer readable aliases (AS default_rate, AS applications) - they become
the column headings a non-technical person reads.
6. Always return the count of rows behind any rate or average, so the reader
can judge whether the number is reliable.
7. Order results so the most interesting row is first, and limit long results.
8. If the question cannot be answered from this schema, or is not about this
loan portfolio, reply with exactly one line:
   CANNOT_ANSWER: <short reason>

SCHEMA
{schema}
```

Why each rule is there:

| Rule | Why it exists |
|---|---|
| 1 | Prose around the SQL means the parser has to guess where the query starts. The validator tolerates a stray ```` ```sql ```` fence anyway (`_strip_markdown_fence`) because rejecting a correct query over formatting would cost a needless retry — but tolerating is not inviting. |
| 2 | Cheap first line of defence. It is *not* the control: `test_write_and_admin_statements_are_refused` asserts the validator refuses all nine write/admin forms regardless of what the prompt says. |
| 3 | The specific failure this system exists to prevent. Paired with the injected schema so "the columns that exist" is a fact in the prompt, not a memory. |
| 4 | Same shape as rule 2 — prompt-level discouragement of the filesystem escape that `FORBIDDEN_FUNCTION_PATTERNS` actually blocks. |
| 5 | The aliases become the column headings a loan officer reads in the UI table. `avg(TARGET)` as a heading is not an answer. |
| 6 | A rate without its denominator is not auditable. Every worked example returns `count(*)` alongside the rate, so the habit is taught twice. |
| 7 | Bounds the result before the validator has to clamp it, and puts the interesting row where the summariser will see it inside `max_summary_rows=50`. |
| 8 | The refusal protocol. See §4.2 — this is the single most important line in the prompt. |

### 2.2 SQL generation — user prompt

```
{examples}Question: {question}

SQL:
```

`{examples}` is filled by `render_examples(select_examples(question, limit=3))`.
Each block is `Question: … / SQL: … / -- note`, where the note teaches a
dataset quirk rather than a SQL pattern:

* `-- A rate is avg(TARGET); always return the denominator with it.`
* `-- DAYS_BIRTH is negative; negate before converting to years.`
* `-- 365243 is the 'not employed' sentinel and must be excluded.`
* `-- Aggregate the child table first, then join, or rows double count.`
* `-- COUNT(DISTINCT client) when a client can appear many times.`
* `-- Payment lateness is DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT.`

### 2.3 Repair prompt

Appended to the original user prompt (so the schema and examples stay in
context) after a validation failure:

```
The SQL you wrote was rejected before it ran.

Your query:
{sql}

Rejected because:
{errors}

Write a corrected single SELECT query using only the columns in the schema
above. Output raw SQL only. If the question genuinely cannot be answered from
the available columns, reply with exactly:
CANNOT_ANSWER: <short reason>

SQL:
```

The errors are the validator's own, phrased for two readers at once — the
person watching the UI and the model reading them back. That is why
`_unknown_column` names the offending identifier *and* suggests a real one:

> `Column 'AMT_INCOME' does not exist on table 'application_train'. Did you mean 'AMT_INCOME_TOTAL'?`

`test_hallucinated_column_triggers_exactly_one_repair` asserts the specific
identifier reaches the repair prompt, not a generic nudge.

### 2.4 Summary prompt

```
You explain query results to bank staff who do not read SQL.

RULES
1. Use only the numbers in the results below. Never introduce a figure that
is not there, and never estimate, extrapolate or recall a value from memory.
2. Lead with the direct answer in one sentence.
3. Add at most two more sentences of context - the comparison that makes the
number meaningful, or the caveat that stops it being misread.
4. Round sensibly: rates as percentages to one decimal, money to whole units
with thousands separators.
5. If the results are empty, say plainly that no matching records were found.
Do not speculate about why.
6. The line above the results states whether they were truncated. Only say
the answer covers a partial sample if that line says so - a small row count
(even 1 row, for a single aggregate like an overall average) is a complete
answer, not a sign of missing data.
7. Plain language. No SQL, no column names in capitals, no bullet points.
```

user:

```
Question: {question}

Results: {row_count} row(s) - {completeness}
{results}

Answer:
```

**Rule 6 is a bug fix, written into the prompt.** `docs/PROGRESS.md` bug #4:
the summariser was originally shown the generated SQL alongside the rows. The
validator injects `LIMIT 500` into every query that lacks one, so a
single-row aggregate arrived at the summariser as one row of data plus a
`LIMIT 500` clause — and the model reported the portfolio-wide default rate as
"truncated to a 500-row sample". It had reasoned from the clause rather than
the row count, which is exactly the kind of plausible, confident, wrong
inference this whole document is about.

The fix was two-part and both parts matter:

1. **The SQL was removed from the summary prompt entirely.** The summariser's
   job is to describe rows. Giving it the query gave it something to
   speculate from. (Verified: `SUMMARY_USER_PROMPT` has no `{sql}` slot, and
   `test_summariser_sees_rows_but_not_the_database` asserts the prompt
   contains the markdown table and nothing else.)
2. **Completeness became one explicit sentence** the model is told to trust
   over anything else. `_summarise` in `nl_to_sql.py` computes it from
   `result.truncated` and `result.row_count`, never from the SQL text:

   * `"this is the complete result, not a sample"`
   * `"truncated to the first N (more rows matched the query but were cut off)"`
   * `"the query returned N, only the first 50 are shown below"`

Rule 1 is enforced downstream by nothing at all — there is no validator for
prose. It holds because of what the summariser *cannot see*: it never receives
the schema, the SQL, or a database handle. It has no source for a number other
than the markdown table in its prompt.

### 2.5 Explanation prompt (SHAP narrative)

`EXPLANATION_SYSTEM_PROMPT` / `EXPLANATION_USER_PROMPT`, used by
`src/ml/explain.py::narrate`. Grounded the same way: the model sees six
computed SHAP contributions and three numbers, and nothing else.

```
Risk score: {probability:.1%} probability of default
Risk band: {band}
Portfolio average: {base_rate:.1%}

Factors that moved this score:
{factors}

Explanation:
```

Rule 4 ("Describe associations, not causes") and rule 5 ("Never state or imply
that the decision is final, and never give the applicant advice about how to
change the outcome") are there because this text can be read by an applicant.
SHAP explains the model's raw log-odds, not the calibrated probability the user
sees, so a contribution can honestly be described as directional but never as
a slice of the displayed percentage.

`{base_rate}` is a live bug scar. `docs/PROGRESS.md` bug #12: the route
originally passed `explanation.base_value` — SHAP's log-odds baseline,
measured at **-2.9605** on the shipped model — into the portfolio-average slot,
rendering `Portfolio average: -296.1%`. The correct value is
`RiskModel.base_rate` = **0.0807**. `narrate()` now takes it as its own
parameter with a docstring saying why it must never be `base_value`.

That bug was invisible for a while because its sibling (#11, an undersized
token budget) truncated the response before the model got far enough to use
the field. Two bugs, each hiding the other.

---

## 3. Token optimisation

### 3.1 The headline: 67% off the SQL prompt

Naive baseline = full schema card (every table, every column) + all 10
worked examples, which is what the prompt would look like written the obvious
way. Actual = what the pipeline sends. Both measured with `count_tokens` on
`gemini-3.7-flash`.

| Question | Naive | Actual | Saved |
|---|---:|---:|---:|
| What is the overall default rate? | 6,611 | 1,617 | 76% |
| How does the default rate vary by education level? | 6,614 | 1,619 | 76% |
| Do clients with overdue bureau credit default more often? | 6,614 | 2,625 | 60% |
| Which occupations have the highest average loan amount? | 6,613 | 2,285 | 65% |
| What is the default rate by age band? | 6,613 | 1,613 | 76% |
| How many clients had a previous application refused? | 6,613 | 2,772 | 58% |
| Are late installment payments associated with default? | 6,612 | 2,291 | 65% |
| Average loan size, cash vs revolving? | 6,617 | 2,263 | 66% |
| Who is the CEO of Home Credit? (out of scope) | 6,612 | 2,271 | 66% |
| **Total, 9 questions** | **59,519** | **19,356** | **67%** |

The `Actual` column matches the live API's reported `prompt_token_count` for
the same nine questions **exactly** (1617, 1619, 2625, 2285, 1613, 2772, 2291,
2263, 2271 — see §5). `count_tokens` and the live meter agree, so the rest of
this section can use `count_tokens` without a live call per measurement.

The rest of §3 breaks that 67% into its parts.

### 3.2 Two-tier schema card + per-question pruning

`render_schema_card` never sends the whole schema. Tier 1 (name, row count,
one-line purpose, join keys) goes for all 7 tables always — the model needs all
of it to plan a join, and it is small. Tier 2 (full column detail) goes only for
the 3 highest-scoring tables, capped at 28 columns each.

**Real database** (`get_catalog()` against `data/credit_risk.duckdb`, 218
chatbot-visible columns: application_train 122, bureau 17, bureau_balance 3,
previous_application 37, installments_payments 8, credit_card_balance 23,
pos_cash_balance 8):

| Question | Cols shown | Est. (chars/4) | Measured (API) | Saved vs full |
|---|---:|---:|---:|---:|
| *(full card, no question)* | 218 | 5,333 | **5,322** | — |
| What is the overall default rate? | 28 | 1,006 | 1,126 | **79%** |
| How does default rate differ by gender? | 28 | 1,007 | 1,125 | **79%** |
| What is the default rate by age band? | 28 | 995 | 1,120 | **79%** |
| Are late installment payments linked to default? | 53 | 1,530 | 1,634 | **69%** |
| Which occupations have the highest average loan amount? | 47 | 1,651 | 1,729 | **68%** |
| Average loan size, cash vs revolving? | 52 | 1,672 | 1,743 | **67%** |
| Do clients with overdue bureau credit default more? | 68 | 1,867 | 1,981 | **63%** |
| How many previous applications were refused? | 79 | 2,171 | 2,208 | **59%** |

**Test fixture** (`tests/conftest.py::_TABLES`, 182 columns — a smaller,
synthetic stand-in used so the pruning tests run offline). Estimated only; the
fixture is never sent to the API:

| Question | Cols | Est. tokens | Saved vs full (1,424) |
|---|---:|---:|---:|
| *(full card)* | 182 | 1,424 | — |
| What is the overall default rate? | 28 | 656 | 54% |
| How does default rate differ by gender? | 28 | 656 | 54% |
| Which education level has the highest default rate? | 28 | 656 | 54% |
| Are late installment payments linked to default? | 53 | 754 | 47% |
| Do clients with overdue bureau credit default more? | 57 | 782 | 45% |
| How many previous applications were refused? | 59 | 785 | 45% |

The fixture's savings are smaller because its tables are narrower — the tier-1
overhead is a larger fraction of a smaller card. **The real database is what
ships, and it saves 59-79%.** The fixture numbers are here because
`tests/test_schema_card.py` asserts against them (`test_pruning_materially_
reduces_size` requires < 70% of full), and a reviewer should be able to tell
the two apart.

Cost is not the only argument. The accuracy argument is that the column a
question is about should not be buried among 217 irrelevant ones.
`test_pruned_card_keeps_the_answering_columns` parametrises 7 question/column
pairs and asserts the answering column survives pruning every time;
`test_join_keys_always_survive_pruning` and `ALWAYS_KEEP` protect `SK_ID_CURR`,
`SK_ID_BUREAU`, `SK_ID_PREV` and `TARGET` unconditionally. A card that is small
but missing the answer is worse than no pruning at all.

Selection is lexical and deterministic (`DOMAIN_SYNONYMS`, `TABLE_KEYWORDS`),
not embedding-based — no model call, no vector store, no warm-up, and an
evaluator re-running a question gets the same prompt.

### 3.3 The domain prior on column selection

Without a prior, `select_columns` scores mostly on question-word overlap. A
question that matches few column names spends its 28-column budget on whatever
sorts first, and this schema sorts alphabetically into
`AMT_REQ_CREDIT_BUREAU_*`, `APARTMENTS_AVG`, `BASEMENTAREA_MEDI`,
`COMMONAREA_MODE`.

`_prior_score` fixes that with two lists:

* `_HIGH_PRIOR_COLUMNS` (41 entries) → **+12.0**. The columns a credit analyst
  would actually name: `TARGET`, `CODE_GENDER`, `DAYS_BIRTH`, `AMT_CREDIT`,
  `NAME_EDUCATION_TYPE`, `EXT_SOURCE_1..3`, `CREDIT_ACTIVE`,
  `NAME_CONTRACT_STATUS`, and so on.
* `_LOW_PRIOR_PATTERNS` → **-10.0**. The wide low-signal blocks:
  `.*_(AVG|MODE|MEDI)$` (42 building statistics), `FLAG_DOCUMENT_\d+` (20
  near-constant document flags), `AMT_REQ_CREDIT_BUREAU_.*` (6 enquiry
  counters), the address-mismatch flags, the phone flags, the social-circle
  counters.

The demotions are evidence-based, not taste. `docs/PROGRESS.md` records the
real measurement: the building statistics are **48.8-69.9% missing** on the
real data (the plan had estimated 50-70%), and none of them appear in a
question anyone asks in plain English. Spending 42 of a 28-column budget on
`COMMONAREA_MEDI` is spending it on a column that is two-thirds null.

`ALWAYS_KEEP` sits above the prior at **+1000.0** — a join key must never lose
a tiebreak, because dropping one silently breaks every multi-table question.

`test_low_signal_columns_lose_to_relevant_ones` asserts `COMMONAREA_MEDI` and
`APARTMENTS_AVG` are absent from the `application_train` block while
`NAME_EDUCATION_TYPE` is present.

### 3.4 Similarity-selected few-shot examples

10 worked examples exist; 3 are sent. Selection scores tag hits (×3) and shared
question vocabulary (×1), deterministically. Measured with `count_tokens`:

| | Measured tokens |
|---|---:|
| All 10 examples | **1,024** |
| Selected 3, best case (`What is the overall default rate?`) | 226 |
| Selected 3, worst case (`Are late installment payments…`) | 391 |
| Selected 3, mean over the 9 measured questions | **~284** |

**~72% saved**, and the examples that survive are the ones nearest the
question, which is the better prompt as well as the cheaper one. The
`FewShotExample` docstring's rule-of-thumb ("roughly 700 tokens" for all,
"about 250" for three) was written from the chars/4 estimate; the measured
figures are 1,024 and ~284. The direction and the ratio hold; the absolute
numbers were ~25% low.

Selection is not always three — `select_examples` returns only examples that
scored above zero. `"How many previous applications were refused?"` selects
just two (130 estimated tokens); the slightly different phrasing
`"How many clients had a previous application refused?"` selects three (297
measured). When nothing scores, it falls back to the first three, which are the
general rate and group-by patterns most questions resemble anyway.

### 3.5 Markdown, not JSON, for result rows

`QueryResult.to_markdown` renders column names once in a header row rather than
repeating them on every record. Measured on the actual result sets from §5
(chars/4, since these are user-prompt fragments):

| Result shape | Markdown | JSON, same formatting | Saved | JSON, raw values | Saved |
|---|---:|---:|---:|---:|---:|
| 10 rows × 3 cols (occupation) | 105 | 241 | 56% | 242 | 57% |
| 5 rows × 3 cols (education) | 62 | 123 | 50% | 134 | 54% |
| 2 rows × 4 cols (bureau overdue) | 44 | 60 | 27% | 63 | 30% |
| 1 row × 3 cols (single aggregate) | 20 | 19 | −5% | 20 | 0% |
| **Total** | **231** | **443** | **48%** | **459** | **50%** |

The saving is real but shape-dependent, and the code's docstring ("roughly 30%
fewer tokens") is the conservative end of the measured range rather than the
average. The header cost is amortised over rows, so it pays more the more rows
there are and **nothing at all on a single-row aggregate** — where markdown is
in fact 1 token worse. The rounding in `_format_cell` (a rate as `0.0807`
rather than `0.08072881945686496`) accounts for the small gap between the two
JSON columns, and applies to markdown too.

### 3.6 `MAX_SUMMARY_ROWS` truncation

`max_summary_rows = 50` (`src/utils/config.py`). Up to 500 rows can be
returned to the UI (`max_sql_rows = 500`), but only the first 50 reach the
summariser. The full result is already on screen; paying to send 500 rows in
order to have three sentences written about them is waste.

The truncation is never silent. `to_markdown` appends
`... N more row(s) not shown`, and `_summarise` states the completeness
explicitly in the line above the table — the mechanism rule 6 depends on.

None of the nine measured questions returned more than 10 rows, so this cap
did not fire in the transcripts below. It is a bound, not a routine saving.

### 3.7 The two-model split

`MODEL_FALLBACKS` in `src/llm/gemini.py`:

| Role | Model | Job | Output cap |
|---|---|---|---|
| `sql` | `gemini-3.7-flash` | schema reasoning, joins, window functions, refusal judgement | 600 |
| `summary` | `gemini-3.5-flash-lite` | describe a table of numbers in three sentences | 400 |

Measured across the 9 live questions in §5 (13 calls total: 9 SQL + 1 repair
+ 8 summary — the refusal makes no summary call):

| | SQL model | Summary model |
|---|---:|---:|
| Calls | 10 | 8 |
| Prompt tokens, total | 21,122 | 2,740 |
| Prompt tokens, mean/call | 2,112 | 343 |
| Output tokens, total | 741 | 509 |
| Thought tokens, total | 100 | 0 |
| Total tokens | 21,963 | 3,249 |
| **Share of all tokens** | **87%** | **13%** |

The summary model handles 44% of the calls for 13% of the tokens, because its
prompt is a 215-token system message plus a small markdown table — never the
schema. Routing it to the cheaper model is worth doing; routing SQL generation
there would not be, since that is where the schema reasoning happens.

Model resolution is **reactive, not listed**. `docs/PROGRESS.md` bug #2:
`client.models.list()` included `gemini-2.5-flash`, but generating with it
404'd as "no longer available to new users". `list()` is now only a cheap first
filter; the first successful `generate_content` is what confirms a model, and a
404/permission error strikes that id and advances to the next candidate without
a retry, while a 429/5xx retries the same model with exponential backoff.

### 3.8 The honest thinking-token finding

Gemini 3.x reasons before answering by default and bills the reasoning as
output tokens. The code sets `thinking_level="low"` on every call, which is the
minimum available. Three things I measured, one of which corrects a claim in
this repo's own docstrings.

**(a) There is no "off", and `thinking_budget=0` behaves differently per
model.** The repo's docstrings state flatly that `thinking_budget=0` is
"rejected with `400 INVALID_ARGUMENT`". Re-measured today on a trivial prompt
(`"Reply with the single word: ok"`, temp 0, seed 42):

| Model | Config | Result | Thought tokens |
|---|---|---|---:|
| `gemini-3.7-flash` | `thinking_budget=0` | **accepted** | 93 |
| `gemini-3.7-flash` | `thinking_level="low"` | accepted | 101 |
| `gemini-3.5-flash-lite` | `thinking_budget=0` | **400 INVALID_ARGUMENT** | — |
| `gemini-3.5-flash-lite` | `thinking_level="low"` | accepted | 0 |

So the precise statement is: `thinking_budget=0` is rejected outright by
`gemini-3.5-flash-lite`, and *silently ignored* by `gemini-3.7-flash` — it
still spent 93 thought tokens to answer "ok". Neither model offers a real off
switch. `thinking_level="low"` is the correct setting, and the earlier
docstring was right about the conclusion and imprecise about which model
produced the 400.

**(b) The cost is not a fixed overhead, and it is not stable across runs.**
`src/ml/explain.py` documents ~1,040 thought tokens on the SHAP-narrative
prompt during an earlier session — enough that a 300-token budget left 10
tokens for the sentence and truncated it mid-word with
`finish_reason=MAX_TOKENS` (`docs/PROGRESS.md` bug #11), and even a 700-token
budget was consumed entirely.

I re-ran that exact prompt path today on three real applicants:

| SK_ID_CURR | Prompt | Thoughts | Output | Total | Model |
|---|---:|---:|---:|---:|---|
| 100002 (High band, p=0.3472) | 399 | **0** | 159 | 558 | `gemini-3.5-flash-lite` |
| 100003 (Low band, p=0.0144) | 392 | **0** | 106 | 498 | `gemini-3.5-flash-lite` |
| 100004 (Low band, p=0.0279) | 386 | **0** | 101 | 487 | `gemini-3.5-flash-lite` |

Zero thought tokens on all three, on the same prompt shape that previously cost
~1,040. That is the finding, and it is the argument for the current
`NARRATIVE_MAX_OUTPUT_TOKENS = 1500`: thinking cost on a given prompt shape is
**not reproducible**, so the budget has to carry headroom for the worst case
observed, not the median. Sizing to today's measurement would reintroduce bug
#11 the next time the model decides to think.

**(c) In the chatbot path, thinking was near-zero.** Across the 13 live calls
in §5, thought tokens totalled **100** — all of it on one SQL call
(`"How many clients had a previous application refused?"`), zero on the other
nine SQL calls and zero on all eight summary calls. That is 0.4% of the 25,212
tokens spent. On the chatbot's prompts, thinking is currently cheap; on the
SHAP prompt it has been expensive. Both are the same `thinking_level="low"`.

---

## 4. Hallucination control — the layered defence

Three layers, and the ordering matters: **the prompt is not a security
control.** A language model can be talked into emitting anything. Layer 1
reduces how often a bad query is generated; layers 2 and 3 are what make it
impossible for one to matter.

```
  Layer 1  PROMPT      schema grounding + explicit refusal   — reduces frequency
  Layer 2  VALIDATOR   sqlglot AST checks vs live catalog    — stops it running
  Layer 3  CONNECTION  read-only + enable_external_access=false — stops it mattering
```

### 4.1 Schema grounding

`src/talk_to_data/catalog.py` reads `information_schema` from the live DuckDB
file at runtime. Nothing is hard-coded. Two consequences fall out for free:

* The schema card describes the database that **exists**, not one someone
  documented six months ago.
* The validator rejects a hallucinated column against the **same source of
  truth** the model was shown, so the two can never drift apart.

`DATASET_NOTES` travels with every card and states the six quirks a model
cannot infer from column names and gets wrong every time otherwise: `TARGET=1`
means default; `DAYS_*` are negative offsets; `DAYS_EMPLOYED = 365243` is the
"not employed" sentinel (**18.0%** of rows on the real data, per
`docs/PROGRESS.md`); `CODE_GENDER` has an `XNA` category; only
`application_train` has `TARGET`; child tables need aggregating before joining.
`test_dataset_quirks_are_always_stated` asserts `365243` and `TARGET = 1` are
in every card.

### 4.2 `CANNOT_ANSWER` — refusal as a first-class output

Rule 8 of the SQL system prompt gives the model an explicit, cheap, legitimate
way to say no. This is not politeness engineering. From
`prompt_templates.py`'s own docstring:

> Without one, a language model asked an unanswerable question will invent a
> plausible query against invented columns, because producing *something* is
> the path of least resistance.

The refusal token is checked literally (`REFUSAL_PREFIX = "CANNOT_ANSWER:"`,
matched case-insensitively on a stripped prefix) and short-circuits the whole
pipeline: no validation, no database, no summary call. One API call, and the
reason is surfaced to the user as the answer.

Measured, live, on `"Who is the CEO of Home Credit?"` — see §5.7. The model
refused correctly. It is worth noting *what it did not do*: it did not invent
an `EMPLOYEES` table, and it did not answer from parametric memory. Both were
available to it.

`test_refusal_short_circuits_without_querying` asserts exactly one model call
and `answer.sql is None`.

### 4.3 AST validation, not regex

`src/talk_to_data/sql_validator.py` parses with `sqlglot` in the DuckDB dialect
and walks the tree. A regex over SQL text loses to:

* **Comments** — `SELECT count(*) /* harmless */ FROM t; -- \nDROP TABLE bureau`
  is a test case (`test_comment_disguised_injection_is_refused`). A line-comment
  guard that strips to end-of-line, or one that does not, gets a different
  answer than the parser does.
* **String literals** — a regex cannot distinguish a column *reference* from a
  column *name inside a string*. `WHERE NAME_CONTRACT_STATUS = 'Refused'`
  contains the word `Refused`; a naive column extractor sees an identifier.
* **Whitespace and casing** — `READ_CSV`, `read_csv_auto`, `Read_Csv (` with a
  newline before the paren are all the same function to the parser and three
  different strings to a pattern.

The parser resolves each of these structurally. It also gives the validator
something a regex never could: **scope**. `build_scope` tells it, for each part
of the query, which sources are visible and whether each is a physical table or
a derived result, so a column can be attributed to the right table through
CTEs, subqueries and aliases.

### 4.4 The specific checks

| # | Check | Why it is separate from the others |
|---|---|---|
| 1 | Parses as SQL in the DuckDB dialect | An unparseable string cannot be reasoned about at all. Reported as a clear message, not a crash. |
| 2 | Exactly one statement | `sqlglot.parse` returns a list. `SELECT 1; DROP TABLE bureau` is length 2 and refused on count alone, before any node inspection. |
| 3 | Statement type ∈ {`Select`, `Union`, `Subquery`} | The positive form of the check. Anything else is refused by name in the error. |
| 4 | No forbidden node **anywhere in the tree** | 18 node types: `Insert`, `Update`, `Delete`, `Drop`, `Create`, `Alter`, `TruncateTable`, `Merge`, `Copy`, `Attach`, `Detach`, `Pragma`, `Set`, `Command`, `Grant`, `Use`, `Transaction`, `Commit`, `Rollback`, `AlterSet`. Walked over the whole tree, not just the root, so a write nested in a subquery cannot hide. |
| 5 | **No filesystem or network function, by name** | The check that matters most. See below. |
| 6 | Every table exists and is in the chatbot allowlist | With a "did you mean" via `_closest`. `CHATBOT_TABLES` excludes `column_descriptions` — it is metadata about the schema, and exposing it invites questions about the glossary rather than the portfolio. |
| 7 | Every column exists on the table it is attributed to | Resolved through CTEs, subqueries and aliases via `build_scope`. With a `suggest_column` "did you mean". |
| 8 | A row limit is present | Injected if absent, clamped if oversized (`max_sql_rows = 500`), respected if smaller. |

**On check 5 specifically.** `SELECT * FROM read_csv('/etc/passwd')` is a
perfectly ordinary `exp.Select` node reading from an `exp.Anonymous` function.
Checks 2, 3 and 4 all pass it. A statement-type check cannot catch it, because
there is nothing wrong with the statement type — the danger is in the function
name. `FORBIDDEN_FUNCTION_PATTERNS` matches on a normalised name so `READ_CSV`,
`read_csv_auto` and `read_parquet` all hit `read_.*`, and `parquet_scan`,
`postgres_scan`, `sqlite_scan` and `iceberg_scan` all hit `.*_scan`. Also
blocked: `glob`, `sniff_csv`, `getenv`, `install`, `load`, and anything
matching `.*_secret.*`.

`tests/test_sql_validator.py::test_filesystem_functions_are_refused` covers
seven concrete escapes: `read_csv`, `read_csv_auto`, `read_parquet`,
`read_json_auto`, `glob`, `parquet_scan`, `sqlite_scan`.

**Errors are returned, not raised.** `validate_sql` returns a
`ValidationResult` rather than throwing, because the caller feeds `errors`
straight back to the model for the repair attempt — an exception would make
that control flow read as an error path rather than the expected one.

### 4.5 The DuckDB connection: read-only is not enough

`get_readonly_connection` in `src/data/database.py` opens with **both**
`read_only=True` and `config={"enable_external_access": "false"}`. The second
flag is the one people leave out. Measured on DuckDB 1.5.5 and recorded in that
module's docstring:

| Operation | read-only alone | read-only + `enable_external_access=false` |
|---|---|---|
| `read_csv` arbitrary file | **ALLOW** | `PermissionException` |
| `glob` filesystem | **ALLOW** | `PermissionException` |
| `COPY … TO` file | **ALLOW** | `PermissionException` |
| `ATTACH` another database | **ALLOW** | `PermissionException` |

Read-only protects the *database*. It does not protect the *filesystem*. A
plain read-only connection will happily read `/etc/passwd` and write a CSV of
your training data to `/tmp`.

`enable_external_access` is a **locked** setting: once the database is open it
cannot be turned back on from inside a query, so a generated query cannot
`SET` its way to escalated privileges. That property is what makes this a
boundary rather than a default.

Two more operational bounds on the same connection: a 30-second watchdog that
calls `cursor.interrupt()` (DuckDB has no statement timeout; verified against
1.5.5 that the cursor stays usable afterwards, so a slow question does not
poison the connection for the next one), and `fetchmany(limit + 1)` so
truncation is *detected* by fetching one row past the limit rather than by
materialising a result set that would only be thrown away.

### 4.6 One repair, not a loop

A rejected query gets exactly one correction attempt, carrying the specific
validator error. A second failure is reported as a failure. From `_repair`'s
docstring:

> Looping until something validates would eventually produce a query that
> passes the checks while answering a different question, and would burn the
> user's quota getting there.

This is the important trade. A retry loop optimises for "produce output", and a
query that passes the validator is not the same thing as a query that answers
the question. Bounding it at one keeps the failure mode "I could not write a
valid query for that" — which is honest — rather than a confidently wrong
answer to a question nobody asked.
`test_repair_is_not_retried_a_second_time` asserts exactly two model calls and
`answer.sql is None` after two bad queries.

**The repair fired for real in the transcripts below**, and the cause is worth
being honest about — it was a validator false positive, not a model
hallucination. See §5.5.

### 4.7 The summariser never sees the database

`_summarise` builds its prompt from `result.to_markdown()` and three scalars.
No schema, no SQL, no connection. The model has no source for a number other
than the table in front of it. That is the whole mechanism behind rule 1 of the
summary prompt; there is no post-hoc check on generated prose, and there does
not need to be one if the model was never handed anything to invent from.

`test_summariser_sees_rows_but_not_the_database` asserts the summary prompt
contains the rendered table and the "Use only the numbers" rule.

### 4.8 The SHAP narrative is grounded the same way

`narrate()` receives six `FeatureContribution` objects — feature name, humanised
label, the applicant's actual value, the signed SHAP value — plus probability,
band and portfolio base rate. Nothing else. It has no database handle, no
feature list beyond the six, and no access to the model internals.

The prompt's rule 1 ("Never invent a reason, and never mention a factor that is
not in the list") is enforced by the same absence-of-material principle as the
chatbot's. In the three narratives measured in §3.8(b), every factor named
appears verbatim in the prompt's factor list.

### 4.9 What the test suite enforces

`PYTHONPATH=. .venv/bin/python -m pytest tests/test_sql_validator.py tests/test_schema_card.py tests/test_nl_to_sql.py tests/test_query_runner.py -q`
→ **73 passed in 1.14s**, fully offline, no API key.

| File | Tests | What it holds down |
|---|---:|---|
| `test_sql_validator.py` | **36** | The adversarial suite. See breakdown below. |
| `test_schema_card.py` | 18 | Pruning correctness, token reduction, the domain prior, join-key survival, dataset notes always present |
| `test_nl_to_sql.py` | 11 | Repair loop bounded at one, refusal short-circuits, schema is injected, usage recorded, degraded mode without a key |
| `test_query_runner.py` | 8 | Limits, timeout interrupt leaves the connection usable, markdown rendering |

The 36 validator tests break down as:

* **8 valid queries that must pass** — including a CTE with an alias, a derived
  table, a window-function decile, and a three-level join key. A validator that
  refuses everything is not a validator.
* **9 write/admin injections** — `DROP`, `DELETE`, `UPDATE`, `INSERT`,
  `ATTACH`, `PRAGMA`, `COPY … TO` (exfiltration), `CREATE TABLE AS`, and a
  **stacked statement** (`SELECT count(*) FROM application_train; DROP TABLE
  bureau`). Each asserts the offending keyword appears in the error message,
  so a test cannot pass by refusing for the wrong reason.
* **7 filesystem escapes** — listed in §4.4.
* **1 comment-disguised injection** — the case a regex guard would most likely
  miss.
* **5 hallucination cases** — unknown table with a suggestion; hallucinated
  column with a suggestion (`AMT_INCOME` → `AMT_INCOME_TOTAL`); hallucinated
  *qualified* column (`a.CUSTOMER_AGE`); a column that exists but **on the
  wrong table** (`a.STATUS`, which lives on `bureau_balance`); and an
  allowlist bypass (a real table excluded from `allowed_tables`).
* **6 normalisation cases** — limit injected, oversized limit clamped, smaller
  limit respected, markdown fence tolerated, unparseable SQL reported clearly,
  empty query refused.

`test_write_attempt_never_reaches_the_database` closes the loop end-to-end: it
feeds the pipeline a `DROP TABLE`, then queries the table and asserts all 1,200
fixture rows are still there.

---

## 5. Sample transcripts

Real captures from `TalkToData.ask()` against the live Gemini API and the real
307,511-row database, 2026-09-07. Prompt version `v1.0.0`. Token counts are the
API's own `usage_metadata`; SQL is shown as the validator rewrote it (which is
what actually executed).

Notation: `sql` = `gemini-3.7-flash`, `sum` = `gemini-3.5-flash-lite`.

### 5.1 Simple aggregate — 1,971 tokens

> **Q:** What is the overall default rate?

```sql
SELECT COUNT(*) AS applications,
       SUM(TARGET) AS defaults,
       AVG(TARGET) AS default_rate
FROM application_train
LIMIT 500
```

| applications | defaults | default_rate |
|---:|---:|---:|
| 307,511 | 24,825 | 0.0807 |

> **A:** The overall default rate is 8.1%. This is out of a total of 307,511
> applications, which resulted in 24,825 defaults.

| Call | Model | Prompt | Thoughts | Output | Total | ms |
|---|---|---:|---:|---:|---:|---:|
| SQL | `gemini-3.7-flash` | 1,617 | 0 | 30 | 1,647 | 1,733 |
| Summary | `gemini-3.5-flash-lite` | 285 | 0 | 39 | 324 | 1,504 |
| | | | | | **1,971** | **3,457** |

This is the transcript bug #4 was found on. One row, plus an injected
`LIMIT 500` — exactly the shape that used to produce "truncated to a 500-row
sample". With rule 6 and the SQL removed from the summary prompt, the
completeness line reads *"this is the complete result, not a sample"* and the
answer states the figure flatly. The 8.1% also matches the independently
measured portfolio rate of 8.073% in `docs/MODEL_CARD.md`.

### 5.2 Group-by segmentation — 2,103 tokens

> **Q:** How does the default rate vary by education level?

```sql
SELECT NAME_EDUCATION_TYPE, COUNT(*) AS applications, AVG(TARGET) AS default_rate
FROM application_train
GROUP BY NAME_EDUCATION_TYPE
ORDER BY default_rate DESC
LIMIT 500
```

| NAME_EDUCATION_TYPE | applications | default_rate |
|---|---:|---:|
| Lower secondary | 3,816 | 0.1093 |
| Secondary / secondary special | 218,391 | 0.0894 |
| Incomplete higher | 10,277 | 0.0848 |
| Higher education | 74,863 | 0.0536 |
| Academic degree | 164 | 0.0183 |

> **A:** Default rates vary significantly by education level, ranging from a
> high of 10.9% down to 1.8%. Customers with lower secondary education have the
> highest default rate at 10.9% across 3,816 applications, while those with an
> academic degree have the lowest default rate at 1.8% based on 164
> applications.

| Call | Model | Prompt | Thoughts | Output | Total | ms |
|---|---|---:|---:|---:|---:|---:|
| SQL | `gemini-3.7-flash` | 1,619 | 0 | 44 | 1,663 | 1,100 |
| Summary | `gemini-3.5-flash-lite` | 363 | 0 | 77 | 440 | 1,136 |
| | | | | | **2,103** | **2,247** |

Rule 6 of the SQL prompt (always return the denominator) is doing visible work:
the summariser carried `n=164` through to the answer, which is what stops
"academic degree defaults least" being read as a strong finding off 164 rows.

### 5.3 Child-table join with a CTE — 3,222 tokens

> **Q:** Do clients with overdue bureau credit default more often?

```sql
WITH bureau_summary AS (
  SELECT SK_ID_CURR,
         CASE WHEN MAX(CREDIT_DAY_OVERDUE) > 0
                OR MAX(COALESCE(AMT_CREDIT_SUM_OVERDUE, 0)) > 0
              THEN 'Has Overdue Bureau Credit'
              ELSE 'No Overdue Bureau Credit' END AS overdue_status
  FROM bureau GROUP BY SK_ID_CURR
)
SELECT b.overdue_status, COUNT(*) AS applications,
       SUM(a.TARGET) AS defaults, AVG(a.TARGET) AS default_rate
FROM application_train AS a
JOIN bureau_summary AS b ON a.SK_ID_CURR = b.SK_ID_CURR
GROUP BY b.overdue_status
ORDER BY default_rate DESC
LIMIT 500
```

| overdue_status | applications | defaults | default_rate |
|---|---:|---:|---:|
| Has Overdue Bureau Credit | 3,594 | 567 | 0.1578 |
| No Overdue Bureau Credit | 259,897 | 19,801 | 0.0762 |

> **A:** Clients with overdue bureau credit default at more than double the
> rate of those without. Specifically, the default rate for clients with
> overdue credit is 15.8%, compared to just 7.6% for clients with no overdue
> credit. This is despite a much smaller application pool of 3,594 clients with
> overdue credit versus 259,897 clients without.

| Call | Model | Prompt | Thoughts | Output | Total | ms |
|---|---|---:|---:|---:|---:|---:|
| SQL | `gemini-3.7-flash` | 2,625 | 0 | 190 | 2,815 | 1,601 |
| Summary | `gemini-3.5-flash-lite` | 328 | 0 | 79 | 407 | 1,156 |
| | | | | | **3,222** | **2,818** |

The model aggregated `bureau` per client *before* joining, which is the habit
the `-- Aggregate the child table first, then join, or rows double count.`
note in the selected few-shot examples exists to teach. Without it, a client
with three overdue bureau credits would be counted three times in the
denominator. The 1,981-token schema card for this question included both
`bureau` and `credit_card_balance` in tier 2 (the selector kept the extra table
rather than risk dropping the right one), which is why this prompt is 1,000
tokens above the simple-aggregate case.

### 5.4 Ranking with a model-chosen LIMIT — 2,929 tokens

> **Q:** Which occupations have the highest average loan amount?

```sql
SELECT OCCUPATION_TYPE, COUNT(*) AS applications, AVG(AMT_CREDIT) AS avg_loan_amount
FROM application_train
WHERE NOT OCCUPATION_TYPE IS NULL
GROUP BY OCCUPATION_TYPE
ORDER BY avg_loan_amount DESC
LIMIT 10
```

| OCCUPATION_TYPE | applications | avg_loan_amount |
|---|---:|---:|
| Managers | 21,371 | 775,091 |
| Accountants | 9,813 | 709,757 |
| HR staff | 563 | 684,983 |
| … | | |
| Medicine staff | 8,537 | 607,431 |

> **A:** Managers have the highest average loan amount at $775,091 across
> 21,371 applications. Accountants follow closely with an average of $709,757
> from 9,813 applications, while medicine staff have the lowest among the top
> ten at $607,431.

| Call | Model | Prompt | Thoughts | Output | Total | ms |
|---|---|---:|---:|---:|---:|---:|
| SQL | `gemini-3.7-flash` | 2,285 | 0 | 66 | 2,351 | 1,674 |
| Summary | `gemini-3.5-flash-lite` | 508 | 0 | 70 | 578 | 2,655 |
| | | | | | **2,929** | **2,655** |

The model wrote `LIMIT 10` itself (rule 7). `_apply_limit` respected it rather
than clamping to 500 — a smaller limit than the ceiling is the model's intent
and is preserved (`test_smaller_limit_is_respected`). Ten rows also make this
the largest summary prompt in the set at 508 tokens, which is where §3.5's
markdown-vs-JSON saving pays: the same rows as JSON would have been ~240
tokens instead of ~105.

### 5.5 A real repair — 4,001 tokens, 3 calls

> **Q:** What is the default rate by age band?

First attempt, rejected by the validator:

```
Column 'age_band_start' does not exist.
```

Repaired query, executed:

```sql
SELECT FLOOR(-DAYS_BIRTH / 365.25 / 10) * 10 AS age_band_start,
       COUNT(*) AS applications, AVG(TARGET) AS default_rate
FROM application_train
GROUP BY FLOOR(-DAYS_BIRTH / 365.25 / 10) * 10
ORDER BY age_band_start
LIMIT 500
```

| age_band_start | applications | default_rate |
|---:|---:|---:|
| 20 | 45,186 | 0.1144 |
| 30 | 82,331 | 0.0959 |
| 40 | 76,599 | 0.0764 |
| 50 | 68,094 | 0.0612 |
| 60 | 35,301 | 0.0492 |

> **A:** Default rates steadily decrease as applicant age increases, ranging
> from a high of 11.4% for the youngest group to a low of 4.9% for applicants
> aged 60 and over. Specifically, the rates are 11.4% for age 20, 9.6% for age
> 30, 7.6% for age 40, 6.1% for age 50, and 4.9% for age 60.

| Call | Model | Prompt | Thoughts | Output | Total | ms |
|---|---|---:|---:|---:|---:|---:|
| SQL (attempt 1) | `gemini-3.7-flash` | 1,613 | 0 | 69 | 1,682 | 1,067 |
| SQL (repair) | `gemini-3.7-flash` | 1,766 | 0 | 87 | 1,853 | 1,562 |
| Summary | `gemini-3.5-flash-lite` | 362 | 0 | 104 | 466 | 1,104 |
| | | | | | **4,001** | **3,742** |

**This was a validator false positive, and it should be reported as one.** The
first attempt used `GROUP BY age_band_start` — grouping by a `SELECT` alias,
which DuckDB accepts. I confirmed this directly: the rejected query runs
against the real database and returns the correct rows. `_check_columns`
resolves `age_band_start` through `build_scope`, finds `application_train` as
the only physical source, finds no such column on it, and refuses.

Notable because the rejected query is *verbatim the few-shot example* in
`FEW_SHOT_EXAMPLES` for this exact question — the prompt taught a pattern the
validator does not accept.

The cost of the false positive is one extra call (1,853 tokens, +1.5s) and the
right answer. That is the trade the layered design makes deliberately: the
validator is conservative, over-refusal is recovered by the bounded repair, and
the failure mode is an extra API call rather than a wrong answer. It is still a
real defect — either `_check_columns` should recognise `SELECT` aliases as
in-scope group keys, or the few-shot example should be rewritten to repeat the
expression. Recording it here rather than hiding it.

### 5.6 `COUNT(DISTINCT)` on a child table — 3,197 tokens

> **Q:** How many clients had a previous application refused?

```sql
SELECT COUNT(DISTINCT p.SK_ID_CURR) AS clients_with_refusal
FROM previous_application AS p
WHERE p.NAME_CONTRACT_STATUS = 'Refused'
LIMIT 500
```

| clients_with_refusal |
|---:|
| 118,277 |

> **A:** There were 118,277 clients who had a previous application refused.

| Call | Model | Prompt | Thoughts | Output | Total | ms |
|---|---|---:|---:|---:|---:|---:|
| SQL | `gemini-3.7-flash` | 2,772 | **100** | 41 | 2,913 | 2,601 |
| Summary | `gemini-3.5-flash-lite` | 266 | 0 | 18 | 284 | 1,237 |
| | | | | | **3,197** | **4,038** |

`COUNT(DISTINCT …)` rather than `COUNT(*)` — the `-- COUNT(DISTINCT client)
when a client can appear many times.` note doing its job on a 1.67M-row child
table where a client appears repeatedly. This is also the only call in the
whole set that spent thought tokens (100 of them; see §3.8c), and it carries
the largest schema card in the set at 2,208 tokens, because the selector
included `previous_application` *and* `credit_card_balance` in tier 2.

### 5.7 The out-of-scope refusal — 2,293 tokens, 1 call

> **Q:** Who is the CEO of Home Credit?

```
sql: None
refused: True
row_count: 0
```

> **A:** The schema contains loan application and repayment data, but no
> executive or corporate management information.

| Call | Model | Prompt | Thoughts | Output | Total | ms |
|---|---|---:|---:|---:|---:|---:|
| SQL | `gemini-3.7-flash` | 2,271 | 0 | 22 | 2,293 | 1,807 |
| | | | | | **2,293** | **1,807** |

Exactly one API call. `is_refusal()` matched, the pipeline short-circuited
before `validate_sql`, the database was never touched, and no summary call was
made. Note the 22 output tokens: refusing is the cheapest thing the model can
do, which is the point of making it cheap and legitimate.

The alternative behaviours this avoided are both plausible and both bad:
inventing an `EMPLOYEES` or `STAFF` table and writing confident SQL against it,
or answering the question from parametric memory as though the database had
said so. The refusal names the reason in the user's terms rather than emitting
an error.

### 5.8 Two more, in brief

**Contract-type comparison** — 2,700 tokens (SQL 2,263+51; summary 325+61):

```sql
SELECT NAME_CONTRACT_TYPE, COUNT(*) AS applications, AVG(AMT_CREDIT) AS avg_loan_size
FROM application_train GROUP BY NAME_CONTRACT_TYPE ORDER BY avg_loan_size DESC LIMIT 500
```

> **A:** The average loan size is $627,966 for cash loans compared to $324,018
> for revolving loans. Cash loans had a much larger volume with 278,232
> applications, while revolving loans had 29,279 applications.

**Installment lateness** — 2,796 tokens (SQL 2,291+141; summary 303+61), a CTE
over the 13.6M-row `installments_payments` table:

```sql
WITH lateness AS (
  SELECT SK_ID_CURR, AVG(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS avg_days_late
  FROM installments_payments WHERE NOT DAYS_ENTRY_PAYMENT IS NULL GROUP BY SK_ID_CURR
)
SELECT a.TARGET, COUNT(*) AS clients, AVG(l.avg_days_late) AS avg_days_late
FROM application_train AS a JOIN lateness AS l ON l.SK_ID_CURR = a.SK_ID_CURR
GROUP BY a.TARGET
```

> **A:** Clients who defaulted actually paid an average of 10.3 days early,
> while non-defaulting clients paid an average of 11.3 days early. Based on
> these 291,635 total clients, both groups generally pay ahead of their due
> dates rather than late.

Worth reading closely: the returned values are −10.3 and −11.3, and the
lateness convention is `DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT`, so negative
means *early*. The summariser read the sign correctly and reported the result
as "both groups pay early" — a null-ish finding — rather than asserting the
expected relationship. Rule 1 ("Use only the numbers in the results below")
covers the case where the numbers disagree with the question's premise.

---

## 6. What it costs

### 6.1 Per question, measured

All nine questions, live, from `Answer.total_tokens`:

| Question | Calls | Total tokens | ms |
|---|---:|---:|---:|
| What is the overall default rate? | 2 | 1,971 | 3,457 |
| How does the default rate vary by education level? | 2 | 2,103 | 2,247 |
| Who is the CEO of Home Credit? *(refusal)* | 1 | 2,293 | 1,807 |
| Average loan size, cash vs revolving? | 2 | 2,700 | 6,018 |
| Are late installment payments associated with default? | 2 | 2,796 | 4,489 |
| Which occupations have the highest average loan amount? | 2 | 2,929 | 2,655 |
| How many clients had a previous application refused? | 2 | 3,197 | 4,038 |
| Do clients with overdue bureau credit default more often? | 2 | 3,222 | 2,818 |
| What is the default rate by age band? *(repaired)* | 3 | 4,001 | 3,742 |

| | All 9 | Successful 8 (excl. refusal) |
|---|---:|---:|
| Mean | **2,801** | 2,865 |
| Median | 2,796 | 2,863 |
| Min | 1,971 | 1,971 |
| Max | 4,001 | 4,001 |
| Sum | 25,212 | 22,919 |

Against the naive baseline of §3.1 (full schema + all 10 examples on every SQL
call), the same nine questions would have spent roughly **65,400 tokens**
instead of 25,212 — the 59,519 naive SQL-prompt tokens plus the same output,
thought and summary tokens actually measured. **A 61% reduction end to end.**

### 6.2 Where the tokens go

| | Tokens | Share |
|---|---:|---:|
| SQL prompt (schema card + examples + rules + question) | 21,122 | 84% |
| SQL output (the query itself) | 741 | 3% |
| Thought tokens (all calls) | 100 | 0.4% |
| Summary prompt (rules + rows) | 2,740 | 11% |
| Summary output (the answer) | 509 | 2% |
| **Total, 9 questions** | **25,212** | |

The input side is 95% of the bill, and the schema card is most of the input.
That is why §3.2 is the optimisation that matters and why the thinking-token
question, though interesting, is a rounding error in this workload — 0.4% of
the total.

Latency, measured: **1.8-6.0 s** end to end per question, median 3.5 s. Roughly
1-3 s per model call plus 4-136 ms of DuckDB execution — the database is not
the bottleneck; the API round trips are.

### 6.3 Money

**No cost figures are given here, because I cannot verify current Gemini
pricing from inside this repository and will not guess at it.** The measured
inputs a reviewer needs to price it themselves:

| Meter | Value |
|---|---|
| Model, SQL role | `gemini-3.7-flash` |
| Model, summary role | `gemini-3.5-flash-lite` |
| Input tokens per question, mean | 2,651 (2,347 SQL + 304 summary) |
| Output tokens per question, mean | 139 (82 SQL + 57 summary) |
| Thought tokens per question, mean | 11 (billed as output) |
| Cached tokens | 0 — no prompt caching is configured |
| Total tokens per question, mean | 2,801 |

Two notes for whoever does price it. **Thought tokens bill as output**, so the
output-side figure to use is 139 + 11 = 150 per question on this workload —
but §3.8(b) shows that number is not stable across prompt shapes, and the SHAP
narrative has been observed at ~1,040 thought tokens on a single call.
**Prompt caching is not configured**, and the SQL system prompt's fixed
scaffolding (253 tokens of rules) plus the tier-1 table block repeat on every
call; that is the obvious next optimisation and it is not implemented.

For what it is worth on the free tier this was measured against: the binding
constraint was not cost but the **5 requests/minute per model** quota, which
`429 RESOURCE_EXHAUSTED`ed four of the first nine questions until they were
paced. The retry/backoff logic in `GeminiClient._generate_with_retry` treats
that as transient and retries the same model, which is correct behaviour and
still not enough at 3 attempts with 1s/2s backoff against a 30-second quota
window.

---

## 7. Reproducing these numbers

```bash
# Schema-card pruning, offline (test fixture, 182 columns)
PYTHONPATH=.:tests .venv/bin/python -c "
from conftest import _TABLES
from src.talk_to_data.catalog import Catalog, ColumnInfo, TableInfo, TABLE_DESCRIPTIONS
from src.talk_to_data.schema_card import render_schema_card
cat = Catalog(tables={n: TableInfo(name=n, row_count=r, description=TABLE_DESCRIPTIONS.get(n,''),
    columns={c: ColumnInfo(name=c, data_type=t) for c,t in cols}) for n,(r,cols) in _TABLES.items()})
full = render_schema_card(cat, None)
print(full.columns_total, 'cols,', full.approx_tokens, 'est. tokens')
for q in ['What is the overall default rate?', 'How many previous applications were refused?']:
    c = render_schema_card(cat, q); print(f'{c.approx_tokens} tokens  {1-c.approx_tokens/full.approx_tokens:.0%} saved  {q}')"

# Schema-card pruning, real database (218 columns) — swap in get_catalog()
PYTHONPATH=. .venv/bin/python -c "
from src.talk_to_data.catalog import get_catalog
from src.talk_to_data.schema_card import render_schema_card
print(render_schema_card(get_catalog(), None).approx_tokens)"

# Few-shot selection
PYTHONPATH=. .venv/bin/python -c "
from src.talk_to_data.prompt_templates import FEW_SHOT_EXAMPLES, select_examples, render_examples
print('all:', len(render_examples(FEW_SHOT_EXAMPLES))//4, 'est. tokens')
for q in ['What is the default rate by age band?', 'How many clients had a previous application refused?']:
    print(q, '->', len(render_examples(select_examples(q)))//4)"

# Live end-to-end, with the per-call usage dicts (needs GOOGLE_API_KEY)
set -a; source .env; set +a
PYTHONPATH=. .venv/bin/python -c "
from src.talk_to_data.nl_to_sql import get_talk_to_data
a = get_talk_to_data().ask('What is the overall default rate?')
print(a.sql); print(a.answer)
[print(u) for u in a.usage]; print('total:', a.total_tokens, 'ms:', a.duration_ms)"

# The guardrail suite (offline, no key)
PYTHONPATH=. .venv/bin/python -m pytest tests/test_sql_validator.py \
  tests/test_schema_card.py tests/test_nl_to_sql.py tests/test_query_runner.py -q
```

One caveat if you reproduce the live measurements: the read-only DuckDB
connection cannot be opened while `uvicorn` holds a handle on the same file.
Stop the server first, or point `DUCKDB_PATH` at a copy.
