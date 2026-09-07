"""Versioned prompt templates for the talk-to-data system.

Prompts are treated as source code: they live here, they are versioned, and
changing one is a reviewable diff rather than an untracked edit in a notebook.
`PROMPT_VERSION` is recorded on every answer the API returns, so a transcript
in the documentation can always be traced back to the prompt that produced it.

Three design choices carry most of the weight:

**The schema is injected, never assumed.** The model is told exactly which
tables and columns exist, rendered by `schema_card` for this specific question.
It is instructed to use nothing else. This is the first line of defence against
hallucinated columns - the validator is the second, and it is the one that
actually holds.

**Refusal is a first-class output.** The model is given an explicit way to say
"this cannot be answered from this data" (`CANNOT_ANSWER:`). Without one, a
language model asked an unanswerable question will invent a plausible query
against invented columns, because producing *something* is the path of least
resistance. Making refusal cheap and legitimate is what stops that.

**Answers are grounded in returned rows only.** The summarising prompt sees the
query results and is forbidden from introducing any number not present in them.
It never sees the database, so it cannot invent a figure and have it look
authoritative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Bump on any change to the wording below. Recorded with every answer.
PROMPT_VERSION = "v1.0.0"

#: The exact token the model must emit when a question cannot be answered.
#: Checked literally, so it must not appear in ordinary prose.
REFUSAL_PREFIX = "CANNOT_ANSWER:"


@dataclass(frozen=True)
class FewShotExample:
    """One worked question -> SQL pair.

    `tags` drive selection. Sending all of these on every request would cost
    roughly 700 tokens; sending the three nearest costs about 250 and, because
    the examples are closer to the question, works better.
    """

    question: str
    sql: str
    tags: tuple[str, ...]
    note: str = ""


#: The query patterns the system is expected to handle. Each one also teaches a
#: dataset-specific habit: how to express a rate, how to convert DAYS_*, how to
#: aggregate a child table before joining so rows are not double counted.
FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        question="What is the overall default rate?",
        sql=(
            "SELECT count(*) AS applications,\n"
            "       sum(TARGET) AS defaults,\n"
            "       avg(TARGET) AS default_rate\n"
            "FROM application_train"
        ),
        tags=("rate", "aggregate", "overall", "default"),
        note="A rate is avg(TARGET); always return the denominator with it.",
    ),
    FewShotExample(
        question="How does the default rate vary by education level?",
        sql=(
            "SELECT NAME_EDUCATION_TYPE,\n"
            "       count(*) AS applications,\n"
            "       avg(TARGET) AS default_rate\n"
            "FROM application_train\n"
            "GROUP BY NAME_EDUCATION_TYPE\n"
            "ORDER BY default_rate DESC"
        ),
        tags=("rate", "group", "segment", "education", "category"),
    ),
    FewShotExample(
        question="What is the default rate by age band?",
        sql=(
            "SELECT floor(-DAYS_BIRTH / 365.25 / 10) * 10 AS age_band_start,\n"
            "       count(*) AS applications,\n"
            "       avg(TARGET) AS default_rate\n"
            "FROM application_train\n"
            "GROUP BY age_band_start\n"
            "ORDER BY age_band_start"
        ),
        tags=("age", "band", "bucket", "days", "distribution"),
        note="DAYS_BIRTH is negative; negate before converting to years.",
    ),
    FewShotExample(
        question="Do applicants with a longer employment history default less?",
        sql=(
            "SELECT CASE\n"
            "         WHEN -DAYS_EMPLOYED / 365.25 < 2 THEN 'under 2 years'\n"
            "         WHEN -DAYS_EMPLOYED / 365.25 < 5 THEN '2-5 years'\n"
            "         ELSE 'over 5 years'\n"
            "       END AS employment_band,\n"
            "       count(*) AS applications,\n"
            "       avg(TARGET) AS default_rate\n"
            "FROM application_train\n"
            "WHERE DAYS_EMPLOYED != 365243\n"
            "GROUP BY employment_band\n"
            "ORDER BY default_rate DESC"
        ),
        tags=("employment", "tenure", "band", "job", "years"),
        note="365243 is the 'not employed' sentinel and must be excluded.",
    ),
    FewShotExample(
        question="Do clients with more prior bureau credits default more often?",
        sql=(
            "WITH per_client AS (\n"
            "  SELECT SK_ID_CURR, count(*) AS bureau_credits\n"
            "  FROM bureau\n"
            "  GROUP BY SK_ID_CURR\n"
            ")\n"
            "SELECT least(p.bureau_credits, 10) AS bureau_credits,\n"
            "       count(*) AS applications,\n"
            "       avg(a.TARGET) AS default_rate\n"
            "FROM application_train a\n"
            "JOIN per_client p ON p.SK_ID_CURR = a.SK_ID_CURR\n"
            "GROUP BY bureau_credits\n"
            "ORDER BY bureau_credits"
        ),
        tags=("bureau", "join", "history", "count", "prior"),
        note="Aggregate the child table first, then join, or rows double count.",
    ),
    FewShotExample(
        question="How many clients had a previous application refused?",
        sql=(
            "SELECT count(DISTINCT p.SK_ID_CURR) AS clients_with_refusal\n"
            "FROM previous_application p\n"
            "WHERE p.NAME_CONTRACT_STATUS = 'Refused'"
        ),
        tags=("previous", "refused", "rejected", "count", "distinct"),
        note="COUNT(DISTINCT client) when a client can appear many times.",
    ),
    FewShotExample(
        question="Which occupations have the highest average loan amount?",
        sql=(
            "SELECT OCCUPATION_TYPE,\n"
            "       count(*) AS applications,\n"
            "       avg(AMT_CREDIT) AS avg_loan_amount\n"
            "FROM application_train\n"
            "WHERE OCCUPATION_TYPE IS NOT NULL\n"
            "GROUP BY OCCUPATION_TYPE\n"
            "ORDER BY avg_loan_amount DESC\n"
            "LIMIT 10"
        ),
        tags=("occupation", "top", "ranking", "highest", "loan", "amount"),
    ),
    FewShotExample(
        question="Are late installment payments associated with default?",
        sql=(
            "WITH lateness AS (\n"
            "  SELECT SK_ID_CURR,\n"
            "         avg(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS avg_days_late\n"
            "  FROM installments_payments\n"
            "  WHERE DAYS_ENTRY_PAYMENT IS NOT NULL\n"
            "  GROUP BY SK_ID_CURR\n"
            ")\n"
            "SELECT a.TARGET,\n"
            "       count(*) AS clients,\n"
            "       avg(l.avg_days_late) AS avg_days_late\n"
            "FROM application_train a\n"
            "JOIN lateness l ON l.SK_ID_CURR = a.SK_ID_CURR\n"
            "GROUP BY a.TARGET"
        ),
        tags=("installment", "late", "payment", "repayment", "behaviour"),
        note="Payment lateness is DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT.",
    ),
    FewShotExample(
        question="What is the default rate for the highest and lowest external score deciles?",
        sql=(
            "WITH scored AS (\n"
            "  SELECT TARGET,\n"
            "         ntile(10) OVER (ORDER BY EXT_SOURCE_2) AS decile\n"
            "  FROM application_train\n"
            "  WHERE EXT_SOURCE_2 IS NOT NULL\n"
            ")\n"
            "SELECT decile, count(*) AS applications, avg(TARGET) AS default_rate\n"
            "FROM scored\n"
            "GROUP BY decile\n"
            "ORDER BY decile"
        ),
        tags=("decile", "score", "external", "distribution", "quantile"),
    ),
    FewShotExample(
        question="Compare the default rate for cash loans versus revolving loans.",
        sql=(
            "SELECT NAME_CONTRACT_TYPE,\n"
            "       count(*) AS applications,\n"
            "       avg(TARGET) AS default_rate\n"
            "FROM application_train\n"
            "GROUP BY NAME_CONTRACT_TYPE\n"
            "ORDER BY default_rate DESC"
        ),
        tags=("compare", "contract", "cash", "revolving", "versus", "segment"),
    ),
)


SQL_SYSTEM_PROMPT = """\
You are a SQL analyst for a bank's credit risk team. You translate questions \
about the Home Credit loan portfolio into a single DuckDB SQL query.

RULES
1. Output raw SQL only. No prose, no explanation, no markdown fences.
2. Write exactly one SELECT statement. Never write INSERT, UPDATE, DELETE, \
CREATE, DROP, COPY, ATTACH or PRAGMA.
3. Use only the tables and columns given in the schema below. Never invent a \
column name. If you need a column that is not listed, the question cannot be \
answered.
4. Never call functions that read outside the database, such as read_csv or \
glob.
5. Prefer readable aliases (AS default_rate, AS applications) - they become \
the column headings a non-technical person reads.
6. Always return the count of rows behind any rate or average, so the reader \
can judge whether the number is reliable.
7. Order results so the most interesting row is first, and limit long results.
8. If the question cannot be answered from this schema, or is not about this \
loan portfolio, reply with exactly one line:
   {refusal} <short reason>

SCHEMA
{schema}
"""


SQL_USER_PROMPT = """\
{examples}Question: {question}

SQL:"""


SQL_REPAIR_PROMPT = """\
The SQL you wrote was rejected before it ran.

Your query:
{sql}

Rejected because:
{errors}

Write a corrected single SELECT query using only the columns in the schema \
above. Output raw SQL only. If the question genuinely cannot be answered from \
the available columns, reply with exactly:
{refusal} <short reason>

SQL:"""


SUMMARY_SYSTEM_PROMPT = """\
You explain query results to bank staff who do not read SQL.

RULES
1. Use only the numbers in the results below. Never introduce a figure that \
is not there, and never estimate, extrapolate or recall a value from memory.
2. Lead with the direct answer in one sentence.
3. Add at most two more sentences of context - the comparison that makes the \
number meaningful, or the caveat that stops it being misread.
4. Round sensibly: rates as percentages to one decimal, money to whole units \
with thousands separators.
5. If the results are empty, say plainly that no matching records were found. \
Do not speculate about why.
6. If the results are a truncated sample, say the answer covers the rows shown.
7. Plain language. No SQL, no column names in capitals, no bullet points.
"""


SUMMARY_USER_PROMPT = """\
Question: {question}

Query that ran:
{sql}

Results ({row_count} row(s){truncation_note}):
{results}

Answer:"""


EXPLANATION_SYSTEM_PROMPT = """\
You explain an individual credit decision to someone with no statistics \
background - a loan officer, or the applicant themselves.

You are given the factors the model weighed, each with a direction and a \
size. Positive means the factor pushed the risk up; negative means it pushed \
the risk down.

RULES
1. Use only the factors listed. Never invent a reason, and never mention a \
factor that is not in the list.
2. Start with the outcome: the risk band and what it means in practice.
3. Name the two or three factors that mattered most, in plain words. Say what \
the applicant's value was and which way it pushed the decision.
4. Describe associations, not causes. The model observed a pattern; it did \
not prove that one thing causes another.
5. Never state or imply that the decision is final, and never give the \
applicant advice about how to change the outcome.
6. Four sentences at most. No jargon, no bullet points, no percentages beyond \
those supplied.
"""


EXPLANATION_USER_PROMPT = """\
Risk score: {probability:.1%} probability of default
Risk band: {band}
Portfolio average: {base_rate:.1%}

Factors that moved this score:
{factors}

Explanation:"""


def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def select_examples(
    question: str, *, limit: int = 3
) -> tuple[FewShotExample, ...]:
    """Pick the few-shot examples closest to the question.

    Scores on tag hits and shared question vocabulary. Deterministic, so an
    evaluator re-running a question sees the same prompt. When nothing matches
    the first `limit` examples are used, which are the general rate and
    group-by patterns most questions resemble anyway.
    """
    words = _tokenise(question)
    lowered = question.lower()

    scored: list[tuple[float, int, FewShotExample]] = []
    for index, example in enumerate(FEW_SHOT_EXAMPLES):
        score = 3.0 * sum(1 for tag in example.tags if tag in lowered)
        score += 1.0 * len(words & _tokenise(example.question))
        scored.append((-score, index, example))

    scored.sort()
    chosen = [example for negative, _, example in scored if -negative > 0][:limit]
    return tuple(chosen) if chosen else FEW_SHOT_EXAMPLES[:limit]


def render_examples(examples: tuple[FewShotExample, ...]) -> str:
    """Format examples for the prompt, notes included.

    The notes are what turn an example from a template to copy into a lesson
    about this dataset's quirks.
    """
    if not examples:
        return ""
    blocks = []
    for example in examples:
        block = f"Question: {example.question}\nSQL:\n{example.sql}"
        if example.note:
            block += f"\n-- {example.note}"
        blocks.append(block)
    return "Worked examples:\n\n" + "\n\n".join(blocks) + "\n\n"


def build_sql_prompt(
    question: str, schema_text: str, *, example_limit: int = 3
) -> tuple[str, str]:
    """Return the (system, user) prompt pair for SQL generation."""
    system = SQL_SYSTEM_PROMPT.format(schema=schema_text, refusal=REFUSAL_PREFIX)
    user = SQL_USER_PROMPT.format(
        examples=render_examples(select_examples(question, limit=example_limit)),
        question=question,
    )
    return system, user


def build_repair_prompt(sql: str, errors: list[str]) -> str:
    """Prompt for one bounded retry after validation failed."""
    return SQL_REPAIR_PROMPT.format(
        sql=sql,
        errors="\n".join(f"- {e}" for e in errors),
        refusal=REFUSAL_PREFIX,
    )


def is_refusal(text: str) -> bool:
    return text.strip().upper().startswith(REFUSAL_PREFIX)


def refusal_reason(text: str) -> str:
    """Extract the human-readable reason from a refusal."""
    stripped = text.strip()
    reason = stripped[len(REFUSAL_PREFIX):].strip() if is_refusal(stripped) else stripped
    return reason or "That question cannot be answered from this dataset."
