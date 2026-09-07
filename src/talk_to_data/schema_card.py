"""Rendering the database schema into the smallest text that still works.

The naive approach - paste all seven tables and their ~220 columns into every
prompt - costs roughly 4,000 tokens per question before the user has said
anything, and it *hurts* accuracy: the column the question is actually about is
buried among two hundred irrelevant ones.

So the card is built per question, in two tiers:

* **Tier 1, always included.** Every table's name, row count, one-line purpose
  and join keys. Small, and the model needs all of it to plan a join.
* **Tier 2, selected.** Full column detail, but only for the tables the
  question plausibly touches, and only the columns within them that scored.

Selection is lexical and deterministic rather than embedding-based. That is a
deliberate trade: it needs no model call, no vector store and no warm-up, it is
reproducible for an evaluator, and on a schema this size a curated synonym map
covering the domain's vocabulary ("age", "gender", "loan size", "overdue")
outperforms generic similarity. `DOMAIN_SYNONYMS` is where that knowledge
lives, and it is grounded in the dataset's actual column names.

Anything the selector is unsure about is included rather than dropped: a
missing column produces a wrong answer, while a spare one costs a few tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.talk_to_data.catalog import Catalog, TableInfo
from src.utils.logger import get_logger

log = get_logger(__name__)

#: Compact type labels. "BIGINT" -> "int" saves a token per column, and across
#: 220 columns that is material.
_TYPE_LABELS: dict[str, str] = {
    "BIGINT": "int", "INTEGER": "int", "SMALLINT": "int", "TINYINT": "int",
    "HUGEINT": "int", "DOUBLE": "num", "FLOAT": "num", "REAL": "num",
    "DECIMAL": "num", "VARCHAR": "text", "BOOLEAN": "bool", "DATE": "date",
    "TIMESTAMP": "ts",
}

#: Business vocabulary -> the columns that answer it. This is the single most
#: load-bearing piece of domain knowledge in the talk-to-data path: it is what
#: lets "how many women defaulted" reach CODE_GENDER and TARGET without the
#: model having to guess from a 220-column dump.
DOMAIN_SYNONYMS: dict[str, tuple[str, ...]] = {
    # outcome
    "default": ("TARGET",),
    "defaulted": ("TARGET",),
    "risk": ("TARGET",),
    "repaid": ("TARGET",),
    "delinquent": ("TARGET",),
    "bad": ("TARGET",),
    # demographics
    "age": ("DAYS_BIRTH",),
    "old": ("DAYS_BIRTH",),
    "young": ("DAYS_BIRTH",),
    "gender": ("CODE_GENDER",),
    "male": ("CODE_GENDER",),
    "female": ("CODE_GENDER",),
    "women": ("CODE_GENDER",),
    "men": ("CODE_GENDER",),
    "children": ("CNT_CHILDREN", "CNT_FAM_MEMBERS"),
    "family": ("NAME_FAMILY_STATUS", "CNT_FAM_MEMBERS"),
    "married": ("NAME_FAMILY_STATUS",),
    "education": ("NAME_EDUCATION_TYPE",),
    "housing": ("NAME_HOUSING_TYPE", "FLAG_OWN_REALTY"),
    "car": ("FLAG_OWN_CAR", "OWN_CAR_AGE"),
    "occupation": ("OCCUPATION_TYPE",),
    "job": ("OCCUPATION_TYPE", "ORGANIZATION_TYPE"),
    "employer": ("ORGANIZATION_TYPE",),
    "employment": ("DAYS_EMPLOYED",),
    "employed": ("DAYS_EMPLOYED",),
    "tenure": ("DAYS_EMPLOYED",),
    "region": ("REGION_RATING_CLIENT", "REGION_POPULATION_RELATIVE"),
    "city": ("REGION_RATING_CLIENT_W_CITY",),
    # money
    "income": ("AMT_INCOME_TOTAL", "NAME_INCOME_TYPE"),
    "salary": ("AMT_INCOME_TOTAL",),
    "earn": ("AMT_INCOME_TOTAL",),
    "loan": ("AMT_CREDIT", "NAME_CONTRACT_TYPE"),
    "credit": ("AMT_CREDIT",),
    "amount": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
    "annuity": ("AMT_ANNUITY",),
    "instalment": ("AMT_ANNUITY", "AMT_INSTALMENT"),
    "installment": ("AMT_ANNUITY", "AMT_INSTALMENT"),
    "payment": ("AMT_PAYMENT", "AMT_INSTALMENT"),
    "goods": ("AMT_GOODS_PRICE",),
    "price": ("AMT_GOODS_PRICE",),
    "cash": ("NAME_CONTRACT_TYPE",),
    "revolving": ("NAME_CONTRACT_TYPE",),
    # external scores
    "score": ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"),
    "external": ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"),
    "bureau score": ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"),
    # credit history
    "history": ("DAYS_CREDIT", "CREDIT_ACTIVE"),
    "overdue": ("AMT_CREDIT_SUM_OVERDUE", "CREDIT_DAY_OVERDUE", "SK_DPD"),
    "arrears": ("AMT_CREDIT_SUM_OVERDUE", "CREDIT_DAY_OVERDUE"),
    "late": ("DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT", "SK_DPD"),
    "debt": ("AMT_CREDIT_SUM_DEBT",),
    "active": ("CREDIT_ACTIVE",),
    "closed": ("CREDIT_ACTIVE",),
    "refused": ("NAME_CONTRACT_STATUS",),
    "rejected": ("NAME_CONTRACT_STATUS",),
    "approved": ("NAME_CONTRACT_STATUS",),
    "previous": ("NAME_CONTRACT_STATUS", "AMT_APPLICATION"),
    "balance": ("AMT_BALANCE", "MONTHS_BALANCE"),
    "card": ("AMT_BALANCE", "AMT_DRAWINGS_CURRENT"),
    "status": ("STATUS", "CREDIT_ACTIVE", "NAME_CONTRACT_STATUS"),
}

#: Tables a question mentions by concept rather than by name.
TABLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "application_train": (
        "applicant", "application", "client", "customer", "borrower",
        "default", "target", "income", "age", "gender", "education",
    ),
    "bureau": (
        "bureau", "other bank", "external credit", "prior credit",
        "credit history", "overdue", "debt", "active credit",
    ),
    "bureau_balance": ("bureau balance", "monthly status", "dpd history"),
    "previous_application": (
        "previous application", "past application", "refused", "rejected",
        "approved", "prior application",
    ),
    "installments_payments": (
        "installment", "instalment", "repayment", "paid late", "payment history",
    ),
    "credit_card_balance": ("credit card", "card balance", "drawings"),
    "pos_cash_balance": ("pos", "point of sale", "cash loan balance"),
}

#: Columns that must survive pruning wherever they exist. Join keys, because
#: dropping one silently breaks every multi-table question, and TARGET, because
#: it is the outcome nearly every business question is ultimately about.
ALWAYS_KEEP: frozenset[str] = frozenset(
    {"SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "TARGET"}
)

#: Quirks of this dataset that the model cannot infer from column names and
#: will get wrong every time if not told. Each one is a bug we would otherwise
#: see in generated SQL.
DATASET_NOTES: tuple[str, ...] = (
    "TARGET = 1 means the client had payment difficulties (defaulted); "
    "0 means they repaid. The average of TARGET is the default rate.",
    "All DAYS_* columns are negative offsets in days from the application "
    "date. Age in years is -DAYS_BIRTH/365.25; years employed is "
    "-DAYS_EMPLOYED/365.25.",
    "DAYS_EMPLOYED = 365243 is a sentinel for 'not employed' (pensioners and "
    "the unemployed), not a real duration. Exclude it from employment "
    "calculations with DAYS_EMPLOYED != 365243.",
    "CODE_GENDER contains 'M', 'F' and a handful of 'XNA' rows.",
    "Only application_train has TARGET. Child tables join on SK_ID_CURR, "
    "except bureau_balance which joins to bureau on SK_ID_BUREAU.",
    "A client can have many rows in the child tables, so aggregate before "
    "joining, or use COUNT(DISTINCT ...), to avoid double counting.",
)


@dataclass
class SchemaCard:
    """A rendered schema, plus what it cost and what it kept."""

    text: str
    tables_included: list[str]
    columns_included: int
    columns_total: int

    @property
    def approx_tokens(self) -> int:
        """Rough token estimate: ~4 characters per token for English + SQL.

        Deliberately an estimate. The exact figure comes from the Gemini API's
        own usage metadata, which `llm.gemini` records per call; this is for
        selection-time budgeting where an API round trip is not available.
        """
        return max(1, len(self.text) // 4)


#: Column families ranked by how often a business question is actually about
#: them. Without this, a question that matches few columns spends its budget on
#: whatever sorts first alphabetically - APARTMENTS_AVG, BASEMENTAREA_MEDI,
#: COMMONAREA_MODE - instead of the columns a credit analyst would name.
#:
#: The demoted families are demoted on evidence, not taste: the 42 building
#: statistics are 50-70% missing in this dataset, the 20 FLAG_DOCUMENT_* columns
#: are near-constant, and neither appears in questions people ask in plain
#: English.
_HIGH_PRIOR_COLUMNS: frozenset[str] = frozenset({
    "TARGET", "CODE_GENDER", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "REGION_RATING_CLIENT",
    "CREDIT_ACTIVE", "CREDIT_TYPE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_OVERDUE", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT",
    "NAME_CONTRACT_STATUS", "CODE_REJECT_REASON", "AMT_APPLICATION",
    "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
    "STATUS", "MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF", "AMT_BALANCE",
})

#: Patterns for the wide, low-signal blocks that should lose every tiebreak.
_LOW_PRIOR_PATTERNS: tuple[str, ...] = (
    r".*_(AVG|MODE|MEDI)$",            # 42 building statistics
    r"FLAG_DOCUMENT_\d+",              # 20 near-constant document flags
    r"AMT_REQ_CREDIT_BUREAU_.*",       # 6 bureau enquiry counters
    r"(REG|LIVE)_(REGION|CITY)_NOT_.*",  # 6 address-mismatch flags
    r"FLAG_(MOBIL|EMP_PHONE|WORK_PHONE|CONT_MOBILE|PHONE|EMAIL)",
    r"(OBS|DEF)_\d+_CNT_SOCIAL_CIRCLE",
)
_LOW_PRIOR_RE = re.compile("|".join(f"(?:{p})" for p in _LOW_PRIOR_PATTERNS))


def _prior_score(column: str) -> float:
    """Baseline usefulness of a column, before the question is considered."""
    if column in _HIGH_PRIOR_COLUMNS:
        return 12.0
    if _LOW_PRIOR_RE.fullmatch(column):
        return -10.0
    return 0.0


def _type_label(data_type: str) -> str:
    base = re.split(r"[(\[]", data_type.upper())[0].strip()
    return _TYPE_LABELS.get(base, base.lower())


def _tokenise(question: str) -> set[str]:
    return set(re.findall(r"[a-z_]+", question.lower()))


def score_tables(question: str, catalog: Catalog) -> dict[str, float]:
    """Score each table's relevance to the question.

    A table scores for being named outright, for a concept keyword, and for
    each of its columns that the question's vocabulary reaches.
    """
    lowered = question.lower()
    words = _tokenise(question)
    scores: dict[str, float] = {}

    wanted_columns = {
        col
        for word in words
        for col in DOMAIN_SYNONYMS.get(word, ())
    }
    for phrase, cols in DOMAIN_SYNONYMS.items():
        if " " in phrase and phrase in lowered:
            wanted_columns.update(cols)

    for name in catalog.chatbot_tables:
        info = catalog.tables[name]
        score = 0.0

        if name.lower() in lowered or name.replace("_", " ") in lowered:
            score += 10.0
        for keyword in TABLE_KEYWORDS.get(name, ()):
            if keyword in lowered:
                score += 3.0
        score += 2.0 * len(wanted_columns & info.column_names)

        scores[name] = score

    # application_train holds TARGET and the applicant attributes, so nearly
    # every business question needs it. Floor it above zero rather than risk
    # dropping the table the question is actually about.
    scores["application_train"] = max(scores.get("application_train", 0.0), 1.0)
    return scores


def select_columns(
    question: str, table: TableInfo, *, budget: int
) -> list[str]:
    """Choose which of a table's columns to describe in full.

    Returns every column when the table is small enough to fit the budget -
    pruning a 17-column table saves nothing and risks dropping the answer.
    """
    if len(table.columns) <= budget:
        return list(table.columns)

    words = _tokenise(question)
    lowered = question.lower()

    wanted = {
        col for word in words for col in DOMAIN_SYNONYMS.get(word, ())
    }
    for phrase, cols in DOMAIN_SYNONYMS.items():
        if " " in phrase and phrase in lowered:
            wanted.update(cols)

    scored: list[tuple[float, str]] = []
    for name, info in table.columns.items():
        if name in ALWAYS_KEEP:
            scored.append((1_000.0, name))
            continue

        score = _prior_score(name)
        if name in wanted:
            score += 50.0
        readable = name.lower().replace("_", " ")
        if readable in lowered:
            score += 30.0
        # Any question word appearing in the column name or its glossary text.
        name_words = set(name.lower().split("_"))
        score += 5.0 * len(words & name_words)
        if info.description:
            score += 1.0 * len(words & _tokenise(info.description))
        scored.append((score, name))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [name for _, name in scored[:budget]]


def render_schema_card(
    catalog: Catalog,
    question: str | None = None,
    *,
    detail_tables: int = 3,
    column_budget: int = 28,
) -> SchemaCard:
    """Render the schema for a prompt.

    With no question, renders everything - used for documentation and for
    measuring what the pruning actually saves. With a question, renders tier 1
    for all tables and tier 2 for the `detail_tables` most relevant.
    """
    tables = catalog.chatbot_tables
    total_columns = sum(len(catalog.tables[t].columns) for t in tables)

    if question is None:
        detailed = set(tables)
        budget = 10_000
    else:
        scores = score_tables(question, catalog)
        ranked = sorted(tables, key=lambda t: (-scores.get(t, 0.0), t))
        detailed = {t for t in ranked[:detail_tables] if scores.get(t, 0.0) > 0}
        detailed.add("application_train")
        budget = column_budget

    lines: list[str] = ["TABLES"]
    included = 0

    for name in tables:
        info = catalog.tables[name]
        keys = [c for c in ("SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV")
                if c in info.columns]
        header = f"\n{name} ({info.row_count:,} rows)"
        if info.description:
            header += f" - {info.description}"
        lines.append(header)

        if name not in detailed:
            # Tier 1 only: enough to know the table exists and how to reach it.
            lines.append(
                f"  [{len(info.columns)} columns; join on {', '.join(keys)}. "
                f"Ask about this table to see its columns.]"
            )
            continue

        chosen = select_columns(question or "", info, budget=budget)
        for column in chosen:
            col = info.columns[column]
            entry = f"  {col.name} {_type_label(col.data_type)}"
            if col.description:
                entry += f" - {_shorten(col.description)}"
            lines.append(entry)
            included += 1

        hidden = len(info.columns) - len(chosen)
        if hidden > 0:
            lines.append(f"  [+{hidden} more columns not shown]")

    lines.append("\nNOTES")
    lines.extend(f"- {note}" for note in DATASET_NOTES)

    return SchemaCard(
        text="\n".join(lines),
        tables_included=sorted(detailed),
        columns_included=included,
        columns_total=total_columns,
    )


def _shorten(description: str, limit: int = 90) -> str:
    """Trim a glossary description to its first useful clause."""
    text = description.strip().rstrip(".")
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}..."
