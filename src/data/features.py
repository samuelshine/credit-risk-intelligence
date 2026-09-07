"""Feature engineering, expressed as SQL.

The model's inputs are built by DuckDB, not pandas. Three reasons, in order of
how much they mattered:

**Memory.** Aggregating 27M rows of `bureau_balance` and 13.6M rows of
`installments_payments` in pandas means materialising them first. In SQL the
engine streams and only the per-client result lands in memory, which is what
lets the same code run on a laptop and on a 2 GB cloud instance.

**Auditability.** A credit model has to be explainable to a regulator. "This
feature is `avg(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)` over the client's
installments" is a sentence someone can check. The equivalent chain of pandas
merges and groupbys is not.

**One source of truth.** The chatbot answers questions against these same
tables. Building features anywhere else invites the model and the chatbot to
disagree about what "average bureau debt" means.

Every aggregate is a LEFT JOIN: a client with no bureau history keeps their row
and gets NULLs, which LightGBM handles natively. Dropping them would discard a
large, systematically different slice of the portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from src.data.database import get_readonly_connection
from src.utils.logger import get_logger, log_duration

log = get_logger(__name__)

#: DAYS_EMPLOYED uses 365243 to mean "not employed" - pensioners and the
#: unemployed, roughly 18% of applicants. Left as-is it becomes a 1000-year
#: tenure that dominates every split. Mapped to NULL, with the fact itself
#: retained as a flag, because "not employed" is genuinely predictive.
_DAYS_EMPLOYED_CLEAN = (
    "CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE DAYS_EMPLOYED END"
)

#: Ratios a credit analyst would compute by hand. These consistently rank among
#: the strongest features on this dataset because they express affordability,
#: which the raw amounts only imply.
APPLICATION_DERIVED = f"""
    -- Affordability
    AMT_CREDIT / nullif(AMT_INCOME_TOTAL, 0)          AS CREDIT_INCOME_RATIO,
    AMT_ANNUITY / nullif(AMT_INCOME_TOTAL, 0)         AS ANNUITY_INCOME_RATIO,
    AMT_ANNUITY / nullif(AMT_CREDIT, 0)               AS CREDIT_TERM,
    AMT_CREDIT / nullif(AMT_GOODS_PRICE, 0)           AS CREDIT_GOODS_RATIO,
    AMT_INCOME_TOTAL / nullif(CNT_FAM_MEMBERS, 0)     AS INCOME_PER_FAMILY_MEMBER,
    AMT_INCOME_TOTAL / nullif(CNT_CHILDREN + 1, 0)    AS INCOME_PER_CHILD,

    -- Human-scale versions of the DAYS_* offsets
    -DAYS_BIRTH / 365.25                              AS AGE_YEARS,
    -{_DAYS_EMPLOYED_CLEAN} / 365.25                  AS YEARS_EMPLOYED,
    {_DAYS_EMPLOYED_CLEAN} / nullif(DAYS_BIRTH, 0)    AS EMPLOYED_LIFE_FRACTION,
    (DAYS_EMPLOYED = 365243)::INT                     AS FLAG_NOT_EMPLOYED,
    -DAYS_REGISTRATION / 365.25                       AS YEARS_REGISTERED,
    -DAYS_ID_PUBLISH / 365.25                         AS YEARS_SINCE_ID_PUBLISH,

    -- The external bureau scores are the strongest single predictors here, so
    -- their combinations are worth making explicit rather than leaving the
    -- trees to rediscover.
    (coalesce(EXT_SOURCE_1, 0) + coalesce(EXT_SOURCE_2, 0)
        + coalesce(EXT_SOURCE_3, 0))
      / nullif((EXT_SOURCE_1 IS NOT NULL)::INT + (EXT_SOURCE_2 IS NOT NULL)::INT
               + (EXT_SOURCE_3 IS NOT NULL)::INT, 0) AS EXT_SOURCE_MEAN,
    least(coalesce(EXT_SOURCE_1, 1), coalesce(EXT_SOURCE_2, 1),
          coalesce(EXT_SOURCE_3, 1))                  AS EXT_SOURCE_MIN,
    greatest(coalesce(EXT_SOURCE_1, 0), coalesce(EXT_SOURCE_2, 0),
             coalesce(EXT_SOURCE_3, 0))               AS EXT_SOURCE_MAX,
    coalesce(EXT_SOURCE_1, 0) * coalesce(EXT_SOURCE_2, 0)
        * coalesce(EXT_SOURCE_3, 0)                   AS EXT_SOURCE_PRODUCT,
    (EXT_SOURCE_1 IS NULL)::INT + (EXT_SOURCE_2 IS NULL)::INT
        + (EXT_SOURCE_3 IS NULL)::INT                 AS EXT_SOURCE_MISSING_COUNT
"""

#: Prior credits held at other institutions.
BUREAU_AGGREGATE = """
SELECT
    SK_ID_CURR,
    count(*)                                          AS BUR_COUNT,
    count(*) FILTER (WHERE CREDIT_ACTIVE = 'Active')  AS BUR_ACTIVE_COUNT,
    count(*) FILTER (WHERE CREDIT_ACTIVE = 'Closed')  AS BUR_CLOSED_COUNT,
    count(*) FILTER (WHERE CREDIT_ACTIVE = 'Active')
        / nullif(count(*), 0)                         AS BUR_ACTIVE_RATIO,
    count(DISTINCT CREDIT_TYPE)                       AS BUR_CREDIT_TYPES,

    sum(AMT_CREDIT_SUM)                               AS BUR_CREDIT_SUM_TOTAL,
    avg(AMT_CREDIT_SUM)                               AS BUR_CREDIT_SUM_MEAN,
    max(AMT_CREDIT_SUM)                               AS BUR_CREDIT_SUM_MAX,
    sum(AMT_CREDIT_SUM_DEBT)                          AS BUR_DEBT_TOTAL,
    sum(AMT_CREDIT_SUM_DEBT) / nullif(sum(AMT_CREDIT_SUM), 0)
                                                      AS BUR_DEBT_CREDIT_RATIO,
    sum(AMT_CREDIT_SUM_OVERDUE)                       AS BUR_OVERDUE_TOTAL,
    max(AMT_CREDIT_MAX_OVERDUE)                       AS BUR_MAX_OVERDUE_EVER,
    max(CREDIT_DAY_OVERDUE)                           AS BUR_DAYS_OVERDUE_MAX,
    count(*) FILTER (WHERE CREDIT_DAY_OVERDUE > 0)    AS BUR_OVERDUE_COUNT,
    sum(CNT_CREDIT_PROLONG)                           AS BUR_PROLONG_TOTAL,

    -- How long the client has had a credit footprint, and how fresh it is.
    min(DAYS_CREDIT)                                  AS BUR_DAYS_CREDIT_MIN,
    max(DAYS_CREDIT)                                  AS BUR_DAYS_CREDIT_MAX,
    avg(DAYS_CREDIT)                                  AS BUR_DAYS_CREDIT_MEAN,
    max(DAYS_CREDIT_UPDATE)                           AS BUR_LAST_UPDATE,
    avg(AMT_ANNUITY)                                  AS BUR_ANNUITY_MEAN
FROM bureau
GROUP BY SK_ID_CURR
"""

#: Monthly repayment status on those bureau credits. STATUS is '0'-'5' for
#: months past due, 'C' for closed and 'X' for unknown, so the digits are the
#: delinquency signal and the letters are not.
BUREAU_BALANCE_AGGREGATE = """
SELECT
    b.SK_ID_CURR,
    count(*)                                          AS BURBAL_MONTHS,
    count(*) FILTER (WHERE bb.STATUS BETWEEN '1' AND '5')
                                                      AS BURBAL_DPD_MONTHS,
    count(*) FILTER (WHERE bb.STATUS BETWEEN '1' AND '5')
        / nullif(count(*), 0)                         AS BURBAL_DPD_RATIO,
    max(try_cast(bb.STATUS AS INTEGER))               AS BURBAL_WORST_STATUS,
    count(*) FILTER (WHERE bb.STATUS = 'C')
        / nullif(count(*), 0)                         AS BURBAL_CLOSED_RATIO,
    min(bb.MONTHS_BALANCE)                            AS BURBAL_OLDEST_MONTH
FROM bureau_balance bb
JOIN bureau b ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU
GROUP BY b.SK_ID_CURR
"""

#: The client's own history with this lender.
PREVIOUS_APPLICATION_AGGREGATE = """
SELECT
    SK_ID_CURR,
    count(*)                                              AS PREV_COUNT,
    count(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved')
                                                          AS PREV_APPROVED,
    count(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Refused')
                                                          AS PREV_REFUSED,
    count(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Refused')
        / nullif(count(*), 0)                             AS PREV_REFUSAL_RATE,
    count(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Canceled')
        / nullif(count(*), 0)                             AS PREV_CANCEL_RATE,

    avg(AMT_APPLICATION)                                  AS PREV_APPLICATION_MEAN,
    avg(AMT_CREDIT)                                       AS PREV_CREDIT_MEAN,
    -- Below 1 means the lender granted less than was asked for, which is a
    -- record of past underwriting caution about this client.
    avg(AMT_CREDIT / nullif(AMT_APPLICATION, 0))          AS PREV_GRANTED_RATIO,
    avg(AMT_DOWN_PAYMENT)                                 AS PREV_DOWNPAYMENT_MEAN,
    avg(CNT_PAYMENT)                                      AS PREV_TERM_MEAN,
    max(DAYS_DECISION)                                    AS PREV_LAST_DECISION,
    min(DAYS_DECISION)                                    AS PREV_FIRST_DECISION,
    count(DISTINCT CODE_REJECT_REASON) FILTER (
        WHERE CODE_REJECT_REASON NOT IN ('XAP', 'XNA'))   AS PREV_REJECT_REASONS
FROM previous_application
GROUP BY SK_ID_CURR
"""

#: Scheduled versus actual repayments. This is the closest thing in the dataset
#: to observed repayment behaviour, and it is where the strongest behavioural
#: features come from.
INSTALLMENTS_AGGREGATE = """
SELECT
    SK_ID_CURR,
    count(*)                                              AS INST_COUNT,
    -- Positive means paid after it was due.
    avg(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)             AS INST_DAYS_LATE_MEAN,
    max(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)             AS INST_DAYS_LATE_MAX,
    count(*) FILTER (WHERE DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT)
        / nullif(count(*), 0)                             AS INST_LATE_RATIO,
    count(*) FILTER (WHERE DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT > 30)
        / nullif(count(*), 0)                             AS INST_LATE_30D_RATIO,

    -- Positive means they paid less than was due.
    avg(AMT_INSTALMENT - AMT_PAYMENT)                     AS INST_SHORTFALL_MEAN,
    sum(AMT_INSTALMENT - AMT_PAYMENT)                     AS INST_SHORTFALL_TOTAL,
    max(AMT_INSTALMENT - AMT_PAYMENT)                     AS INST_SHORTFALL_MAX,
    count(*) FILTER (WHERE AMT_PAYMENT < AMT_INSTALMENT)
        / nullif(count(*), 0)                             AS INST_UNDERPAID_RATIO,
    sum(AMT_PAYMENT) / nullif(sum(AMT_INSTALMENT), 0)     AS INST_PAYMENT_RATIO,
    max(DAYS_INSTALMENT)                                  AS INST_LAST_DUE
FROM installments_payments
GROUP BY SK_ID_CURR
"""

CREDIT_CARD_AGGREGATE = """
SELECT
    SK_ID_CURR,
    count(*)                                              AS CC_MONTHS,
    count(DISTINCT SK_ID_PREV)                            AS CC_CARDS,
    avg(AMT_BALANCE)                                      AS CC_BALANCE_MEAN,
    max(AMT_BALANCE)                                      AS CC_BALANCE_MAX,
    avg(AMT_BALANCE / nullif(AMT_CREDIT_LIMIT_ACTUAL, 0)) AS CC_UTILISATION_MEAN,
    max(AMT_BALANCE / nullif(AMT_CREDIT_LIMIT_ACTUAL, 0)) AS CC_UTILISATION_MAX,
    avg(AMT_DRAWINGS_CURRENT)                             AS CC_DRAWINGS_MEAN,
    avg(CNT_DRAWINGS_CURRENT)                             AS CC_DRAWINGS_COUNT_MEAN,
    max(SK_DPD)                                           AS CC_DPD_MAX,
    count(*) FILTER (WHERE SK_DPD > 0)
        / nullif(count(*), 0)                             AS CC_DPD_RATIO
FROM credit_card_balance
GROUP BY SK_ID_CURR
"""

POS_CASH_AGGREGATE = """
SELECT
    SK_ID_CURR,
    count(*)                                              AS POS_MONTHS,
    count(DISTINCT SK_ID_PREV)                            AS POS_CONTRACTS,
    max(SK_DPD)                                           AS POS_DPD_MAX,
    avg(SK_DPD)                                           AS POS_DPD_MEAN,
    count(*) FILTER (WHERE SK_DPD > 0)
        / nullif(count(*), 0)                             AS POS_DPD_RATIO,
    max(SK_DPD_DEF)                                       AS POS_DPD_DEF_MAX,
    avg(CNT_INSTALMENT_FUTURE)                            AS POS_REMAINING_MEAN,
    min(MONTHS_BALANCE)                                   AS POS_OLDEST_MONTH
FROM pos_cash_balance
GROUP BY SK_ID_CURR
"""

#: Each aggregate, joined onto the application table by SK_ID_CURR.
AGGREGATES: dict[str, str] = {
    "bur": BUREAU_AGGREGATE,
    "burbal": BUREAU_BALANCE_AGGREGATE,
    "prev": PREVIOUS_APPLICATION_AGGREGATE,
    "inst": INSTALLMENTS_AGGREGATE,
    "cc": CREDIT_CARD_AGGREGATE,
    "pos": POS_CASH_AGGREGATE,
}

#: Columns never given to the model. SK_ID_CURR identifies the row and would
#: let the model memorise it; TARGET is the label.
EXCLUDED_COLUMNS: frozenset[str] = frozenset({"SK_ID_CURR", "TARGET"})


@dataclass
class FeatureMatrix:
    """Model inputs, their labels, and the ids that tie them back to a client."""

    frame: pd.DataFrame
    target: pd.Series | None
    ids: pd.Series
    categorical_columns: list[str]

    @property
    def feature_names(self) -> list[str]:
        return list(self.frame.columns)

    def __len__(self) -> int:
        return len(self.frame)


def build_feature_sql(source_table: str = "application_train") -> str:
    """Assemble the full feature query for one application table.

    Ends with `ORDER BY app.SK_ID_CURR`, which is not cosmetic: DuckDB gives
    no row-order guarantee for a query without one, particularly with
    multiple LEFT JOINs and parallel execution. Without it, two runs of this
    exact query can return rows in different orders - and since
    `sklearn.train_test_split(..., random_state=42)` splits by *position*,
    not by id, an unordered result silently makes "the same random_state"
    produce a *different* train/holdout split every run. That surfaced as a
    real bug: a holdout evaluation reconstructed from a fresh query call
    scored the model at ROC-AUC 0.855 against the 0.785 the original training
    run measured on its own (correctly ordered, in-process) split - the
    mismatch was rows leaking across the split rather than a modelling issue.
    """
    joins = "\n".join(
        f"LEFT JOIN ({sql.strip()}) {alias} ON {alias}.SK_ID_CURR = app.SK_ID_CURR"
        for alias, sql in AGGREGATES.items()
    )
    aggregate_columns = ",\n    ".join(
        f"{alias}.* EXCLUDE (SK_ID_CURR)" for alias in AGGREGATES
    )
    return f"""
SELECT
    app.* EXCLUDE (SK_ID_CURR),
    app.SK_ID_CURR,
    {APPLICATION_DERIVED.strip()},
    {aggregate_columns}
FROM {source_table} app
{joins}
ORDER BY app.SK_ID_CURR
"""


def load_features(
    source_table: str = "application_train",
    conn: duckdb.DuckDBPyConnection | None = None,
    limit: int | None = None,
) -> FeatureMatrix:
    """Build and materialise the feature matrix.

    Returns everything the training and inference paths need, so both go
    through exactly the same code and cannot drift apart.
    """
    conn = conn or get_readonly_connection()
    sql = build_feature_sql(source_table)
    if limit:
        sql += f"\nLIMIT {limit}"

    with log_duration(log, f"build features from {source_table}"):
        frame = conn.execute(sql).df()

    ids = frame["SK_ID_CURR"].astype("int64")
    target = frame["TARGET"].astype("int8") if "TARGET" in frame.columns else None

    features = frame.drop(columns=[c for c in EXCLUDED_COLUMNS if c in frame.columns])

    # Object columns are the dataset's categoricals. LightGBM consumes pandas
    # `category` dtype directly, which avoids one-hot expanding
    # ORGANIZATION_TYPE's 58 levels into 58 sparse columns.
    categorical_columns = [
        c for c in features.columns if features[c].dtype == object
    ]
    for column in categorical_columns:
        features[column] = features[column].astype("category")

    log.info(
        "features: %s rows x %d columns (%d categorical, %d numeric)",
        f"{len(features):,}", features.shape[1], len(categorical_columns),
        features.shape[1] - len(categorical_columns),
    )
    return FeatureMatrix(
        frame=features,
        target=target,
        ids=ids,
        categorical_columns=categorical_columns,
    )
