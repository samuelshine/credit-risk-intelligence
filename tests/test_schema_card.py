"""Tests for schema-card pruning.

Two things must hold at once: the card has to be materially smaller than the
full schema (the token argument), and it must still contain the columns the
question is about (the accuracy argument). A card that is small but missing the
answer is worse than no pruning at all.
"""

from __future__ import annotations

import pytest

from src.talk_to_data.catalog import Catalog
from src.talk_to_data.schema_card import (
    DATASET_NOTES,
    render_schema_card,
    score_tables,
)


def test_full_card_contains_every_table(realistic_catalog: Catalog) -> None:
    card = render_schema_card(realistic_catalog, None)
    for table in realistic_catalog.chatbot_tables:
        assert table in card.text
    assert card.columns_total == sum(
        len(t.columns) for t in realistic_catalog.tables.values()
    )


def test_pruning_materially_reduces_size(realistic_catalog: Catalog) -> None:
    full = render_schema_card(realistic_catalog, None)
    pruned = render_schema_card(
        realistic_catalog, "How does default rate differ by gender?"
    )
    assert pruned.approx_tokens < full.approx_tokens * 0.7


@pytest.mark.parametrize(
    "question,required_columns",
    [
        ("How does default rate differ by gender?", ["CODE_GENDER", "TARGET"]),
        ("What is the average income of applicants who defaulted?",
         ["AMT_INCOME_TOTAL", "TARGET"]),
        ("Show the default rate by age band", ["DAYS_BIRTH", "TARGET"]),
        ("Which education level defaults most?",
         ["NAME_EDUCATION_TYPE", "TARGET"]),
        ("Do clients with overdue bureau credit default more?",
         ["AMT_CREDIT_SUM_OVERDUE", "CREDIT_ACTIVE"]),
        ("How many previous applications were refused?",
         ["NAME_CONTRACT_STATUS"]),
        ("Are late installment payments linked to default?",
         ["DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"]),
    ],
)
def test_pruned_card_keeps_the_answering_columns(
    realistic_catalog: Catalog, question: str, required_columns: list[str]
) -> None:
    """The whole point: pruning must never drop the column that answers it."""
    card = render_schema_card(realistic_catalog, question)
    for column in required_columns:
        assert column in card.text, f"{column!r} pruned out of: {question!r}"


def test_join_keys_always_survive_pruning(realistic_catalog: Catalog) -> None:
    """Dropping a join key silently breaks every multi-table question."""
    card = render_schema_card(
        realistic_catalog, "What is the average loan amount?"
    )
    assert "SK_ID_CURR" in card.text


def test_low_signal_columns_lose_to_relevant_ones(
    realistic_catalog: Catalog,
) -> None:
    """Building statistics must not crowd out the columns people ask about."""
    card = render_schema_card(
        realistic_catalog, "How does default rate differ by gender?"
    )
    application_block = card.text.split("bureau (")[0]
    assert "COMMONAREA_MEDI" not in application_block
    assert "APARTMENTS_AVG" not in application_block
    assert "NAME_EDUCATION_TYPE" in application_block


@pytest.mark.parametrize(
    "question,expected_table",
    [
        ("How many bureau credits are active?", "bureau"),
        ("How many previous applications were refused?", "previous_application"),
        ("Are installment payments made late?", "installments_payments"),
        ("What is the credit card balance?", "credit_card_balance"),
    ],
)
def test_question_routes_to_the_right_table(
    realistic_catalog: Catalog, question: str, expected_table: str
) -> None:
    scores = score_tables(question, realistic_catalog)
    assert scores[expected_table] == max(scores.values())


def test_application_train_is_never_dropped(realistic_catalog: Catalog) -> None:
    """It holds TARGET, so nearly every business question needs it."""
    card = render_schema_card(realistic_catalog, "What is the pos cash balance?")
    assert "application_train" in card.tables_included


def test_dataset_quirks_are_always_stated(realistic_catalog: Catalog) -> None:
    """These are the errors the model makes without being told."""
    card = render_schema_card(realistic_catalog, "What is the average age?")
    assert "365243" in card.text          # the DAYS_EMPLOYED sentinel
    assert "DAYS_BIRTH" in card.text
    assert len(DATASET_NOTES) >= 5
    assert "TARGET = 1" in card.text


def test_small_tables_are_never_pruned(realistic_catalog: Catalog) -> None:
    """bureau_balance has 3 columns; pruning it saves nothing."""
    card = render_schema_card(
        realistic_catalog, "What is the monthly bureau balance status history?"
    )
    for column in ("SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"):
        assert column in card.text
