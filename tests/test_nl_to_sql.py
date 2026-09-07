"""Pipeline tests with a scripted model.

The point is the control flow around the model, not the model itself: does a
rejected query get exactly one repair attempt, does a refusal short-circuit,
does a hallucinated column never reach the database. A stub returns canned SQL
so these run in CI with no API key and no network.
"""

from __future__ import annotations

import os

import pytest

from src.llm.gemini import LLMResponse, Usage
from src.talk_to_data.catalog import Catalog, ColumnInfo, TableInfo


class StubClient:
    """Returns queued responses in order and records the prompts it saw."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, str]] = []

    available = True

    def generate(self, *, role, system, user, **kwargs) -> LLMResponse:
        self.calls.append({"role": role, "system": system, "user": user})
        text = self.responses.pop(0) if self.responses else ""
        return LLMResponse(
            text=text,
            usage=Usage(model=f"stub-{role}", prompt_tokens=100,
                        output_tokens=20, total_tokens=120),
        )


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("nl2sql")
    os.environ["DUCKDB_PATH"] = str(tmp / "t.duckdb")
    os.environ["DATA_DIR"] = str(tmp)
    os.environ["DUCKDB_MEMORY_LIMIT"] = "1GB"

    from src.utils.config import get_settings
    get_settings.cache_clear()

    from src.data.database import reset_readonly_connection, writable_connection
    reset_readonly_connection()
    with writable_connection() as conn:
        conn.execute(
            "CREATE TABLE application_train AS SELECT "
            "  i AS SK_ID_CURR, (i % 12 = 0)::BIGINT AS TARGET, "
            "  CASE WHEN i % 2 = 0 THEN 'F' ELSE 'M' END AS CODE_GENDER, "
            "  (20000 + i * 13)::DOUBLE AS AMT_INCOME_TOTAL "
            "FROM range(1200) t(i)"
        )
    yield tmp
    reset_readonly_connection()


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(tables={
        "application_train": TableInfo(
            name="application_train", row_count=1200,
            columns={c.name: c for c in [
                ColumnInfo("SK_ID_CURR", "BIGINT"),
                ColumnInfo("TARGET", "BIGINT"),
                ColumnInfo("CODE_GENDER", "VARCHAR"),
                ColumnInfo("AMT_INCOME_TOTAL", "DOUBLE"),
            ]},
        )
    })


def _service(catalog, responses):
    from src.talk_to_data.nl_to_sql import TalkToData
    stub = StubClient(responses)
    return TalkToData(client=stub, catalog=catalog), stub


def test_happy_path_answers_from_rows(db, catalog) -> None:
    service, stub = _service(catalog, [
        "SELECT avg(TARGET) AS default_rate, count(*) AS applications "
        "FROM application_train",
        "About 8.3% of applicants defaulted, across 1,200 applications.",
    ])
    answer = service.ask("What is the overall default rate?")

    assert answer.ok
    assert not answer.repaired
    assert answer.row_count == 1
    assert answer.tables_used == ["application_train"]
    assert "8.3%" in answer.answer
    assert len(stub.calls) == 2
    assert stub.calls[0]["role"] == "sql"
    assert stub.calls[1]["role"] == "summary"


def test_schema_is_injected_into_the_prompt(db, catalog) -> None:
    """The model must be told what exists rather than left to guess."""
    service, stub = _service(catalog, [
        "SELECT count(*) AS n FROM application_train", "1,200 applications.",
    ])
    service.ask("How many applications are there?")

    system = stub.calls[0]["system"]
    assert "application_train" in system
    assert "CODE_GENDER" in system
    assert "365243" in system          # the dataset quirks travel with it
    assert "CANNOT_ANSWER" in system   # refusal is offered explicitly


def test_hallucinated_column_triggers_exactly_one_repair(db, catalog) -> None:
    service, stub = _service(catalog, [
        "SELECT avg(CUSTOMER_INCOME) FROM application_train",   # invented
        "SELECT avg(AMT_INCOME_TOTAL) AS avg_income FROM application_train",
        "The average income is about 27,800.",
    ])
    answer = service.ask("What is the average income?")

    assert answer.ok
    assert answer.repaired
    assert answer.validation_errors
    assert "CUSTOMER_INCOME" in answer.validation_errors[0]
    assert len(stub.calls) == 3
    # The repair prompt must carry the specific error, not a generic nudge.
    assert "CUSTOMER_INCOME" in stub.calls[1]["user"]


def test_repair_is_not_retried_a_second_time(db, catalog) -> None:
    """Two bad queries end the attempt. Looping would eventually pass by luck."""
    service, stub = _service(catalog, [
        "SELECT avg(BAD_ONE) FROM application_train",
        "SELECT avg(BAD_TWO) FROM application_train",
        "should never be reached",
    ])
    answer = service.ask("What is the average income?")

    assert not answer.ok
    assert answer.sql is None
    assert len(stub.calls) == 2


def test_refusal_short_circuits_without_querying(db, catalog) -> None:
    service, stub = _service(catalog, [
        "CANNOT_ANSWER: this dataset has no information about branch locations",
    ])
    answer = service.ask("Which branch has the nicest carpet?")

    assert answer.refused
    assert not answer.ok
    assert answer.sql is None
    assert "branch locations" in answer.answer
    assert len(stub.calls) == 1


def test_write_attempt_never_reaches_the_database(db, catalog) -> None:
    """Even if the model emits a DROP, it is rejected before execution."""
    service, stub = _service(catalog, [
        "DROP TABLE application_train",
        "DELETE FROM application_train",
    ])
    answer = service.ask("Delete everything")

    assert not answer.ok
    assert answer.sql is None
    from src.talk_to_data.query_runner import run_query
    assert run_query("SELECT count(*) AS n FROM application_train").rows[0][0] == 1200


def test_summariser_sees_rows_but_not_the_database(db, catalog) -> None:
    service, stub = _service(catalog, [
        "SELECT CODE_GENDER, avg(TARGET) AS default_rate "
        "FROM application_train GROUP BY CODE_GENDER",
        "Women defaulted more often than men in this sample.",
    ])
    service.ask("How does default rate differ by gender?")

    summary_prompt = stub.calls[1]["user"]
    assert "CODE_GENDER | default_rate" in summary_prompt
    assert "Use only the numbers" in stub.calls[1]["system"]


def test_empty_result_is_reported_honestly(db, catalog) -> None:
    service, stub = _service(catalog, [
        "SELECT * FROM application_train WHERE SK_ID_CURR < 0",
        "No matching records were found.",
    ])
    answer = service.ask("Show applicants with negative ids")

    assert answer.ok
    assert answer.row_count == 0
    assert "No matching records" in answer.answer


def test_token_usage_is_recorded_for_every_call(db, catalog) -> None:
    service, _ = _service(catalog, [
        "SELECT count(*) AS n FROM application_train", "1,200 applications.",
    ])
    answer = service.ask("How many applications?")

    assert len(answer.usage) == 2
    assert answer.total_tokens == 240
    assert answer.prompt_version


def test_row_limit_is_injected_into_generated_sql(db, catalog) -> None:
    service, _ = _service(catalog, [
        "SELECT SK_ID_CURR FROM application_train", "Showing applicant ids.",
    ])
    answer = service.ask("List applicant ids")

    assert "LIMIT" in answer.sql.upper()
    assert answer.row_count <= 500


def test_missing_api_key_degrades_with_a_clear_message(db, catalog) -> None:
    from src.llm.gemini import GeminiClient
    from src.talk_to_data.nl_to_sql import TalkToData
    from src.utils.config import Settings

    settings = Settings(google_api_key="", duckdb_path=os.environ["DUCKDB_PATH"])
    service = TalkToData(client=GeminiClient(settings), catalog=catalog)
    answer = service.ask("What is the default rate?")

    assert not answer.ok
    assert "GOOGLE_API_KEY" in answer.error
