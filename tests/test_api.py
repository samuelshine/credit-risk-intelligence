"""HTTP-level tests for the FastAPI service.

The route modules are thin adapters over already-tested modules
(`src.ml.*`, `src.talk_to_data.*`) - these tests check the HTTP contract
(status codes, response shapes, validation) and the degraded-mode paths
(no database, no trained model, no LLM key), not the underlying logic
those other test files already cover in depth.

Singletons (`RiskModel`, `Explainer`, `GeminiClient`, `TalkToData`) are
module-level and read settings at first use, so each test resets them
after pointing the environment at its own isolated fixtures - otherwise
state from one test's model would leak into the next's assertions.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from tests.conftest import create_empty_child_tables
from tests.conftest_ml import build_tiny_trained_model, point_settings_at


def _reset_all_singletons() -> None:
    from src.data.database import reset_readonly_connection
    from src.llm.gemini import reset_llm_client
    from src.ml.explain import reset_explainer
    from src.ml.predict import reset_risk_model
    from src.talk_to_data.catalog import get_catalog
    from src.talk_to_data.nl_to_sql import reset_talk_to_data

    reset_readonly_connection()
    reset_risk_model()
    reset_explainer()
    reset_llm_client()
    reset_talk_to_data()
    get_catalog.cache_clear()


@pytest.fixture
def client(tmp_path):
    """A TestClient with nothing set up: no database, no model, no key -
    the state the app is in on a completely fresh checkout."""
    point_settings_at(tmp_path)
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ["DUCKDB_PATH"] = str(tmp_path / "t.duckdb")
    os.environ["GOOGLE_API_KEY"] = ""
    from src.utils.config import get_settings
    get_settings.cache_clear()
    _reset_all_singletons()

    from src.api.main import app
    with TestClient(app) as test_client:
        yield test_client
    _reset_all_singletons()


@pytest.fixture
def trained_client(tmp_path):
    """A TestClient with a real (tiny) trained model and a real (tiny)
    database - enough to exercise every route's success path."""
    point_settings_at(tmp_path)
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ["DUCKDB_PATH"] = str(tmp_path / "t.duckdb")
    os.environ["GOOGLE_API_KEY"] = ""
    from src.utils.config import get_settings
    get_settings.cache_clear()
    _reset_all_singletons()

    build_tiny_trained_model(tmp_path)

    from src.data.database import writable_connection
    with writable_connection() as conn:
        # EXT_SOURCE_MEAN (what the tiny trained model was fit on) is a
        # *derived* feature - src.data.features.APPLICATION_DERIVED computes
        # it from EXT_SOURCE_1/2/3, so this table supplies those raw columns
        # rather than EXT_SOURCE_MEAN itself, plus every other raw column the
        # derived-feature SQL references unconditionally.
        conn.execute("""
            CREATE TABLE application_train AS
            SELECT
                100000 + i AS SK_ID_CURR,
                (i % 10 = 0)::BIGINT AS TARGET,
                (0.1 + 0.008 * i) AS EXT_SOURCE_1,
                (0.1 + 0.008 * i) AS EXT_SOURCE_2,
                (0.1 + 0.008 * i) AS EXT_SOURCE_3,
                (50000.0 + i * 1000) AS AMT_INCOME_TOTAL,
                (200000.0 + i * 2000) AS AMT_CREDIT,
                (15000.0 + i * 100) AS AMT_ANNUITY,
                (180000.0 + i * 1800) AS AMT_GOODS_PRICE,
                2::BIGINT AS CNT_FAM_MEMBERS,
                0::BIGINT AS CNT_CHILDREN,
                (-8000 - i * 30) AS DAYS_BIRTH,
                (-2000 - i * 10) AS DAYS_EMPLOYED,
                -500 AS DAYS_REGISTRATION,
                -500 AS DAYS_ID_PUBLISH,
                CASE WHEN i % 2 = 0 THEN 'F' ELSE 'M' END AS CODE_GENDER,
                'Higher education' AS NAME_EDUCATION_TYPE
            FROM range(100) t(i)
        """)
        create_empty_child_tables(conn)
        # A real ETL run always loads application_test too (it's one of the
        # required Kaggle files); RiskModel.predict_one falls back to it for
        # an id not found in application_train, so it must exist here even
        # though these tests never populate it with rows.
        conn.execute(
            "CREATE TABLE application_test AS "
            "SELECT * EXCLUDE (TARGET) FROM application_train WHERE 1=0"
        )

    from src.eda.analysis import build_eda_artifacts
    from src.eda.charts import render_all
    from src.utils.helpers import write_json
    with writable_connection() as conn:
        artifacts = build_eda_artifacts(conn)
    write_json(get_settings().models_dir / "eda_artifacts.json", artifacts)
    from src.eda.analysis import missing_value_report, run_all_insights
    with writable_connection() as conn:
        insights = run_all_insights(conn)
        missing = missing_value_report(conn, tables=("application_train",))
    render_all(insights, missing, get_settings().charts_dir)

    from src.ml.rules import build_and_save_rules
    build_and_save_rules()

    from src.api.main import app
    with TestClient(app) as test_client:
        yield test_client
    _reset_all_singletons()


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
def test_health_reports_not_ready_on_a_fresh_checkout(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["database_ready"] is False
    assert body["model_ready"] is False
    assert body["llm_ready"] is False


def test_health_reports_ready_once_trained(trained_client) -> None:
    body = trained_client.get("/health").json()
    assert body["database_ready"] is True
    assert body["model_ready"] is True
    assert body["llm_ready"] is False  # no key in these fixtures


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def test_eda_summary_returns_503_before_artifacts_exist(client) -> None:
    response = client.get("/api/eda/summary")
    assert response.status_code == 503
    assert "build_eda_artifacts" in response.json()["detail"]


def test_eda_summary_returns_real_data_once_built(trained_client) -> None:
    body = trained_client.get("/api/eda/summary").json()
    assert any(t["name"] == "application_train" for t in body["table_summary"])
    assert "demographic" in body["feature_categories"] or body["feature_categories"]


def test_eda_insights_include_chart_urls(trained_client) -> None:
    body = trained_client.get("/api/eda/insights").json()
    assert body["insights"]
    for insight in body["insights"]:
        assert insight["chart_url"].startswith("/api/eda/charts/")


def test_eda_chart_is_served_as_png(trained_client) -> None:
    insights = trained_client.get("/api/eda/insights").json()["insights"]
    chart_url = insights[0]["chart_url"]
    response = trained_client.get(chart_url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_eda_chart_rejects_path_traversal(trained_client) -> None:
    response = trained_client.get("/api/eda/charts/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code in (400, 404)


def test_eda_chart_rejects_non_png_extension(trained_client) -> None:
    response = trained_client.get("/api/eda/charts/model.txt")
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_score_returns_503_before_a_model_is_trained(client) -> None:
    response = client.post("/api/score", json={"raw_fields": {"EXT_SOURCE_MEAN": 0.5}})
    assert response.status_code == 503


def test_score_by_id_matches_the_real_applicant(trained_client) -> None:
    sample = trained_client.get("/api/applicants/sample?n=5").json()
    assert len(sample) == 5
    sk_id = sample[0]["sk_id_curr"]

    response = trained_client.post("/api/score", json={"sk_id_curr": sk_id})
    assert response.status_code == 200
    body = response.json()
    assert body["sk_id_curr"] == sk_id
    assert 0 <= body["probability"] <= 1
    assert body["risk_band"] in {"Low", "Medium", "High"}
    assert body["actual_target"] in {0, 1}


def test_score_by_id_404s_for_an_unknown_id(trained_client) -> None:
    response = trained_client.post("/api/score", json={"sk_id_curr": 9_999_999})
    assert response.status_code == 404


def test_score_raw_fields_scores_a_hand_entered_applicant(trained_client) -> None:
    response = trained_client.post(
        "/api/score", json={"raw_fields": {"EXT_SOURCE_MEAN": 0.8}}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sk_id_curr"] is None
    assert body["actual_target"] is None


def test_score_rejects_both_id_and_raw_fields(trained_client) -> None:
    response = trained_client.post(
        "/api/score",
        json={"sk_id_curr": 1, "raw_fields": {"x": 1}},
    )
    assert response.status_code == 422


def test_score_rejects_neither_id_nor_raw_fields(trained_client) -> None:
    response = trained_client.post("/api/score", json={})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Explain
# --------------------------------------------------------------------------- #
def test_explain_degrades_gracefully_without_an_llm_key(trained_client) -> None:
    sample = trained_client.get("/api/applicants/sample?n=1").json()
    sk_id = sample[0]["sk_id_curr"]

    response = trained_client.post("/api/explain", json={"sk_id_curr": sk_id})
    assert response.status_code == 200
    body = response.json()
    assert body["top_factors"]
    assert body["narrative"] is None
    assert body["narrative_error"] is not None


def test_explain_404s_for_an_unknown_id(trained_client) -> None:
    response = trained_client.post("/api/explain", json={"sk_id_curr": 9_999_999})
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def test_rules_returns_503_before_rules_are_derived(client) -> None:
    response = client.get("/api/rules")
    assert response.status_code == 503


def test_rules_returns_the_real_derived_rules(trained_client) -> None:
    body = trained_client.get("/api/rules").json()
    assert body["n_rules"] == len(body["rules"])
    assert body["n_rules"] > 0
    for rule in body["rules"]:
        assert "IF" in rule["sentence"]


# --------------------------------------------------------------------------- #
# Ask
# --------------------------------------------------------------------------- #
def test_ask_reports_llm_unavailable_without_a_key(trained_client) -> None:
    response = trained_client.post("/api/ask", json={"question": "What is the default rate?"})
    assert response.status_code == 200  # the chatbot degrades, it doesn't 500
    body = response.json()
    assert not body["refused"]
    assert body["error"] is not None
    assert "GOOGLE_API_KEY" in body["error"]


def test_ask_rejects_an_empty_question(trained_client) -> None:
    response = trained_client.post("/api/ask", json={"question": ""})
    assert response.status_code == 422
