"""Live tests against the real Gemini API.

Skipped automatically when GOOGLE_API_KEY is not set - these make real network
calls and cost real (tiny) money, so they are not part of the default offline
suite. Run explicitly with a key present:

    PYTHONPATH=. .venv/bin/python -m pytest tests/test_gemini_live.py -v

What these exist to catch: SDK/API contract drift that no amount of stubbing
can - a `thinking_config` field the installed SDK version accepts but the API
rejects, a model id that stops resolving, a usage-metadata field renamed
upstream. Every one of the specific behaviours asserted below was discovered
by running against the real API during development, not from documentation.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set; skipping live Gemini tests",
)


@pytest.fixture(scope="module")
def client():
    from src.utils.config import get_settings
    get_settings.cache_clear()
    from src.llm.gemini import get_llm_client
    return get_llm_client()


def test_client_reports_available(client) -> None:
    assert client.available


def test_sql_and_summary_models_resolve_without_fallback(client) -> None:
    """The pinned model ids in .env.example must actually work.

    A silent fallback here would mean the documented model ids in the README
    are not what actually answers questions - worth failing loudly on.
    """
    from src.utils.config import get_settings

    settings = get_settings()
    response = client.generate(
        role="sql", system="Reply with exactly: OK", user="ping",
        max_output_tokens=200,
    )
    assert client.resolve_model("sql") == settings.gemini_model_sql

    response2 = client.generate(
        role="summary", system="Reply with exactly: OK", user="ping",
        max_output_tokens=200,
    )
    assert client.resolve_model("summary") == settings.gemini_model_summary
    assert response.usage.total_tokens > 0
    assert response2.usage.total_tokens > 0


def test_thinking_level_low_does_not_error(client) -> None:
    """Regression test for the 400 INVALID_ARGUMENT from thinking_budget=0.

    Gemini 3.x rejects thinking_budget entirely; only thinking_level survives.
    If a future SDK/API change breaks this again, this is where it shows up.
    """
    response = client.generate(
        role="summary", system="Answer in one word.", user="What is 2+2?",
        max_output_tokens=200,
    )
    assert response.text
    assert response.usage.total_tokens > 0


def test_usage_metadata_is_fully_populated(client) -> None:
    response = client.generate(
        role="sql", system="You write SQL.", user="Say: SELECT 1",
        max_output_tokens=200,
    )
    usage = response.usage
    assert usage.prompt_tokens > 0
    assert usage.total_tokens >= usage.prompt_tokens + usage.output_tokens
    assert usage.model  # a real resolved model id, not empty


def test_deterministic_with_fixed_seed(client) -> None:
    """Same prompt, same seed, same temperature - answers should match.

    This is the guarantee docs/PROMPTS.md's sample transcripts rely on: an
    evaluator re-running a documented question should see the same SQL.
    """
    kwargs = dict(
        role="sql",
        system="Output only: SELECT 1 + 1 AS two",
        user="Write the query.",
        max_output_tokens=100,
        temperature=0.0,
    )
    first = client.generate(**kwargs)
    second = client.generate(**kwargs)
    assert first.text == second.text
