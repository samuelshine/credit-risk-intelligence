"""The talk-to-data pipeline: a question in, a grounded answer out.

    question
       -> schema card, pruned to this question
       -> Gemini writes SQL
       -> validated against the catalog        (reject -> one repair attempt)
       -> executed read-only, time and row bounded
       -> Gemini writes the answer from the returned rows only

Every stage is recorded on the `Answer` object - the SQL, the validation
errors, whether a repair happened, the row count, the token usage. The UI shows
that trail, which is what makes an answer auditable rather than something a
chatbot merely asserted.

Hallucination control, in the order it bites:

* The model only ever sees columns that exist, rendered from the live catalog.
* It is given an explicit way to refuse, so an unanswerable question does not
  have to be answered with an invented one.
* Generated SQL is parsed and checked against the catalog before it runs.
* A rejected query gets exactly one correction attempt, with the specific
  error. A second failure is reported as a failure - not retried until
  something happens to pass.
* The answer is written from returned rows only. The summarising model never
  sees the database, so it has nothing to invent from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.llm.gemini import GeminiClient, LLMUnavailable, Usage, get_llm_client
from src.talk_to_data.catalog import Catalog, get_catalog
from src.talk_to_data.prompt_templates import (
    PROMPT_VERSION,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT,
    build_repair_prompt,
    build_sql_prompt,
    is_refusal,
    refusal_reason,
)
from src.talk_to_data.query_runner import QueryResult, run_query
from src.talk_to_data.schema_card import render_schema_card
from src.talk_to_data.sql_validator import validate_sql
from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

#: SQL is short. Capping output stops a runaway generation from costing tokens,
#: and a query longer than this is a sign something has gone wrong anyway.
MAX_SQL_TOKENS = 600
MAX_SUMMARY_TOKENS = 400


@dataclass
class Answer:
    """One answered question, with the full trail behind it."""

    question: str
    answer: str = ""
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    refused: bool = False
    error: str | None = None
    repaired: bool = False
    validation_errors: list[str] = field(default_factory=list)
    tables_used: list[str] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION
    usage: list[dict[str, int | str]] = field(default_factory=list)
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and not self.refused

    @property
    def total_tokens(self) -> int:
        return sum(int(u.get("total_tokens", 0)) for u in self.usage)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "refused": self.refused,
            "error": self.error,
            "repaired": self.repaired,
            "validation_errors": self.validation_errors,
            "tables_used": self.tables_used,
            "prompt_version": self.prompt_version,
            "usage": self.usage,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
        }


class TalkToData:
    """Orchestrates the pipeline. Stateless between questions by design.

    No conversation memory: each question is answered from the schema and the
    database alone. Carrying prior turns would grow the prompt monotonically
    and let an earlier wrong answer contaminate a later one.
    """

    def __init__(
        self,
        client: GeminiClient | None = None,
        catalog: Catalog | None = None,
    ) -> None:
        self._client = client
        self._catalog = catalog

    @property
    def client(self) -> GeminiClient:
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            self._catalog = get_catalog()
        return self._catalog

    # -- public ------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        """Answer one natural-language question about the portfolio."""
        started = time.perf_counter()
        answer = Answer(question=question.strip())

        if not answer.question:
            answer.error = "Ask a question about the loan portfolio."
            return answer

        try:
            self._run(answer)
        except LLMUnavailable as exc:
            answer.error = str(exc)
        except Exception as exc:  # never surface a traceback to the UI
            log.exception("talk-to-data failed for %r", question)
            answer.error = f"Something went wrong answering that: {exc}"

        answer.duration_ms = int((time.perf_counter() - started) * 1000)
        return answer

    # -- pipeline ----------------------------------------------------------
    def _run(self, answer: Answer) -> None:
        settings = get_settings()
        card = render_schema_card(self.catalog, answer.question)
        system, user = build_sql_prompt(answer.question, card.text)

        log.info(
            "question=%r schema_card=~%d tokens, tables=%s",
            answer.question, card.approx_tokens, ",".join(card.tables_included),
        )

        generated = self.client.generate(
            role="sql", system=system, user=user,
            max_output_tokens=MAX_SQL_TOKENS,
        )
        answer.usage.append(generated.usage.as_dict())

        if is_refusal(generated.text):
            answer.refused = True
            answer.answer = refusal_reason(generated.text)
            log.info("model refused: %s", answer.answer)
            return

        validation = validate_sql(
            generated.text, self.catalog, max_rows=settings.max_sql_rows
        )

        if not validation.ok:
            answer.validation_errors = list(validation.errors)
            log.info("validation rejected the query: %s", validation.error_message)
            validation, repair_usage, refused = self._repair(
                answer, system, user, generated.text, validation.errors
            )
            if repair_usage is not None:
                answer.usage.append(repair_usage.as_dict())
            if refused:
                return
            if validation is None or not validation.ok:
                errors = validation.errors if validation else answer.validation_errors
                answer.error = (
                    "I could not write a valid query for that question. "
                    + (errors[0] if errors else "")
                ).strip()
                return
            answer.repaired = True

        answer.sql = validation.sql
        answer.tables_used = validation.tables_used

        result = run_query(
            validation.sql,
            timeout_seconds=settings.sql_timeout_seconds,
            max_rows=settings.max_sql_rows,
        )
        if not result.ok:
            answer.error = result.error
            return

        answer.columns = result.columns
        answer.rows = [list(row) for row in result.rows]
        answer.row_count = result.row_count
        answer.truncated = result.truncated

        answer.answer = self._summarise(answer, result, settings.max_summary_rows)

    def _repair(
        self,
        answer: Answer,
        system: str,
        user: str,
        bad_sql: str,
        errors: list[str],
    ):
        """One bounded correction attempt.

        Exactly one. Looping until something validates would eventually produce
        a query that passes the checks while answering a different question,
        and would burn the user's quota getting there.
        """
        repair_user = user + "\n\n" + build_repair_prompt(bad_sql, errors)
        try:
            repaired = self.client.generate(
                role="sql", system=system, user=repair_user,
                max_output_tokens=MAX_SQL_TOKENS,
            )
        except LLMUnavailable:
            raise
        except Exception as exc:
            log.warning("repair attempt failed: %s", exc)
            return None, None, False

        if is_refusal(repaired.text):
            answer.refused = True
            answer.answer = refusal_reason(repaired.text)
            return None, repaired.usage, True

        validation = validate_sql(
            repaired.text, self.catalog, max_rows=get_settings().max_sql_rows
        )
        if validation.ok:
            log.info("repair succeeded")
        else:
            log.info("repair still invalid: %s", validation.error_message)
        return validation, repaired.usage, False

    def _summarise(
        self, answer: Answer, result: QueryResult, max_rows: int
    ) -> str:
        """Turn rows into a sentence, using a cheaper model.

        Only `max_rows` rows are sent. The full result is already on screen;
        the model's job is to state what it shows, and paying to send 500 rows
        to have three sentences written about them would be waste.
        """
        truncation_note = ""
        if result.truncated:
            truncation_note = f", truncated to the first {result.row_count}"
        elif result.row_count > max_rows:
            truncation_note = f", showing the first {max_rows} to the summariser"

        user = SUMMARY_USER_PROMPT.format(
            question=answer.question,
            sql=answer.sql,
            row_count=result.row_count,
            truncation_note=truncation_note,
            results=result.to_markdown(max_rows=max_rows),
        )
        summary = self.client.generate(
            role="summary",
            system=SUMMARY_SYSTEM_PROMPT,
            user=user,
            max_output_tokens=MAX_SUMMARY_TOKENS,
        )
        answer.usage.append(summary.usage.as_dict())
        return summary.text or "The query ran but produced no description."


_service: TalkToData | None = None


def get_talk_to_data() -> TalkToData:
    global _service
    if _service is None:
        _service = TalkToData()
    return _service


def ask(question: str) -> Answer:
    return get_talk_to_data().ask(question)
