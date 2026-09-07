"""Talk-to-data route: one question in, one grounded answer out.

Thin by design - `src.talk_to_data.nl_to_sql.TalkToData` already owns the
entire pipeline (schema pruning, generation, validation, repair, execution,
summarisation) and is fully tested on its own; this route just adapts its
`Answer` dataclass to the HTTP response shape.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import AskRequest, AskResponse
from src.talk_to_data.nl_to_sql import ask as ask_question

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    answer = ask_question(request.question)
    return AskResponse(
        question=answer.question,
        answer=answer.answer,
        sql=answer.sql,
        columns=answer.columns,
        rows=answer.rows,
        row_count=answer.row_count,
        truncated=answer.truncated,
        refused=answer.refused,
        error=answer.error,
        repaired=answer.repaired,
        tables_used=answer.tables_used,
        total_tokens=answer.total_tokens,
        duration_ms=answer.duration_ms,
    )
