"""Gemini client.

One thin wrapper around `google-genai`, so the rest of the platform never
touches the SDK directly and every call is measured the same way.

Four things this layer is responsible for:

**Token accounting.** Every call records the API's own reported usage rather
than an estimate. `docs/PROMPTS.md` is built from these numbers, so the
optimisation claims there are measured, not asserted.

**Determinism.** Temperature 0 and a fixed seed. An evaluator re-running a
question should get the same SQL, and a changing answer would make the
documented transcripts worthless.

**Thinking minimised.** Gemini 3 models reason before answering by default,
billed as output tokens. Measured against the live API: `thinking_budget=0`
(the Gemini 2.x way to disable it outright) is rejected outright on the 3.x
line with `400 INVALID_ARGUMENT` - there is no full "off" any more, only
`thinking_level` in {low, medium, high}. Both calls here set `low`, which on a
one-word answer still cost 58-77 thought tokens in testing. That floor is
priced into `docs/PROMPTS.md`'s token accounting rather than assumed away.

**Degrading, not crashing.** With no API key configured the client raises
`LLMUnavailable` rather than failing at import. The API still starts, EDA,
scoring, explanations and rules all keep working, and only the chatbot reports
that it needs a key.

**Resolution is reactive, not just listed.** `client.models.list()` overstates
what a key can actually call: on the key this was built against, `list()`
included `gemini-2.5-flash`, but generating with it 404'd as "no longer
available to new users." So `list()` is used only as a first, cheap filter;
the real check is the first successful `generate_content` call, and a model
that 404s at call time is struck from the candidates and the next one tried.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from src.utils.config import Settings, get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

#: Tried in order when the configured model turns out unusable for this key -
#: either absent from `models.list()`, or 404ing at generation time despite
#: being listed (see the module docstring). A free-tier or newly created key
#: may not reach every model, and failing the whole chatbot over that would be
#: a poor trade.
MODEL_FALLBACKS: dict[str, tuple[str, ...]] = {
    "sql": ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
            "gemini-flash-latest"),
    "summary": ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                "gemini-flash-lite-latest"),
}

#: Node types of google.genai errors that mean "this model id will never work
#: for this key" rather than "try again" - triggers advancing to the next
#: fallback candidate instead of a retry.
_MODEL_UNAVAILABLE_STATUSES = frozenset({"NOT_FOUND", "PERMISSION_DENIED"})

_RETRYABLE_TOKENS = (
    "429", "500", "502", "503", "504", "unavailable", "overloaded",
    "deadline", "timeout", "resource_exhausted",
)


class LLMUnavailable(RuntimeError):
    """No API key configured, or no usable model for this key."""


@dataclass
class Usage:
    """Token usage for one call, as reported by the API."""

    model: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thought_tokens": self.thought_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
        }


@dataclass
class LLMResponse:
    """Model output plus what it cost."""

    text: str
    usage: Usage


@dataclass
class UsageTotals:
    """Running totals for the process, surfaced on the API's /health endpoint."""

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def add(self, usage: Usage) -> None:
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.by_model[usage.model] = self.by_model.get(usage.model, 0) + usage.total_tokens


class GeminiClient:
    """Minimal, measured interface to Gemini."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = None
        self._lock = threading.Lock()
        self._resolved: dict[str, str] = {}
        self.totals = UsageTotals()

    # -- lifecycle ----------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._settings.llm_enabled

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.available:
            raise LLMUnavailable(
                "GOOGLE_API_KEY is not set. Add it to .env to enable the "
                "chatbot and generated explanations. Get a key at "
                "https://aistudio.google.com/apikey"
            )
        with self._lock:
            if self._client is None:
                from google import genai  # lazy: keeps import cost off startup

                self._client = genai.Client(api_key=self._settings.google_api_key)
                log.info("Gemini client initialised")
        return self._client

    def resolve_model(self, role: str) -> str:
        """The model id currently in use for `role`.

        Before the first successful call this is just the configured value -
        an intention, not a confirmed fact. `generate()` is what actually
        confirms a model works and updates this cache; see the module
        docstring for why a model appearing in `models.list()` is not enough
        to trust on its own.
        """
        if role in self._resolved:
            return self._resolved[role]
        return (
            self._settings.gemini_model_sql if role == "sql"
            else self._settings.gemini_model_summary
        )

    def _candidates(self, role: str) -> list[str]:
        """Ordered models to try for `role`: resolved first if we have one."""
        configured = (
            self._settings.gemini_model_sql if role == "sql"
            else self._settings.gemini_model_summary
        )
        ordered = [self._resolved[role]] if role in self._resolved else [configured]
        for candidate in MODEL_FALLBACKS.get(role, ()):
            if candidate not in ordered:
                ordered.append(candidate)
        return ordered

    # -- generation ---------------------------------------------------------
    def generate(
        self,
        *,
        role: str,
        system: str,
        user: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        stop_sequences: list[str] | None = None,
        max_attempts: int = 3,
    ) -> LLMResponse:
        """Generate, falling back across models, retrying transient failures.

        Two failure modes need different handling, and confusing them either
        wastes quota (retrying a model id that will never work) or gives up
        too early (treating a rate limit as permanent):

        - **Model unavailable** (404 / permission denied): this model id will
          never work for this key. Move to the next fallback candidate
          immediately, no retry.
        - **Transient** (429, 5xx, timeout): the model is fine, this attempt
          was not. Retry the same model with exponential backoff.
        """
        client = self._ensure_client()
        candidates = self._candidates(role)

        last_error: Exception | None = None
        for model in candidates:
            try:
                response, usage = self._generate_with_retry(
                    client, model, system, user, max_output_tokens,
                    temperature, stop_sequences, max_attempts,
                )
            except Exception as exc:
                if not self._is_model_unavailable(exc):
                    raise
                last_error = exc
                log.warning(
                    "model %r unavailable for this key (%s); trying next "
                    "candidate for %s", model, type(exc).__name__, role,
                )
                continue

            if model != self._resolved.get(role):
                log.info("using %s for %s", model, role)
            self._resolved[role] = model
            self.totals.add(usage)
            return LLMResponse(text=(response.text or "").strip(), usage=usage)

        raise LLMUnavailable(
            f"None of the {role} models are available to this API key. "
            f"Tried: {', '.join(candidates)}."
        ) from last_error

    def _generate_with_retry(
        self, client, model, system, user, max_output_tokens, temperature,
        stop_sequences, max_attempts,
    ):
        """One model, retried on transient failures only."""
        from google.genai import types  # lazy, as in _ensure_client

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=42,
            stop_sequences=stop_sequences or [],
            # Gemini 3.x has no full "off": thinking_budget=0, the Gemini 2.x
            # way to disable it, is rejected with 400 INVALID_ARGUMENT
            # (confirmed against the live API). `low` is the closest available
            # and is what docs/PROMPTS.md's token accounting is measured against.
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        )

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                response = client.models.generate_content(
                    model=model, contents=user, config=config
                )
            except Exception as exc:
                last_error = exc
                if self._is_model_unavailable(exc):
                    raise  # let the caller advance to the next candidate
                if not self._is_retryable(exc) or attempt == max_attempts:
                    raise
                backoff = 2 ** (attempt - 1)
                log.warning("Gemini call failed (%s), retrying in %ds [%d/%d]",
                            type(exc).__name__, backoff, attempt, max_attempts)
                time.sleep(backoff)
                continue

            usage = self._extract_usage(
                response, model, int((time.perf_counter() - started) * 1000)
            )
            log.debug(
                "Gemini %s: %d prompt + %d thought + %d output tokens in %dms",
                model, usage.prompt_tokens, usage.thought_tokens,
                usage.output_tokens, usage.latency_ms,
            )
            return response, usage

        raise RuntimeError("unreachable") from last_error

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(token in message for token in _RETRYABLE_TOKENS)

    @staticmethod
    def _is_model_unavailable(exc: Exception) -> bool:
        """True when this model id will never work for this key.

        Checked on typed attributes first (`google.genai.errors.APIError`
        carries `.code` and `.status`), falling back to the message text for
        any other exception shape the SDK might raise.
        """
        status = getattr(exc, "status", None)
        if status in _MODEL_UNAVAILABLE_STATUSES:
            return True
        code = getattr(exc, "code", None)
        if code in (404, 403):
            return True
        message = str(exc).lower()
        return "not_found" in message or "permission_denied" in message

    @staticmethod
    def _extract_usage(response, model: str, latency_ms: int) -> Usage:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return Usage(model=model, latency_ms=latency_ms)
        return Usage(
            model=model,
            prompt_tokens=meta.prompt_token_count or 0,
            output_tokens=meta.candidates_token_count or 0,
            thought_tokens=meta.thoughts_token_count or 0,
            cached_tokens=meta.cached_content_token_count or 0,
            total_tokens=meta.total_token_count or 0,
            latency_ms=latency_ms,
        )


_client: GeminiClient | None = None
_client_lock = threading.Lock()


def get_llm_client() -> GeminiClient:
    """Process-wide client. Cheap to call: the SDK is initialised on first use."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = GeminiClient()
    return _client
