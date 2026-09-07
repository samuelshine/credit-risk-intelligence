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

**Thinking disabled.** Gemini 3 models reason before answering by default.
For NL-to-SQL that reasoning is billed output tokens spent re-deriving a
translation the few-shot examples already demonstrate, so `thinking_budget=0`
is set explicitly on both calls.

**Degrading, not crashing.** With no API key configured the client raises
`LLMUnavailable` rather than failing at import. The API still starts, EDA,
scoring, explanations and rules all keep working, and only the chatbot reports
that it needs a key.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from src.utils.config import Settings, get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

#: Tried in order when the configured model is unavailable to the key's tier.
#: A free-tier key may not reach the newest model, and failing the whole
#: chatbot over that would be a poor trade.
MODEL_FALLBACKS: dict[str, tuple[str, ...]] = {
    "sql": ("gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest",
            "gemini-2.5-flash"),
    "summary": ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                "gemini-flash-lite-latest", "gemini-2.5-flash-lite"),
}

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
        """Pick a usable model id for `role`, preferring the configured one.

        The key's tier decides what is reachable, so the available models are
        listed once and the first working candidate is cached. Logged, because
        an evaluator needs to know which model actually produced the results.
        """
        if role in self._resolved:
            return self._resolved[role]

        configured = (
            self._settings.gemini_model_sql if role == "sql"
            else self._settings.gemini_model_summary
        )
        candidates = [configured, *MODEL_FALLBACKS.get(role, ())]

        try:
            client = self._ensure_client()
            available = {
                m.name.removeprefix("models/")
                for m in client.models.list()
                if m.name
            }
        except LLMUnavailable:
            raise
        except Exception as exc:
            # Listing is a convenience. If it fails, trust the configuration
            # and let the actual call report a real error.
            log.warning("could not list Gemini models (%s); using %s",
                        exc, configured)
            self._resolved[role] = configured
            return configured

        for candidate in candidates:
            if candidate in available:
                if candidate != configured:
                    log.warning(
                        "configured model %r unavailable for this key; "
                        "using %r for %s", configured, candidate, role,
                    )
                else:
                    log.info("using %s for %s", candidate, role)
                self._resolved[role] = candidate
                return candidate

        raise LLMUnavailable(
            f"None of the {role} models are available to this API key. "
            f"Tried: {', '.join(candidates)}."
        )

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
        """One generation call, retried on transient failures only.

        Retries use exponential backoff and cover rate limits and 5xx. A
        malformed request or an auth failure is raised immediately - retrying
        those wastes the user's quota and their time.
        """
        from google.genai import types  # lazy, as above

        client = self._ensure_client()
        model = self.resolve_model(role)

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=42,
            stop_sequences=stop_sequences or [],
            # Billed output tokens spent re-deriving what the few-shot
            # examples already show. See docs/PROMPTS.md for the measurement.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
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
            self.totals.add(usage)
            text = (response.text or "").strip()
            log.debug("Gemini %s: %d prompt + %d output tokens in %dms",
                      model, usage.prompt_tokens, usage.output_tokens,
                      usage.latency_ms)
            return LLMResponse(text=text, usage=usage)

        raise RuntimeError("unreachable") from last_error

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(token in message for token in _RETRYABLE_TOKENS)

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
