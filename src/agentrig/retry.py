"""Retries that know *why* a call failed.

Blind retry loops waste money and time: a 401 will never succeed, and a
context-overflow needs a shorter prompt, not another attempt. This module
classifies the failure first, then backs off with full jitter (the AWS
recommendation) only for the errors that can actually heal.

    >>> from agentrig import retry_call
    >>> text = retry_call(lambda: call_model(prompt))

Works with any SDK: the classifier reads ``status_code`` / ``status`` /
``http_status`` attributes when they exist and falls back to matching the
error message.
"""

from __future__ import annotations

import functools
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

__all__ = [
    "RETRYABLE",
    "classify_error",
    "compute_delay",
    "RetryPolicy",
    "RetryError",
    "retry_call",
    "retryable",
]

#: Error kinds worth another attempt.
RETRYABLE = ("rate_limit", "overloaded", "timeout", "transient", "unknown")

_STATUS_KINDS = {
    408: "timeout",
    409: "transient",
    425: "transient",
    429: "rate_limit",
    500: "transient",
    502: "transient",
    503: "overloaded",
    504: "timeout",
    529: "overloaded",
    401: "auth",
    403: "auth",
    400: "invalid_request",
    404: "invalid_request",
    413: "context_overflow",
    422: "invalid_request",
}

# Order matters: the first pattern that matches wins.
_PATTERNS: Sequence[tuple[str, tuple[str, ...]]] = (
    (
        "context_overflow",
        (
            "context length",
            "context window",
            "maximum context",
            "too many tokens",
            "reduce the length",
            "prompt is too long",
            "input is too long",
            "token limit",
        ),
    ),
    (
        "auth",
        (
            "invalid api key",
            "invalid_api_key",
            "unauthorized",
            "authentication",
            "permission denied",
            "no api key",
            "api key not",
        ),
    ),
    (
        "rate_limit",
        (
            "rate limit",
            "rate_limit",
            "too many requests",
            "quota exceeded",
            "resource exhausted",
            "resourceexhausted",
            "429",
        ),
    ),
    (
        "overloaded",
        ("overloaded", "over capacity", "service unavailable", "503", "529"),
    ),
    (
        "timeout",
        ("timeout", "timed out", "deadline exceeded", "504"),
    ),
    (
        "transient",
        (
            "connection reset",
            "connection error",
            "connection aborted",
            "econnreset",
            "502",
            "500",
            "internal server error",
            "temporarily unavailable",
            "server error",
        ),
    ),
    (
        "invalid_request",
        ("invalid request", "bad request", "validation", "400", "unsupported"),
    ),
)


def classify_error(exc: BaseException) -> str:
    """Return one of the error kinds for ``exc``.

    Kinds: ``rate_limit``, ``overloaded``, ``timeout``, ``transient``,
    ``context_overflow``, ``auth``, ``invalid_request``, ``unknown``.
    """
    status = None
    for attr in ("status_code", "status", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            status = value
            break
        if isinstance(value, str) and value.isdigit():
            status = int(value)
            break
    if status is not None and status in _STATUS_KINDS:
        return _STATUS_KINDS[status]

    text = f"{type(exc).__name__} {exc}".lower()
    for kind, needles in _PATTERNS:
        if any(needle in text for needle in needles):
            return kind

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transient"
    return "unknown"


def compute_delay(
    attempt: int,
    policy: "RetryPolicy",
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff with full jitter, capped at ``policy.max_delay``."""
    if attempt < 1:
        attempt = 1
    raw = policy.base_delay * (policy.multiplier ** (attempt - 1))
    raw = min(raw, policy.max_delay)
    if policy.jitter:
        generator = rng or random
        return generator.uniform(0.0, raw)
    return raw


@dataclass(frozen=True)
class RetryPolicy:
    """How to retry. Immutable, so it is safe to share across threads."""

    max_attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: bool = True
    retry_kinds: tuple[str, ...] = RETRYABLE
    retry_if: Callable[[BaseException], bool] | None = None
    on_retry: Callable[[int, BaseException, str, float], None] | None = None

    def should_retry(self, exc: BaseException, kind: str) -> bool:
        if self.retry_if is not None:
            return bool(self.retry_if(exc))
        return kind in self.retry_kinds


class RetryError(RuntimeError):
    """Raised when every attempt was used up."""

    def __init__(self, attempts: int, last_error: BaseException, kind: str) -> None:
        self.attempts = attempts
        self.last_error = last_error
        self.kind = kind
        super().__init__(
            f"giving up after {attempts} attempt(s); last error was {kind}: {last_error}"
        )


def retry_call(
    fn: Callable[..., Any],
    *args: Any,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    **kwargs: Any,
) -> Any:
    """Call ``fn`` with backoff until it succeeds or the policy gives up.

    Non-retryable errors (auth, bad request, context overflow) are re-raised
    immediately and untouched. Exhausted retries raise :class:`RetryError`
    chained to the original exception.
    """
    policy = policy or RetryPolicy()
    last_error: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001 - classified, not swallowed
            kind = classify_error(exc)
            last_error = exc
            if not policy.should_retry(exc, kind):
                raise
            if attempt == policy.max_attempts:
                raise RetryError(attempt, exc, kind) from exc
            delay = compute_delay(attempt, policy, rng)
            if policy.on_retry is not None:
                policy.on_retry(attempt, exc, kind, delay)
            sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise RetryError(policy.max_attempts, last_error or RuntimeError("no error"), "unknown")


def retryable(
    policy: RetryPolicy | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form of :func:`retry_call`.

    >>> @retryable(RetryPolicy(max_attempts=3))
    ... def call_model(prompt): ...
    """
    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return retry_call(fn, *args, policy=policy, sleep=sleep, **kwargs)

        return wrapper

    return decorate
