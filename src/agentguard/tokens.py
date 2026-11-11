"""Know what a run costs before the invoice does.

Three things agents usually get wrong about tokens:

1. They guess prices from memory -- so :data:`PRICES` is a small, editable table
   you can override with :func:`register_price`.
2. They count tokens with a tokenizer they did not install -- so
   :func:`estimate_tokens` is a heuristic that never fails, and you can pass
   your own estimator anywhere it matters.
3. They let one long conversation silently blow the context window -- so
   :func:`fit_context` trims the oldest turns while keeping the system prompt.

Prices below are an indicative snapshot in USD per **1 million tokens**.
Providers change them; check your provider's pricing page and override.
Cached-input discounts are modelled at 10% of the input rate, which matches
the common prompt-caching scheme but not every vendor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "PRICES",
    "register_price",
    "get_price",
    "estimate_tokens",
    "trim_to_tokens",
    "Usage",
    "CostTracker",
    "fit_context",
]

#: model id -> (input USD per 1M tokens, output USD per 1M tokens)
PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o4-mini": (1.10, 4.40),
    # Anthropic
    "claude-opus-4-1": (15.00, 75.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Google
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    # Open-weight models served by hosted providers
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "mistral-large-latest": (2.00, 6.00),
    "qwen3-32b": (0.29, 0.59),
}

#: Fraction of the input price charged for cache-read tokens.
CACHE_READ_FACTOR = 0.1


def register_price(model: str, input_per_million: float, output_per_million: float) -> None:
    """Add or override a model's price (USD per 1M tokens)."""
    if input_per_million < 0 or output_per_million < 0:
        raise ValueError("prices cannot be negative")
    PRICES[model] = (float(input_per_million), float(output_per_million))


def get_price(model: str) -> tuple[float, float]:
    """Look up ``(input, output)`` for ``model``.

    Exact match first, then the longest known prefix -- so
    ``gpt-4o-2024-08-06`` resolves to ``gpt-4o`` and never to ``gpt-4o-mini``.
    """
    key = model.strip().lower()
    if key in PRICES:
        return PRICES[key]
    for known in sorted(PRICES, key=len, reverse=True):
        if key.startswith(known) or known in key:
            return PRICES[known]
    raise KeyError(
        f"unknown model {model!r}; add it with "
        f"agentguard.register_price({model!r}, input, output)"
    )


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------

def estimate_tokens(text: Any) -> int:
    """Rough token count: about 4 characters per token for English.

    Deliberately conservative (rounds up). Expect +-20% error. Devanagari and
    other non-Latin scripts run closer to 2 characters per token, so pass
    ``estimator=`` with a real tokenizer when precision matters.
    """
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def trim_to_tokens(text: str, max_tokens: int, *, estimator: Callable[[Any], int] = estimate_tokens) -> str:
    """Cut ``text`` down to roughly ``max_tokens``, on a word boundary."""
    if max_tokens <= 0:
        return ""
    if estimator(text) <= max_tokens:
        return text
    words = text.split()
    while words and estimator(" ".join(words)) > max_tokens:
        words.pop()
    return " ".join(words)


# --------------------------------------------------------------------------
# usage and cost
# --------------------------------------------------------------------------

@dataclass
class Usage:
    """Token counts for one call, in a provider-neutral shape."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cached_input_tokens

    def cost(self, model: str, *, prices: Mapping[str, tuple[float, float]] | None = None) -> float:
        """Cost in USD for this usage on ``model``."""
        if prices is not None and model in prices:
            price_in, price_out = prices[model]
        else:
            price_in, price_out = get_price(model)
        cached = self.cached_input_tokens * CACHE_READ_FACTOR
        return (self.input_tokens * price_in + cached * price_in + self.output_tokens * price_out) / 1_000_000

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class CostEntry:
    model: str
    usage: Usage
    label: str = ""
    cost_usd: float = 0.0
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class CostTracker:
    """Running tally across many calls, optionally against a budget.

    >>> tracker = CostTracker(budget=0.50)
    >>> tracker.record("gpt-4o-mini", Usage(1000, 500))
    >>> tracker.total_cost
    0.00045
    >>> tracker.exceeded
    False
    """

    def __init__(self, budget: float | None = None) -> None:
        if budget is not None and budget < 0:
            raise ValueError("budget cannot be negative")
        self.budget = budget
        self.entries: list[CostEntry] = []

    def record(
        self,
        model: str,
        usage: Usage | None = None,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        label: str = "",
    ) -> CostEntry:
        """Add one call and return its entry."""
        if usage is None:
            usage = Usage(input_tokens, output_tokens, cached_input_tokens)
        entry = CostEntry(model=model, usage=usage, label=label, cost_usd=usage.cost(model))
        self.entries.append(entry)
        return entry

    @property
    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.usage.total_tokens for e in self.entries)

    @property
    def calls(self) -> int:
        return len(self.entries)

    @property
    def remaining(self) -> float | None:
        return None if self.budget is None else self.budget - self.total_cost

    @property
    def exceeded(self) -> bool:
        return self.budget is not None and self.total_cost > self.budget

    def would_exceed(self, model: str, usage: Usage) -> bool:
        """Check a hypothetical call against the budget before making it."""
        if self.budget is None:
            return False
        return self.total_cost + usage.cost(model) > self.budget

    def by_model(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for e in self.entries:
            row = out.setdefault(e.model, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
            row["calls"] += 1
            row["tokens"] += e.usage.total_tokens
            row["cost_usd"] += e.cost_usd
        return out

    def by_label(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for e in self.entries:
            row = out.setdefault(e.label or "-", {"calls": 0, "tokens": 0, "cost_usd": 0.0})
            row["calls"] += 1
            row["tokens"] += e.usage.total_tokens
            row["cost_usd"] += e.cost_usd
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "tokens": self.total_tokens,
            "cost_usd": round(self.total_cost, 6),
            "budget_usd": self.budget,
            "remaining_usd": None if self.remaining is None else round(self.remaining, 6),
            "exceeded": self.exceeded,
            "by_model": {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in self.by_model().items()},
            "by_label": {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in self.by_label().items()},
        }

    def reset(self) -> None:
        self.entries.clear()


# --------------------------------------------------------------------------
# context window management
# --------------------------------------------------------------------------

def _content_of(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                parts.append(str(part.get("text", part.get("content", ""))))
        return " ".join(parts)
    return str(content)


def fit_context(
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int,
    *,
    reserve: int = 0,
    keep_system: bool = True,
    estimator: Callable[[Any], int] = estimate_tokens,
    marker: str = "[earlier messages omitted]",
) -> tuple[list[dict[str, Any]], int]:
    """Drop the oldest turns until the conversation fits ``max_tokens``.

    System messages are always kept (they carry the instructions), and the
    newest turns are kept in order. Returns ``(messages, dropped_count)``.
    ``reserve`` leaves headroom for the model's own output.
    """
    budget = max_tokens - reserve
    if budget <= 0:
        raise ValueError("max_tokens must be greater than reserve")

    kept_system: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if keep_system and item.get("role") == "system":
            kept_system.append(item)
        else:
            rest.append(item)

    used = sum(estimator(_content_of(m)) for m in kept_system)
    tail: list[dict[str, Any]] = []
    dropped = 0
    for message in reversed(rest):
        cost = estimator(_content_of(message))
        if used + cost <= budget:
            tail.append(message)
            used += cost
        else:
            dropped += 1

    out = kept_system + list(reversed(tail))
    if dropped and out:
        insert_at = len(kept_system)
        out.insert(insert_at, {"role": "user", "content": marker})
    return out, dropped
