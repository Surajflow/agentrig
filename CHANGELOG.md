# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-11

First public release. Five modules, zero dependencies, 152 tests.

### Added

- **`agentrig.schema`** — `parse_json` extracts the first JSON value from fenced,
  prefixed or prose-wrapped model output (braces inside strings handled);
  `validate` covers the JSON Schema subset agents need with readable paths such
  as `$.line_items[2].amount`; `structured_call` re-prompts with the exact
  violations and returns a `StructuredResult` carrying `attempts` and `repaired`.
- **`agentrig.retry`** — `classify_error` sorts failures into `rate_limit`,
  `overloaded`, `timeout`, `transient`, `context_overflow`, `auth`,
  `invalid_request` and `unknown` from status codes or message text;
  `retry_call` backs off with full jitter and never retries an error that cannot
  heal; `RetryError` reports the attempt count and the original exception.
- **`agentrig.tokens`** — `Usage` with provider-neutral token counts and a
  `cost()` method; `CostTracker` with per-model and per-label breakdowns, budgets
  and `would_exceed()`; `estimate_tokens`, `trim_to_tokens` and `fit_context`
  (keeps the system prompt, drops the oldest turns).
- **`agentrig.tools`** — `@tool` turns type hints and a Google-style docstring
  into a JSON Schema; `ToolRegistry.specs()` renders OpenAI, Anthropic or Gemini
  declarations; `dispatch()` returns `{"error": ...}` instead of raising, so an
  agent loop can recover.
- **`agentrig.providers`** — `tool_spec`, `normalize_usage`, `extract_text` and
  `extract_tool_calls` translate between provider shapes without any SDK import.
- **`agentrig.trace`** — `Tracer` writes append-only JSONL with sequence numbers,
  run ids and span durations; `summary` and `replay` read it back.
- **CLI** — `agentrig validate | cost | models | trace | doctor`.
- Four offline examples and a GitHub Actions matrix across Python 3.9–3.13.

[0.1.0]: https://github.com/Surajflow/agentrig/releases/tag/v0.1.0
