# agentguard

**The reliability layer for LLM agents. Zero dependencies, any model.**

[![CI](https://github.com/Surajflow/agentguard/actions/workflows/ci.yml/badge.svg)](https://github.com/Surajflow/agentguard/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dependencies: none](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](pyproject.toml)
[![Tests: 152](https://img.shields.io/badge/tests-152-brightgreen.svg)](tests)

Your agent works in the demo and falls apart in production, always for the same
five reasons:

| What breaks | What `agentguard` gives you |
| --- | --- |
| The model returns prose around your JSON, or a string where you wanted a number | [`structured_call`](#1-structured-output-that-repairs-itself) re-prompts with the exact validation errors until it parses |
| Your retry loop hammers a dead endpoint, or gives up on a 429 that would have healed | [`retry_call`](#2-retries-that-know-why) classifies the error first, then backs off only where it helps |
| Every provider wants tools declared differently | [`@tool`](#3-python-functions-as-tools) reads your type hints and emits OpenAI, Anthropic or Gemini schemas |
| Nobody knows the run cost ₹4 or ₹4,000 until the invoice | [`CostTracker`](#4-cost-and-context) prices every call against a table you control |
| A bad run cannot be reproduced | [`Tracer`](#5-tracing-you-can-replay) writes one JSONL file you can read, diff and replay |

It is plain Python, standard library only, and it does not import a single
provider SDK. Nothing to migrate: your model call stays your model call.

---

## Install

```bash
pip install agentguard
```

Working from a clone:

```bash
git clone https://github.com/Surajflow/agentguard
cd agentguard
pip install -e ".[dev]"
python -m unittest discover -s tests    # or: pytest
```

Every example runs offline, with a scripted model and no API key:

```bash
python examples/01_structured_output.py
python examples/02_tool_calling.py
python examples/03_cost_tracking.py
python examples/04_full_agent_loop.py
```

---

## 1. Structured output that repairs itself

```python
from agentguard import structured_call, schema_instruction

INVOICE = {
    "type": "object",
    "required": ["vendor", "total", "currency"],
    "properties": {
        "vendor": {"type": "string", "minLength": 1},
        "total": {"type": "number", "minimum": 0},
        "currency": {"enum": ["INR", "USD", "EUR"]},
    },
}

messages = [
    {"role": "system", "content": "Extract invoice data.\n\n" + schema_instruction(INVOICE)},
    {"role": "user", "content": document_text},
]

result = structured_call(call_model, messages, INVOICE)
print(result.value, result.attempts)   # {'vendor': 'Brightline Traders', ...} 2
```

When the first reply fails, `agentguard` appends the model's own output plus a
repair prompt naming every violation, and tries again:

```
attempt 1: rejected -> $.total: expected number, got string;
                       $.currency: 'Rs' is not one of ['INR', 'USD', 'EUR']
attempt 2: ok
```

Also included: `parse_json` (pulls the first real JSON value out of fenced,
prefixed or prose-wrapped text, braces inside strings included), `validate`
(JSON Schema subset with readable paths like `$.line_items[2].amount`), and
`is_valid` when you only need a boolean.

## 2. Retries that know *why*

```python
from agentguard import RetryPolicy, retry_call

policy = RetryPolicy(max_attempts=5, base_delay=0.5, max_delay=30, jitter=True)

reply = retry_call(lambda: client.messages.create(...), policy=policy)
```

Failures are classified before anything is retried:

| Kind | Retried? | Why |
| --- | --- | --- |
| `rate_limit`, `overloaded`, `timeout`, `transient`, `unknown` | yes | waiting is the fix |
| `auth` | no | a 401 never heals; retrying burns your quota |
| `invalid_request` | no | the payload is wrong |
| `context_overflow` | no | needs a shorter prompt, not another attempt |

The classifier reads `status_code` / `status` / `http_status` / `code` when your
SDK sets them and falls back to message matching, so it works with raw
`requests`, OpenAI, Anthropic, Gemini, Groq, Ollama and whatever you write next.
Delays use full jitter (exponential ceiling, uniform sample below it), which
spreads a retry storm instead of synchronising it.

## 3. Python functions as tools

```python
from typing import Literal, Optional
from agentguard import ToolRegistry, tool

@tool
def refund_estimate(order_id: str, reason: Optional[Literal["damaged", "late"]] = None) -> dict:
    """Estimate a refund for an order.

    Args:
        order_id: Order id to refund.
        reason: Why the customer wants the refund.
    """
    return {"order_id": order_id, "refund_inr": 2499}
```

Type hints become the JSON Schema, the docstring becomes the description and the
`Args:` block becomes per-parameter docs. Then:

```python
registry = ToolRegistry([refund_estimate])

registry.specs("openai")      # [{"type": "function", "function": {...}}]
registry.specs("anthropic")   # [{"name": ..., "input_schema": {...}}]
registry.specs("gemini")      # [{"name": ..., "parameters": {"type": "OBJECT", ...}}]

registry.dispatch("refund_estimate", '{"order_id": "ORD-1042"}')
# '{"order_id": "ORD-1042", "refund_inr": 2499}'
```

Dispatch is built for agent loops: it never raises by default. A missing
argument, a hallucinated tool name, truncated JSON and a tool that throws all
come back as `{"error": "..."}`, which is exactly what the model needs to
recover on the next turn.

```
get_order        {"error": "LookupError: no order ORD-9999"}
get_order        {"error": "invalid arguments for 'get_order': $: missing required property 'order_id'"}
delete_everything {"error": "unknown tool 'delete_everything'; available: convert_currency, get_order, refund_estimate"}
```

`extract_tool_calls(provider, response)` normalises tool calls from a raw
response dict into `{"id", "name", "arguments"}` with `arguments` already a dict.

## 4. Cost and context

```python
from agentguard import CostTracker, Usage, normalize_usage

tracker = CostTracker(budget=0.50)

usage = normalize_usage("openai", response["usage"])      # or anthropic / gemini
tracker.record("gpt-4o-mini", usage, label="extract")

tracker.total_cost      # 0.00045
tracker.by_label()      # {'extract': {'calls': 1, 'tokens': 1500, 'cost_usd': 0.00045}}
tracker.remaining       # 0.49955
tracker.would_exceed("gpt-4o", Usage(50_000, 2_000))   # True -> skip the call
```

`PRICES` ships as an editable table of USD-per-million-token rates for common
models; override or add your own with `register_price()`. Unknown models raise a
`KeyError` that tells you the exact line to add — silence is how cost bugs hide.

`estimate_tokens()` is a ±20% heuristic that never needs a tokenizer (pass a real
one anywhere an `estimator=` is accepted), and `fit_context()` trims the oldest
turns to fit a window while always keeping the system prompt:

```python
fitted, dropped = fit_context(messages, max_tokens=1200, reserve=300)
```

## 5. Tracing you can replay

```python
from agentguard import Tracer

with Tracer("runs/agent_loop.jsonl") as tracer:
    tracer.event("run_started", model="gpt-4o-mini")
    with tracer.span("llm_call", model="gpt-4o-mini") as span:
        reply = call_model(prompt)
        span["input_tokens"] = usage.input_tokens
        span["output_tokens"] = usage.output_tokens
```

One line per record, append-only, no server and no account:

```json
{"seq": 2, "ts": "2026-11-11T14:20:31.412+00:00", "run_id": "9f2c1a",
 "type": "span", "name": "llm_call", "status": "ok",
 "duration_ms": 812.4, "model": "gpt-4o-mini", "input_tokens": 1200, "output_tokens": 240}
```

Read it back with `summary(path)` (counts, timings, tokens, cost, errors per
span) or `replay(path)` for a timeline. Errors are recorded on the span *and*
re-raised, so tracing never changes your control flow.

---

## Command line

```bash
agentguard validate schema.json data.json    # exit 1 + readable errors when invalid
agentguard cost --model gpt-4o --input 1200000 --output 300000 --calls 1000
agentguard models                            # the price table
agentguard trace runs/agent_loop.jsonl --timeline
agentguard doctor                            # self-check: extraction, repair loop, pricing
```

```
$ agentguard trace runs/agent_loop.jsonl
trace   runs/agent_loop.jsonl
runs    1
spans   5   events 3   errors 0
time    41.2 ms
tokens  1,690 in / 235 out   cost $0.000398

by span
  llm_call    x3      39.8 ms  errors 0
  tool_call   x2       1.4 ms  errors 0
```

---

## What it deliberately does not do

No agent framework, no prompt templates, no vector store, no HTTP client, no
provider SDKs, no telemetry. `agentguard` is the layer *under* whatever you already
use — LangGraph, Pydantic AI, the OpenAI SDK, or a `while` loop you wrote
yourself. If you swap providers next month, none of this code changes.

**Honest limits:** the price table is an indicative snapshot, not a billing
source; `estimate_tokens` is a heuristic; the JSON Schema support is the useful
subset, not the full specification; and tracing writes local files — ship them to
your own backend if you need aggregation.

## Project layout

```
src/agentguard/
  schema.py     parse_json, validate, structured_call
  retry.py      classify_error, RetryPolicy, retry_call
  tokens.py     Usage, CostTracker, estimate_tokens, fit_context
  tools.py      tool, Tool, ToolRegistry
  providers.py  tool_spec, normalize_usage, extract_text, extract_tool_calls
  trace.py      Tracer, load, summary, replay
  cli.py        the agentguard command
tests/          152 tests, standard library only
examples/       four runnable, offline, no-API-key demos
docs/           quickstart and design notes
```

## Roadmap

- [ ] Streaming-aware spans (`Tracer.stream()`) for first-token latency
- [ ] `structured_call` with parallel best-of-N sampling
- [ ] Price table refresh command that reads provider pricing pages
- [ ] Optional OpenTelemetry exporter alongside the JSONL writer
- [ ] Bedrock and Azure OpenAI adapters in `providers.py`

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
The bar is simple: no new runtime dependencies, a test for every behaviour, and
nothing that assumes one provider.

## License

MIT — see [LICENSE](LICENSE).
