# Design notes

Why `agentrig` looks the way it does. Read this before proposing a change — most
of the odd-looking decisions are load-bearing.

## 1. The model call is yours, not ours

Every function takes a callable that maps `messages -> text`. There is no client,
no `BaseProvider`, no registry of SDKs. Two consequences:

- **You can test anything offline.** The whole test suite runs with scripted
  strings; no keys, no network, no recorded fixtures to re-record.
- **Provider churn cannot break you.** When a vendor renames a field, you fix one
  line in your wrapper, not a dependency upgrade.

The cost is that `agentrig` cannot stream for you or manage your HTTP session. It
does not want to.

## 2. Retry decisions come from classification, not from the exception type

`classify_error` tries, in order: numeric status attributes, then message
patterns, then builtin exception types. This is deliberately unprincipled — it
inspects strings — because the alternative is importing every provider's
exception hierarchy, which breaks rule 1.

Two design choices follow:

- **`context_overflow` is not retried.** It looks like a failure but the fix is a
  smaller prompt. Retrying it wastes a full context window per attempt.
- **Full jitter, not decorrelated jitter.** `delay = uniform(0, min(cap, base *
  2^(n-1)))`. It is one line, and it is the variant AWS measured as best for
  spreading a synchronised retry storm. Tests pin the un-jittered sequence so the
  curve stays reviewable.

## 3. Schema validation covers the subset agents actually use

Full JSON Schema is a specification with `$ref`, `$defs`, `if/then`, `patternProperties`,
`unevaluatedProperties` and a resolution algorithm. Agents need `type`,
`required`, `properties`, `items`, `enum`, `anyOf` and bounds. Implementing the
rest badly would be worse than not implementing it, so unknown keywords are
**ignored**, not guessed at.

Two rules that avoid real bugs:

- **`bool` is not an `integer`.** Python says `isinstance(True, int)`; every
  model that emits `"age": true` would otherwise pass validation.
- **Errors carry paths.** `$.line_items[2].amount` tells the model exactly where
  to look, which is what makes the repair prompt work.

## 4. Dispatch returns errors instead of raising

An agent loop is a conversation. If a tool raises and the process dies, the model
never learns anything. If the tool returns `{"error": "no order ORD-9999"}`, the
model can apologise, try another id, or escalate.

So `dispatch()` catches unknown names, malformed JSON, schema violations and tool
exceptions and returns JSON error strings. `strict=True` restores raising for the
places where you want a real traceback (tests, scripts, background jobs).

The one thing it does not swallow: `KeyboardInterrupt` and `SystemExit` — those
are the user's, not the model's.

## 5. Cost is a table you own

Hardcoding prices in code is a bug with a delay fuse. The table is data:
`PRICES` is a plain dict, `register_price()` mutates it, and lookup falls back to
the **longest** matching prefix so `gpt-4o-2024-08-06` cannot resolve to
`gpt-4o-mini`.

Unknown models raise instead of defaulting to zero. A silently free model is how
a budget gets blown without anyone noticing.

Cached input is billed at 10% of the input rate. That is the common scheme, not a
universal one; the constant is `CACHE_READ_FACTOR` if your vendor differs.

## 6. Traces are files

One JSONL file per run, append-only, one JSON object per line. Not SQLite (needs
a schema, breaks `tail -f`), not OpenTelemetry-first (needs a collector to be
useful). `jq`, `grep` and `diff` already work on it, and `replay()` is a twenty
line function because of that choice.

Spans record duration in the `finally` block so a crash still leaves evidence,
and the exception is re-raised so tracing cannot change control flow.

## 7. What we would add, and what we would refuse

**Would add:** streaming spans, `oneOf`/`$ref` with tests, a Bedrock and Azure
response adapter, an optional OTLP exporter.

**Would refuse:** a bundled HTTP client, a prompt template library, an agent
framework, a hosted dashboard, or anything that makes `pip install agentrig` pull
a dependency. This package is at its best when it stays the boring layer that
never breaks.
