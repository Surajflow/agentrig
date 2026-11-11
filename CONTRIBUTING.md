# Contributing to agentrig

Thanks for looking. This project stays useful only if it stays small, so the
rules below are strict on purpose.

## The three rules

1. **No runtime dependencies.** Standard library only. A PR that adds a
   `dependencies = [...]` entry will be declined — offer it as an optional extra
   instead (see `exact` in `pyproject.toml`).
2. **No provider lock-in.** Nothing in `src/agentrig/` may import a provider SDK.
   Provider differences belong in `providers.py` as plain dict translation.
3. **Every behaviour gets a test.** Standard-library `unittest`, no fixtures
   needed, no network, no API keys.

## Getting started

```bash
git clone https://github.com/Surajflow/agentrig
cd agentrig
pip install -e ".[dev]"
make test          # or: python -m unittest discover -s tests -v
make lint
make examples
```

## What a good PR looks like

- One concern per PR, with the *why* in the description: which real agent
  failure does this prevent?
- New public functions are exported from `__init__.py`, added to `__all__`, and
  documented in the README section they belong to.
- Error messages say what went wrong **and** what to do:
  `unknown model 'x'; add it with agentrig.register_price('x', input, output)`.
- Docstrings explain the trade-off, not just the signature.

## Ideas that would genuinely help

- A price-table refresh that reads provider pricing pages.
- Streaming spans (first-token latency is the number people actually want).
- More JSON Schema keywords — `oneOf`, `$ref`, `format: date-time` — with tests.
- Adapters for Bedrock and Azure OpenAI response shapes.

## Reporting a bug

Include the Python version, the smallest snippet that reproduces it, the
provider (or "scripted model"), and what you expected instead. If it is a
parsing or classification miss, paste the raw model output or error text — that
is the whole input to the failing code path.

## Code of conduct

Be decent. Critique code, never people. Harassment of any kind gets you removed.
