"""Command line: validate JSON, price a call, inspect a trace, self-check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .schema import SchemaError, validate
from .tokens import PRICES, Usage, estimate_tokens, get_price
from .trace import replay, summary

__all__ = ["main", "build_parser"]


def _read_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _cmd_validate(args: argparse.Namespace) -> int:
    schema = _read_json(args.schema)
    data = _read_json(args.data)
    try:
        validate(data, schema)
    except SchemaError as exc:
        print("INVALID", file=sys.stderr)
        for problem in getattr(exc, "errors", [str(exc)]):
            print(f"  - {problem}", file=sys.stderr)
        return 1
    if not args.quiet:
        print("valid")
    return 0


def _cmd_cost(args: argparse.Namespace) -> int:
    try:
        price_in, price_out = get_price(args.model)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    usage = Usage(args.input, args.output, args.cached)
    cost = usage.cost(args.model)
    payload = {
        "model": args.model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "total_tokens": usage.total_tokens,
        "price_per_million": {"input": price_in, "output": price_out},
        "cost_usd": round(cost, 6),
        "cost_per_1k_calls_usd": round(cost * 1000, 4),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"{args.model}: {usage.total_tokens:,} tokens -> ${cost:.6f} USD")
        if args.calls > 1:
            print(f"  x{args.calls:,} calls -> ${cost * args.calls:,.2f} USD")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    rows = sorted(PRICES.items())
    if args.json:
        print(json.dumps({k: {"input": v[0], "output": v[1]} for k, v in rows}, indent=2))
        return 0
    width = max((len(name) for name, _ in rows), default=4)
    print(f"{'model'.ljust(width)}  {'input/1M':>9}  {'output/1M':>10}")
    for name, (price_in, price_out) in rows:
        print(f"{name.ljust(width)}  {price_in:>9.2f}  {price_out:>10.2f}")
    print("\nUSD per 1M tokens, indicative snapshot -- override with agentguard.register_price().")
    return 0


def _cmd_trace(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"no such trace file: {path}", file=sys.stderr)
        return 2
    stats = summary(path)
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0
    print(f"trace   {stats['path']}")
    print(f"runs    {len(stats['runs'])}")
    print(f"spans   {stats['spans']}   events {stats['events']}   errors {stats['errors']}")
    print(f"time    {stats['duration_ms']:.1f} ms")
    if stats["total_tokens"]:
        print(
            f"tokens  {stats['input_tokens']:,} in / {stats['output_tokens']:,} out"
            f"   cost ${stats['cost_usd']:.6f}"
        )
    if stats["by_name"]:
        print("\nby span")
        width = max(len(name) for name in stats["by_name"])
        for name, row in sorted(stats["by_name"].items(), key=lambda kv: -kv[1]["duration_ms"]):
            print(
                f"  {name.ljust(width)}  x{row['count']:<3} "
                f"{row['duration_ms']:>9.1f} ms  errors {row['errors']}"
            )
    if args.timeline:
        print("\ntimeline")
        print(replay(path))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    ok = True
    print(f"agentguard   {__version__}")
    print(f"python     {sys.version.split()[0]}")
    print(f"prices     {len(PRICES)} models")

    try:
        import tiktoken  # noqa: F401
    except Exception:
        print("tiktoken   not installed (optional: exact token counts)")
    else:
        print("tiktoken   installed (pass estimator= for exact counts)")

    checks = []

    from .schema import parse_json, structured_call

    try:
        assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
        checks.append(("json extraction", True))
    except Exception as exc:  # pragma: no cover
        checks.append((f"json extraction ({exc})", False))
        ok = False

    try:
        schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
        replies = iter(['nope', '{"a": 2}'])
        result = structured_call(lambda _msgs: next(replies), [{"role": "user", "content": "hi"}], schema)
        checks.append(("self-repair loop", result.value == {"a": 2} and result.attempts == 2))
    except Exception as exc:  # pragma: no cover
        checks.append((f"self-repair loop ({exc})", False))
        ok = False

    try:
        assert estimate_tokens("hello world, this is a test") > 0
        checks.append(("token estimate", True))
    except Exception as exc:  # pragma: no cover
        checks.append((f"token estimate ({exc})", False))
        ok = False

    for name, passed in checks:
        print(f"{'ok  ' if passed else 'FAIL'}       {name}")

    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentguard",
        description="Reliability tools for LLM agents: validate JSON, price calls, inspect traces.",
    )
    parser.add_argument("--version", action="version", version=f"agentguard {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="validate JSON data against a JSON Schema")
    p_validate.add_argument("schema", help="path to the schema JSON file")
    p_validate.add_argument("data", help="path to the data JSON file, or - for stdin")
    p_validate.add_argument("-q", "--quiet", action="store_true", help="print nothing on success")
    p_validate.set_defaults(func=_cmd_validate)

    p_cost = sub.add_parser("cost", help="price a single call")
    p_cost.add_argument("--model", required=True, help="model id, e.g. gpt-4o-mini")
    p_cost.add_argument("--input", type=int, default=0, help="input tokens")
    p_cost.add_argument("--output", type=int, default=0, help="output tokens")
    p_cost.add_argument("--cached", type=int, default=0, help="cached input tokens")
    p_cost.add_argument("--calls", type=int, default=1, help="multiply by this many calls")
    p_cost.add_argument("--json", action="store_true", help="machine-readable output")
    p_cost.set_defaults(func=_cmd_cost)

    p_models = sub.add_parser("models", help="list known model prices")
    p_models.add_argument("--json", action="store_true", help="machine-readable output")
    p_models.set_defaults(func=_cmd_models)

    p_trace = sub.add_parser("trace", help="summarise a JSONL trace file")
    p_trace.add_argument("path", help="path to the .jsonl trace")
    p_trace.add_argument("--json", action="store_true", help="machine-readable output")
    p_trace.add_argument("--timeline", action="store_true", help="also print the event timeline")
    p_trace.set_defaults(func=_cmd_trace)

    p_doctor = sub.add_parser("doctor", help="check the installation")
    p_doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
