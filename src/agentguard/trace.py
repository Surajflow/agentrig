"""Local-first tracing: one JSONL file you can read, diff and replay.

No server, no account, no vendor. Every run appends lines like::

    {"seq": 1, "ts": "...", "run_id": "9f2c", "type": "event", "name": "user_message"}
    {"seq": 2, "ts": "...", "run_id": "9f2c", "type": "span", "name": "llm_call",
     "status": "ok", "duration_ms": 812.4, "input_tokens": 1200, "output_tokens": 240}

That is enough to answer the questions that matter after a bad run: which step
was slow, which call failed, and what it cost.

    >>> with Tracer("runs/agent.jsonl") as tracer:
    ...     with tracer.span("llm_call", model="gpt-4o-mini"):
    ...         reply = call_model(prompt)
    ...     tracer.event("final_answer", chars=len(reply))
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

__all__ = ["Tracer", "load", "summary", "replay", "iter_records"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Tracer:
    """Append-only JSONL writer. One instance per run.

    Safe to share between threads: writes are serialised with a lock.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
        append: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        self._seq = 0
        self._handle = open(self.path, "a" if append else "w", encoding="utf-8")
        if metadata:
            self.event("run_started", **dict(metadata))

    # -- writing ----------------------------------------------------------
    def _write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            payload = {"seq": self._seq, "ts": _now(), "run_id": self.run_id}
            payload.update(record)
            self._handle.write(json.dumps(payload, default=str) + "\n")
            self._handle.flush()

    def event(self, name: str, **attributes: Any) -> None:
        """Record a point in time."""
        self._write({"type": "event", "name": name, **attributes})

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
        """Time a block. Errors are recorded and re-raised.

        The yielded dict is mutable, so you can attach fields discovered inside
        the block (tokens, model, tool name) before it closes.
        """
        record: dict[str, Any] = {"type": "span", "name": name, "status": "ok", **attributes}
        started = time.perf_counter()
        try:
            yield record
        except BaseException as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self._write(record)

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Tracer(path={str(self.path)!r}, run_id={self.run_id!r}, events={self._seq})"


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a trace file, skipping blank or corrupt lines."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load(path: str | Path) -> list[dict[str, Any]]:
    """Read a whole trace file."""
    return list(iter_records(path))


def summary(path: str | Path) -> dict[str, Any]:
    """Aggregate counts, timings, tokens, cost and errors for a trace file."""
    events = spans = errors = 0
    duration_ms = 0.0
    input_tokens = output_tokens = 0
    cost_usd = 0.0
    runs: list[str] = []
    by_name: dict[str, dict[str, Any]] = {}

    for record in iter_records(path):
        run_id = record.get("run_id")
        if run_id and run_id not in runs:
            runs.append(run_id)
        kind = record.get("type")
        if kind == "event":
            events += 1
        elif kind == "span":
            spans += 1
            took = float(record.get("duration_ms") or 0.0)
            duration_ms += took
            name = str(record.get("name"))
            row = by_name.setdefault(name, {"count": 0, "duration_ms": 0.0, "errors": 0})
            row["count"] += 1
            row["duration_ms"] = round(row["duration_ms"] + took, 3)
            if record.get("status") == "error":
                row["errors"] += 1
        if record.get("status") == "error" or record.get("error"):
            errors += 1
        input_tokens += int(record.get("input_tokens") or 0)
        output_tokens += int(record.get("output_tokens") or 0)
        cost_usd += float(record.get("cost_usd") or 0.0)

    return {
        "path": str(path),
        "runs": runs,
        "events": events,
        "spans": spans,
        "errors": errors,
        "duration_ms": round(duration_ms, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(cost_usd, 6),
        "by_name": by_name,
    }


def replay(path: str | Path) -> str:
    """Human-readable timeline, one line per record."""
    lines: list[str] = []
    for record in iter_records(path):
        kind = record.get("type", "?")
        name = record.get("name", "")
        extra: list[str] = []
        if "duration_ms" in record:
            extra.append(f"{record['duration_ms']}ms")
        if record.get("status") == "error":
            extra.append(f"ERROR {record.get('error', '')}".strip())
        for key in ("model", "tool", "input_tokens", "output_tokens", "cost_usd"):
            if key in record:
                extra.append(f"{key}={record[key]}")
        suffix = ("  " + " ".join(extra)) if extra else ""
        lines.append(f"{record.get('seq', 0):>4}  {kind:<5} {name}{suffix}")
    return "\n".join(lines)
