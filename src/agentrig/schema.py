"""Structured output for any LLM.

Model text is messy: it arrives wrapped in markdown fences, prefixed with
"Sure! Here is the JSON", or truncated mid-object. This module gives you
three small, dependency-free pieces:

* :func:`parse_json` -- pull the first real JSON value out of arbitrary text.
* :func:`validate` -- check it against a JSON Schema subset, with readable errors.
* :func:`structured_call` -- call the model, validate, and re-prompt with the
  exact errors when validation fails.

Nothing here knows which provider you use. ``call_fn`` is any callable that
takes a list of chat messages and returns the model's text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "SchemaError",
    "ValidationError",
    "parse_json",
    "validate",
    "is_valid",
    "schema_instruction",
    "repair_prompt",
    "StructuredResult",
    "structured_call",
]


class SchemaError(ValueError):
    """Raised when model output cannot be parsed as JSON."""


class ValidationError(SchemaError):
    """Raised when parsed JSON does not satisfy the schema."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)```", re.S)


def _balanced(text: str, start: int) -> str | None:
    """Return the balanced bracket run starting at ``start``, or None."""
    opener = text[start]
    closer = {"{": "}", "[": "]"}.get(opener)
    if closer is None:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json(text: Any) -> Any:
    """Extract the first JSON value from ``text``.

    Handles: clean JSON, JSON inside markdown fences, JSON with prose around it,
    and JSON containing braces inside strings. Raises :class:`SchemaError`
    when nothing parseable is found.
    """
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        raise SchemaError(f"expected str, got {type(text).__name__}")
    stripped = text.strip()
    if not stripped:
        raise SchemaError("empty response")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for match in _FENCE.finditer(stripped):
        body = match.group(1).strip()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            candidate = _balanced(body, body.find("{")) if "{" in body else None
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    for opener in ("{", "["):
        idx = stripped.find(opener)
        while idx != -1:
            candidate = _balanced(stripped, idx)
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            idx = stripped.find(opener, idx + 1)

    raise SchemaError(f"no JSON value found in response: {stripped[:120]!r}")


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

_TYPE_MAP: dict[str, Any] = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _type_ok(value: Any, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    expected = _TYPE_MAP.get(name)
    if expected is None:
        return True  # unknown type keyword: do not fail the caller
    return isinstance(value, expected)


def _validate(value: Any, schema: Any, path: str, errors: list[str]) -> None:
    if schema is True or schema is None:
        return
    if not isinstance(schema, Mapping):
        return
    if not schema:
        return

    if "anyOf" in schema:
        sub_errors: list[str] = []
        for sub in schema["anyOf"]:
            attempt: list[str] = []
            _validate(value, sub, path, attempt)
            if not attempt:
                sub_errors = []
                break
            sub_errors.append(" / ".join(attempt))
        else:
            errors.append(f"{path}: does not match any allowed schema ({'; '.join(sub_errors)})")
            return

    declared = schema.get("type")
    if declared is not None:
        names = [declared] if isinstance(declared, str) else list(declared)
        if not any(_type_ok(value, n) for n in names):
            errors.append(
                f"{path}: expected {' or '.join(names)}, got {_type_name(value)}"
            )
            return

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {list(schema['enum'])}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: string shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: string longer than {schema['maxLength']} characters")
        if "pattern" in schema:
            try:
                if re.search(schema["pattern"], value) is None:
                    errors.append(f"{path}: string does not match pattern {schema['pattern']!r}")
            except re.error:
                pass

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is less than minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is greater than maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {value} must be greater than {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: {value} must be less than {schema['exclusiveMaximum']}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for i, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{i}]", errors)

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        for key, sub in properties.items():
            if key in value:
                _validate(value[key], sub, f"{path}.{key}", errors)
        extra = set(value) - set(properties)
        additional = schema.get("additionalProperties", True)
        if additional is False and extra:
            errors.append(
                f"{path}: unexpected propert{'ies' if len(extra) > 1 else 'y'} "
                f"{', '.join(sorted(map(str, extra)))}"
            )
        elif isinstance(additional, Mapping):
            for key in extra:
                _validate(value[key], additional, f"{path}.{key}", errors)


def validate(value: Any, schema: Any, *, path: str = "$") -> Any:
    """Validate ``value`` against ``schema``; raise :class:`ValidationError`.

    Supports the JSON Schema subset agents actually need: ``type``, ``required``,
    ``properties``, ``items``, ``enum``, ``const``, ``anyOf``, numeric and string
    bounds, ``pattern``, ``additionalProperties``. Unsupported keywords are
    ignored rather than guessed at.
    """
    errors: list[str] = []
    _validate(value, schema, path, errors)
    if errors:
        raise ValidationError(errors)
    return value


def is_valid(value: Any, schema: Any) -> bool:
    """Boolean form of :func:`validate`."""
    errors: list[str] = []
    _validate(value, schema, "$", errors)
    return not errors


# --------------------------------------------------------------------------
# prompting
# --------------------------------------------------------------------------

def schema_instruction(schema: Mapping[str, Any]) -> str:
    """A ready-to-paste system instruction that pins the output shape."""
    body = json.dumps(schema, indent=2, sort_keys=True)
    return (
        "Respond with a single JSON value and nothing else. "
        "No prose before or after it, no markdown fences.\n"
        "The JSON must validate against this JSON Schema:\n"
        f"{body}"
    )


def repair_prompt(error: Exception, schema: Mapping[str, Any]) -> str:
    """Build the corrective message sent after a bad attempt."""
    details = getattr(error, "errors", None) or [str(error)]
    bullets = "\n".join(f"- {d}" for d in details)
    body = json.dumps(schema, indent=2, sort_keys=True)
    return (
        "Your previous reply could not be used. Problems:\n"
        f"{bullets}\n"
        "Return the corrected JSON only. It must validate against:\n"
        f"{body}"
    )


# --------------------------------------------------------------------------
# the retry loop
# --------------------------------------------------------------------------

@dataclass
class StructuredResult:
    """Outcome of :func:`structured_call`."""

    value: Any
    attempts: int
    raw: str
    repaired: bool = False

    def __iter__(self):
        # lets you write: data, attempts = structured_call(...)
        yield self.value
        yield self.attempts


def structured_call(
    call_fn: Callable[[list[dict[str, Any]]], str],
    messages: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    *,
    max_attempts: int = 3,
    validator: Callable[[Any], Any] | None = None,
    on_attempt: Callable[[int, Exception | None], None] | None = None,
) -> StructuredResult:
    """Call a model until its output parses and validates, or give up.

    ``call_fn`` receives the conversation and returns text. On a bad attempt the
    raw reply is appended as the assistant turn and a repair prompt as the next
    user turn, so the model sees exactly what was wrong. No hidden state, no
    provider SDKs.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    convo = [dict(m) for m in messages]
    check = validator or (lambda obj: validate(obj, schema))
    last_error: Exception | None = None
    raw = ""

    for attempt in range(1, max_attempts + 1):
        raw = call_fn(convo)
        if not isinstance(raw, str):
            raw = str(raw)
        try:
            value = parse_json(raw)
            check(value)
        except SchemaError as exc:
            last_error = exc
            if on_attempt is not None:
                on_attempt(attempt, exc)
            if attempt == max_attempts:
                break
            convo = convo + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair_prompt(exc, schema)},
            ]
            continue
        if on_attempt is not None:
            on_attempt(attempt, None)
        return StructuredResult(value=value, attempts=attempt, raw=raw, repaired=attempt > 1)

    raise SchemaError(
        f"structured output failed after {max_attempts} attempt(s): {last_error}"
    ) from last_error
