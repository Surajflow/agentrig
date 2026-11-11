"""Turn plain Python functions into tools any model can call.

Write the function you would have written anyway. Type hints become the JSON
Schema, the docstring becomes the description, and :class:`ToolRegistry` gives
you provider-ready declarations plus safe dispatch.

    >>> from agentrig import tool, ToolRegistry
    >>> @tool
    ... def get_weather(city: str, units: str = "celsius") -> dict:
    ...     '''Current weather for a city.
    ...
    ...     Args:
    ...         city: City name, e.g. "Pune".
    ...         units: Either "celsius" or "fahrenheit".
    ...     '''
    ...     return {"city": city, "temp": 31, "units": units}
    >>> registry = ToolRegistry([get_weather])
    >>> registry.specs("anthropic")[0]["input_schema"]["required"]
    ['city']

Dispatch never raises by default: a bad argument list or a crashing tool comes
back as a JSON error string the model can read and recover from.
"""

from __future__ import annotations

import enum
import inspect
import json
import re
import types
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .schema import SchemaError, validate

__all__ = [
    "Tool",
    "ToolRegistry",
    "tool",
    "json_schema_from_signature",
    "parse_docstring",
]

_UNION_TYPES = (Union, getattr(types, "UnionType", Union))

_SECTION_ARGS = {"args", "arguments", "parameters", "params"}
_SECTION_OTHER = {
    "returns", "return", "raises", "raise", "yields", "example", "examples",
    "note", "notes", "attributes", "see also",
}


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into ``(summary, {param: description})``.

    Understands the Google style (``Args:`` block, one ``name: text`` per line,
    continuation lines indented under it). Other styles degrade to a summary
    with no parameter docs, which is harmless.
    """
    text = inspect.cleandoc(doc or "")
    summary: list[str] = []
    params: dict[str, str] = {}
    mode = "summary"

    for line in text.splitlines():
        stripped = line.strip()
        heading = stripped.lower().rstrip(":").strip()
        if heading in _SECTION_ARGS:
            mode = "args"
            continue
        if heading in _SECTION_OTHER:
            mode = "other"
            continue
        if mode == "summary":
            if stripped:
                summary.append(stripped)
        elif mode == "args":
            match = re.match(r"([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
            if match:
                name, description = match.group(1), match.group(2).strip()
                if description:
                    params[name] = description
                else:
                    params.setdefault(name, "")
            elif stripped and params:
                last = next(reversed(params))
                params[last] = (params[last] + " " + stripped).strip()

    return " ".join(summary).strip(), {k: v for k, v in params.items() if v}


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    """Best-effort JSON Schema for one Python annotation."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in _UNION_TYPES:
        real = [a for a in args if a is not type(None)]
        if len(real) == 1:
            return _annotation_schema(real[0])
        return {"anyOf": [_annotation_schema(a) for a in real]}

    if origin is Literal:
        return {"enum": list(args)}

    if origin in (list, set, frozenset, tuple, Sequence, List):
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _annotation_schema(args[0])}
        return {"type": "array", "items": _annotation_schema(args[0]) if args else {}}

    if origin in (dict, Dict, Mapping):
        return {"type": "object"}

    if isinstance(annotation, type):
        if issubclass(annotation, enum.Enum):
            return {"enum": [member.value for member in annotation]}
        if issubclass(annotation, bool):
            return {"type": "boolean"}
        if issubclass(annotation, int):
            return {"type": "integer"}
        if issubclass(annotation, float):
            return {"type": "number"}
        if issubclass(annotation, str):
            return {"type": "string"}
        if issubclass(annotation, (list, tuple, set, frozenset)):
            return {"type": "array"}
        if issubclass(annotation, dict):
            return {"type": "object"}

    return {}


def json_schema_from_signature(func: Callable[..., Any]) -> dict[str, Any]:
    """Build an object schema from ``func``'s signature and docstring."""
    try:
        hints = get_type_hints(func)
    except Exception:  # unresolved forward refs: fall back to raw annotations
        hints = {}
    signature = inspect.signature(func)
    _, doc_params = parse_docstring(inspect.getdoc(func))
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in ("self", "cls"):
            continue
        annotation = hints.get(name, parameter.annotation)
        schema = _annotation_schema(annotation)
        if name in doc_params:
            schema["description"] = doc_params[name]
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        elif parameter.default is not None:
            schema.setdefault("default", parameter.default)
        properties[name] = schema

    return {"type": "object", "properties": properties, "required": required}


@dataclass
class Tool:
    """A callable plus everything a model needs to decide to call it."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Provider-neutral description."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def spec(self, provider: str = "openai") -> dict[str, Any]:
        """Declaration in ``provider``'s own format."""
        from .providers import tool_spec

        return tool_spec(self, provider)

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r}, params={sorted(self.parameters.get('properties', {}))})"


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator: ``@tool`` or ``@tool(name="search", description="...")``."""
    def build(fn: Callable[..., Any]) -> Tool:
        summary, _ = parse_docstring(inspect.getdoc(fn))
        return Tool(
            name=name or fn.__name__,
            description=description or summary or f"Call {fn.__name__}.",
            parameters=json_schema_from_signature(fn),
            func=fn,
        )

    if func is not None:
        return build(func)
    return build


class ToolRegistry:
    """A dict of tools with provider specs and forgiving dispatch."""

    def __init__(self, tools: Iterable[Tool | Callable[..., Any]] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for item in tools or ():
            self.add(item)

    def add(self, item: Tool | Callable[..., Any]) -> Tool:
        """Register a :class:`Tool` or a bare function."""
        entry = item if isinstance(item, Tool) else tool(item)
        if entry.name in self._tools:
            raise ValueError(f"duplicate tool name {entry.name!r}")
        self._tools[entry.name] = entry
        return entry

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, provider: str = "openai") -> list[dict[str, Any]]:
        """All declarations, in ``provider``'s format."""
        return [t.spec(provider) for t in self._tools.values()]

    def dispatch(
        self,
        name: str,
        arguments: str | Mapping[str, Any] | None = None,
        *,
        strict: bool = False,
    ) -> str:
        """Run the named tool and return a string for the model.

        Unknown tools, malformed JSON, schema violations and tool exceptions all
        come back as ``{"error": "..."}`` so the agent loop can continue.
        Pass ``strict=True`` to raise instead.
        """
        entry = self._tools.get(name)
        if entry is None:
            message = f"unknown tool {name!r}; available: {', '.join(self.names()) or 'none'}"
            if strict:
                raise KeyError(message)
            return json.dumps({"error": message})

        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                message = f"arguments for {name!r} are not valid JSON: {exc}"
                if strict:
                    raise ValueError(message) from exc
                return json.dumps({"error": message})
        elif arguments is None:
            parsed = {}
        else:
            parsed = dict(arguments)

        try:
            validate(parsed, entry.parameters)
        except SchemaError as exc:
            message = f"invalid arguments for {name!r}: {exc}"
            if strict:
                raise
            return json.dumps({"error": message})

        try:
            result = entry.func(**parsed)
        except Exception as exc:  # noqa: BLE001 - reported to the model
            message = f"{type(exc).__name__}: {exc}"
            if strict:
                raise
            return json.dumps({"error": message})

        if isinstance(result, str):
            return result
        return json.dumps(result, default=str)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def __repr__(self) -> str:
        return f"ToolRegistry({self.names()})"
