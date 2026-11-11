"""Small adapters between providers' shapes and agentrig's neutral ones.

Every provider spells the same three things differently: tool declarations,
token usage, and where the text hides in a response. These functions translate
without importing any SDK, so they work offline and in tests.
"""

from __future__ import annotations

from typing import Any, Mapping

from .tokens import Usage

__all__ = [
    "PROVIDERS",
    "tool_spec",
    "normalize_usage",
    "extract_text",
    "extract_tool_calls",
]

#: Canonical provider ids accepted by the helpers below.
PROVIDERS = ("openai", "anthropic", "gemini", "openai_compatible")

_GEMINI_TYPES = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
    "null": "NULL",
}


def _gemini_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Gemini wants upper-case type names; the rest of the schema is the same."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out[key] = _GEMINI_TYPES.get(value, value.upper())
        elif key == "properties" and isinstance(value, Mapping):
            out[key] = {k: _gemini_schema(v) for k, v in value.items() if isinstance(v, Mapping)}
        elif key == "items" and isinstance(value, Mapping):
            out[key] = _gemini_schema(value)
        else:
            out[key] = value
    return out


def tool_spec(tool: Any, provider: str = "openai") -> dict[str, Any]:
    """Render one tool in ``provider``'s declaration format."""
    name = tool.name
    description = tool.description
    parameters = tool.parameters

    if provider == "openai":
        return {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        }
    if provider == "anthropic":
        return {"name": name, "description": description, "input_schema": parameters}
    if provider == "gemini":
        return {
            "name": name,
            "description": description,
            "parameters": _gemini_schema(parameters),
        }
    if provider == "openai_compatible":
        return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}
    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")


def normalize_usage(provider: str, payload: Mapping[str, Any] | None) -> Usage:
    """Convert a provider's usage block into :class:`~agentrig.tokens.Usage`."""
    data = dict(payload or {})

    if provider in ("openai", "openai_compatible"):
        details = data.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0) if isinstance(details, Mapping) else 0
        prompt = int(data.get("prompt_tokens", 0) or 0)
        return Usage(
            input_tokens=max(prompt - int(cached or 0), 0),
            output_tokens=int(data.get("completion_tokens", 0) or 0),
            cached_input_tokens=int(cached or 0),
        )

    if provider == "anthropic":
        return Usage(
            input_tokens=int(data.get("input_tokens", 0) or 0),
            output_tokens=int(data.get("output_tokens", 0) or 0),
            cached_input_tokens=int(data.get("cache_read_input_tokens", 0) or 0),
        )

    if provider == "gemini":
        return Usage(
            input_tokens=int(data.get("promptTokenCount", 0) or 0),
            output_tokens=int(data.get("candidatesTokenCount", 0) or 0),
            cached_input_tokens=int(data.get("cachedContentTokenCount", 0) or 0),
        )

    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")


def extract_text(provider: str, response: Mapping[str, Any]) -> str:
    """Pull the assistant's text out of a raw response dict."""
    if provider in ("openai", "openai_compatible"):
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) else ""

    if provider == "anthropic":
        blocks = response.get("content") or []
        return "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, Mapping) and block.get("type", "text") == "text"
        )

    if provider == "gemini":
        candidates = response.get("candidates") or []
        if not candidates:
            return ""
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        return "".join(part.get("text", "") for part in parts if isinstance(part, Mapping))

    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")


def extract_tool_calls(provider: str, response: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return tool calls as ``[{"id", "name", "arguments"}]`` with JSON arguments.

    ``arguments`` is always a dict, so you can hand it straight to
    :meth:`~agentrig.tools.ToolRegistry.dispatch`.
    """
    import json

    calls: list[dict[str, Any]] = []

    if provider in ("openai", "openai_compatible"):
        choices = response.get("choices") or []
        message = (choices[0].get("message") if choices else {}) or {}
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            raw = function.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {"_raw": raw}
            calls.append({"id": call.get("id"), "name": function.get("name"), "arguments": args})
        return calls

    if provider == "anthropic":
        for block in response.get("content") or []:
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                calls.append(
                    {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "arguments": dict(block.get("input") or {}),
                    }
                )
        return calls

    if provider == "gemini":
        candidates = response.get("candidates") or []
        parts = ((candidates[0].get("content") or {}).get("parts")) if candidates else []
        for index, part in enumerate(parts or []):
            function_call = part.get("functionCall") if isinstance(part, Mapping) else None
            if function_call:
                calls.append(
                    {
                        "id": f"call_{index}",
                        "name": function_call.get("name"),
                        "arguments": dict(function_call.get("args") or {}),
                    }
                )
        return calls

    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
