"""agentrig -- the reliability layer for LLM agents.

Five small modules, zero dependencies, no provider lock-in:

* :mod:`agentrig.schema`   structured output that repairs itself
* :mod:`agentrig.retry`    retries that understand *why* a call failed
* :mod:`agentrig.tokens`   token estimation, cost tracking, context fitting
* :mod:`agentrig.tools`    Python functions -> tool schemas -> safe dispatch
* :mod:`agentrig.trace`    local JSONL tracing you can replay

Quick start::

    from agentrig import structured_call, ToolRegistry, tool, CostTracker, Tracer

    def call_model(messages):          # your provider, any provider
        return my_llm(messages)        # returns text

    result = structured_call(call_model, messages, schema)
    print(result.value, result.attempts)
"""

from __future__ import annotations

from .providers import (
    PROVIDERS,
    extract_text,
    extract_tool_calls,
    normalize_usage,
    tool_spec,
)
from .retry import (
    RETRYABLE,
    RetryError,
    RetryPolicy,
    classify_error,
    compute_delay,
    retry_call,
    retryable,
)
from .schema import (
    SchemaError,
    StructuredResult,
    ValidationError,
    is_valid,
    parse_json,
    repair_prompt,
    schema_instruction,
    structured_call,
    validate,
)
from .tokens import (
    PRICES,
    CostTracker,
    Usage,
    estimate_tokens,
    fit_context,
    get_price,
    register_price,
    trim_to_tokens,
)
from .tools import Tool, ToolRegistry, json_schema_from_signature, parse_docstring, tool
from .trace import Tracer, load, replay, summary

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # schema
    "SchemaError",
    "ValidationError",
    "StructuredResult",
    "parse_json",
    "validate",
    "is_valid",
    "schema_instruction",
    "repair_prompt",
    "structured_call",
    # retry
    "RETRYABLE",
    "RetryError",
    "RetryPolicy",
    "classify_error",
    "compute_delay",
    "retry_call",
    "retryable",
    # tokens
    "PRICES",
    "Usage",
    "CostTracker",
    "estimate_tokens",
    "trim_to_tokens",
    "fit_context",
    "get_price",
    "register_price",
    # tools
    "Tool",
    "ToolRegistry",
    "tool",
    "json_schema_from_signature",
    "parse_docstring",
    # providers
    "PROVIDERS",
    "tool_spec",
    "normalize_usage",
    "extract_text",
    "extract_tool_calls",
    # trace
    "Tracer",
    "load",
    "replay",
    "summary",
]
