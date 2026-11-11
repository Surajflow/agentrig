import json
import unittest

from agentrig.providers import (
    extract_text,
    extract_tool_calls,
    normalize_usage,
    tool_spec,
)
from agentrig.tokens import Usage
from agentrig.tools import Tool


def _tool():
    return Tool(
        name="get_weather",
        description="Weather lookup.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
        func=lambda city, days=1: {"city": city},
    )


class ToolSpecTests(unittest.TestCase):
    def test_openai_shape(self):
        spec = tool_spec(_tool(), "openai")
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["parameters"]["required"], ["city"])

    def test_anthropic_shape(self):
        spec = tool_spec(_tool(), "anthropic")
        self.assertEqual(sorted(spec), ["description", "input_schema", "name"])

    def test_gemini_shape(self):
        spec = tool_spec(_tool(), "gemini")
        self.assertEqual(spec["parameters"]["type"], "OBJECT")
        self.assertEqual(spec["parameters"]["properties"]["days"]["type"], "INTEGER")

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            tool_spec(_tool(), "nope")


class UsageTests(unittest.TestCase):
    def test_openai(self):
        usage = normalize_usage(
            "openai",
            {"prompt_tokens": 1000, "completion_tokens": 200, "prompt_tokens_details": {"cached_tokens": 400}},
        )
        self.assertEqual(usage, Usage(input_tokens=600, output_tokens=200, cached_input_tokens=400))
        self.assertEqual(usage.total_tokens, 1200)

    def test_anthropic(self):
        usage = normalize_usage(
            "anthropic",
            {"input_tokens": 900, "output_tokens": 100, "cache_read_input_tokens": 500},
        )
        self.assertEqual(usage, Usage(900, 100, 500))

    def test_gemini(self):
        usage = normalize_usage(
            "gemini",
            {"promptTokenCount": 800, "candidatesTokenCount": 80, "cachedContentTokenCount": 300},
        )
        self.assertEqual(usage, Usage(800, 80, 300))

    def test_missing_payload(self):
        self.assertEqual(normalize_usage("openai", None), Usage(0, 0, 0))

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            normalize_usage("nope", {})


class TextExtractionTests(unittest.TestCase):
    def test_openai(self):
        response = {"choices": [{"message": {"content": "hello"}}]}
        self.assertEqual(extract_text("openai", response), "hello")

    def test_openai_empty(self):
        self.assertEqual(extract_text("openai", {"choices": []}), "")

    def test_anthropic_joins_blocks(self):
        response = {"content": [{"type": "text", "text": "a"}, {"type": "tool_use"}, {"type": "text", "text": "b"}]}
        self.assertEqual(extract_text("anthropic", response), "ab")

    def test_gemini(self):
        response = {"candidates": [{"content": {"parts": [{"text": "hi"}, {"text": " there"}]}}]}
        self.assertEqual(extract_text("gemini", response), "hi there")

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            extract_text("nope", {})


class ToolCallExtractionTests(unittest.TestCase):
    def test_openai(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "Pune"}'}}
                        ]
                    }
                }
            ]
        }
        calls = extract_tool_calls("openai", response)
        self.assertEqual(calls, [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Pune"}}])

    def test_openai_bad_json_arguments(self):
        response = {"choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "{oops"}}]}}]}
        calls = extract_tool_calls("openai", response)
        self.assertEqual(calls[0]["arguments"], {"_raw": "{oops"})

    def test_anthropic(self):
        response = {"content": [{"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Pune"}}]}
        calls = extract_tool_calls("anthropic", response)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(calls[0]["arguments"], {"city": "Pune"})

    def test_gemini(self):
        response = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Pune"}}}]}}]}
        calls = extract_tool_calls("gemini", response)
        self.assertEqual(calls[0]["id"], "call_0")
        self.assertEqual(calls[0]["arguments"], {"city": "Pune"})

    def test_no_calls(self):
        self.assertEqual(extract_tool_calls("openai", {"choices": [{"message": {"content": "hi"}}]}), [])

    def test_arguments_are_json_ready(self):
        response = {
            "choices": [
                {"message": {"tool_calls": [{"id": "1", "function": {"name": "x", "arguments": json.dumps({"a": 1})}}]}}
            ]
        }
        calls = extract_tool_calls("openai", response)
        self.assertEqual(json.dumps(calls[0]["arguments"]), '{"a": 1}')


if __name__ == "__main__":
    unittest.main()
