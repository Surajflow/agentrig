import enum
import json
import unittest
from typing import List, Literal, Optional

from agentguard.tools import (
    Tool,
    ToolRegistry,
    json_schema_from_signature,
    parse_docstring,
    tool,
)


class Unit(enum.Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


@tool
def get_weather(city: str, units: Optional[Literal["celsius", "fahrenheit"]] = None) -> dict:
    """Current weather for a city.

    Args:
        city: City name, e.g. "Pune".
        units: Unit system for the temperature.
            Defaults to celsius.
    """
    return {"city": city, "temp": 31, "units": units or "celsius"}


@tool
def search_docs(query: str, top_k: int = 3, tags: Optional[List[str]] = None) -> str:
    """Search the internal docs.

    Args:
        query: What to look for.
        top_k: How many results to return.
    """
    return f"{top_k} results for {query} (tags={tags})"


def plain_function(city: str) -> str:
    """Bare function without the decorator."""
    return city


class DocstringTests(unittest.TestCase):
    def test_summary_and_params(self):
        summary, params = parse_docstring(get_weather.func.__doc__)
        self.assertEqual(summary, "Current weather for a city.")
        self.assertEqual(params["city"], 'City name, e.g. "Pune".')
        self.assertIn("Unit system", params["units"])
        self.assertIn("Defaults to celsius.", params["units"])

    def test_no_docstring(self):
        summary, params = parse_docstring(None)
        self.assertEqual(summary, "")
        self.assertEqual(params, {})

    def test_returns_section_not_in_summary(self):
        summary, _ = parse_docstring("Do a thing.\n\nReturns:\n    something")
        self.assertEqual(summary, "Do a thing.")


class SchemaTests(unittest.TestCase):
    def test_required_and_defaults(self):
        schema = get_weather.parameters
        self.assertEqual(schema["required"], ["city"])
        self.assertEqual(schema["properties"]["city"]["type"], "string")
        self.assertEqual(schema["properties"]["units"]["enum"], ["celsius", "fahrenheit"])

    def test_list_annotation(self):
        schema = search_docs.parameters
        self.assertEqual(schema["properties"]["tags"]["type"], "array")
        self.assertEqual(schema["properties"]["tags"]["items"]["type"], "string")
        self.assertEqual(schema["properties"]["top_k"]["type"], "integer")
        self.assertEqual(schema["properties"]["top_k"]["default"], 3)

    def test_enum_annotation(self):
        def pick(unit: Unit) -> str:
            """Pick a unit."""
            return unit.value

        schema = json_schema_from_signature(pick)
        self.assertEqual(schema["properties"]["unit"]["enum"], ["celsius", "fahrenheit"])

    def test_no_annotations(self):
        def vague(x, y=2):
            """Vague."""
            return x

        schema = json_schema_from_signature(vague)
        self.assertEqual(schema["properties"]["x"], {})
        self.assertEqual(schema["required"], ["x"])

    def test_var_args_skipped(self):
        def variadic(a: int, *args, **kwargs):
            """Variadic."""
            return a

        schema = json_schema_from_signature(variadic)
        self.assertEqual(list(schema["properties"]), ["a"])

    def test_decorator_with_options(self):
        @tool(name="renamed", description="Custom description")
        def something(a: int) -> int:
            """Original summary."""
            return a

        self.assertEqual(something.name, "renamed")
        self.assertEqual(something.description, "Custom description")

    def test_default_description_from_docstring(self):
        self.assertEqual(get_weather.description, "Current weather for a city.")

    def test_repr(self):
        self.assertIn("get_weather", repr(get_weather))


class ProviderSpecTests(unittest.TestCase):
    def test_openai(self):
        spec = get_weather.spec("openai")
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], "get_weather")
        self.assertIn("properties", spec["function"]["parameters"])

    def test_anthropic(self):
        spec = get_weather.spec("anthropic")
        self.assertEqual(spec["name"], "get_weather")
        self.assertIn("input_schema", spec)

    def test_gemini_uppercases_types(self):
        spec = get_weather.spec("gemini")
        self.assertEqual(spec["parameters"]["type"], "OBJECT")
        self.assertEqual(spec["parameters"]["properties"]["city"]["type"], "STRING")

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            get_weather.spec("carrier-pigeon")


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry([get_weather, search_docs])

    def test_len_names_contains(self):
        self.assertEqual(len(self.registry), 2)
        self.assertIn("get_weather", self.registry)
        self.assertEqual(self.registry.names(), ["get_weather", "search_docs"])

    def test_add_bare_function(self):
        registry = ToolRegistry()
        entry = registry.add(plain_function)
        self.assertIsInstance(entry, Tool)
        self.assertIn("plain_function", registry)

    def test_duplicate_name(self):
        with self.assertRaises(ValueError):
            self.registry.add(get_weather)

    def test_specs_all(self):
        self.assertEqual(len(self.registry.specs("openai")), 2)
        self.assertEqual(len(self.registry.specs("anthropic")), 2)

    def test_dispatch_json_string(self):
        result = self.registry.dispatch("get_weather", '{"city": "Pune"}')
        self.assertEqual(json.loads(result)["city"], "Pune")

    def test_dispatch_dict(self):
        result = self.registry.dispatch("get_weather", {"city": "Delhi", "units": "celsius"})
        self.assertEqual(json.loads(result)["temp"], 31)

    def test_dispatch_string_result_passthrough(self):
        result = self.registry.dispatch("search_docs", {"query": "refunds"})
        self.assertEqual(result, "3 results for refunds (tags=None)")

    def test_unknown_tool_is_an_error_payload(self):
        payload = json.loads(self.registry.dispatch("nope", {}))
        self.assertIn("unknown tool", payload["error"])
        self.assertIn("get_weather", payload["error"])

    def test_unknown_tool_strict(self):
        with self.assertRaises(KeyError):
            self.registry.dispatch("nope", {}, strict=True)

    def test_invalid_json_arguments(self):
        payload = json.loads(self.registry.dispatch("get_weather", "{not json"))
        self.assertIn("not valid JSON", payload["error"])

    def test_schema_violation(self):
        payload = json.loads(self.registry.dispatch("get_weather", {}))
        self.assertIn("missing required property 'city'", payload["error"])

    def test_enum_violation(self):
        payload = json.loads(self.registry.dispatch("get_weather", {"city": "Pune", "units": "kelvin"}))
        self.assertIn("is not one of", payload["error"])

    def test_tool_exception_becomes_payload(self):
        @tool
        def explode(x: int) -> int:
            """Always fails.

            Args:
                x: ignored.
            """
            raise RuntimeError("boom")

        registry = ToolRegistry([explode])
        payload = json.loads(registry.dispatch("explode", {"x": 1}))
        self.assertEqual(payload["error"], "RuntimeError: boom")

    def test_strict_reraises(self):
        @tool
        def explode(x: int) -> int:
            """Always fails.

            Args:
                x: ignored.
            """
            raise RuntimeError("boom")

        registry = ToolRegistry([explode])
        with self.assertRaises(RuntimeError):
            registry.dispatch("explode", {"x": 1}, strict=True)

    def test_iteration(self):
        names = sorted(t.name for t in self.registry)
        self.assertEqual(names, ["get_weather", "search_docs"])

    def test_to_dict(self):
        payload = get_weather.to_dict()
        self.assertEqual(payload["name"], "get_weather")
        self.assertIn("parameters", payload)


if __name__ == "__main__":
    unittest.main()
