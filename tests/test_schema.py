import unittest

from agentguard.schema import (
    SchemaError,
    ValidationError,
    is_valid,
    parse_json,
    repair_prompt,
    schema_instruction,
    structured_call,
    validate,
)

PERSON = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "role": {"enum": ["admin", "member"]},
    },
    "additionalProperties": False,
}


class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_fence_without_language(self):
        self.assertEqual(parse_json("```\n[1, 2]\n```"), [1, 2])

    def test_prose_around_json(self):
        text = 'Sure! Here is the data:\n{"name": "Asha", "age": 30}\nHope that helps.'
        self.assertEqual(parse_json(text), {"name": "Asha", "age": 30})

    def test_braces_inside_strings(self):
        text = 'prefix {"note": "use {curly} braces", "n": 2} suffix'
        self.assertEqual(parse_json(text), {"note": "use {curly} braces", "n": 2})

    def test_already_parsed_passthrough(self):
        self.assertEqual(parse_json({"a": 1}), {"a": 1})

    def test_empty_raises(self):
        with self.assertRaises(SchemaError):
            parse_json("   ")

    def test_garbage_raises(self):
        with self.assertRaises(SchemaError):
            parse_json("I cannot help with that.")

    def test_truncated_json_raises(self):
        with self.assertRaises(SchemaError):
            parse_json('{"a": 1, "b": [2, 3')


class ValidateTests(unittest.TestCase):
    def test_valid_object(self):
        validate({"name": "Asha", "age": 30}, PERSON)

    def test_missing_required(self):
        with self.assertRaises(ValidationError) as ctx:
            validate({"name": "Asha"}, PERSON)
        self.assertIn("missing required property 'age'", str(ctx.exception))

    def test_wrong_type(self):
        with self.assertRaises(ValidationError) as ctx:
            validate({"name": "Asha", "age": "thirty"}, PERSON)
        self.assertIn("expected integer, got string", str(ctx.exception))

    def test_bool_is_not_integer(self):
        with self.assertRaises(ValidationError):
            validate({"name": "Asha", "age": True}, PERSON)

    def test_bounds(self):
        with self.assertRaises(ValidationError) as ctx:
            validate({"name": "Asha", "age": 200}, PERSON)
        self.assertIn("greater than maximum", str(ctx.exception))

    def test_enum(self):
        with self.assertRaises(ValidationError):
            validate({"name": "Asha", "age": 30, "role": "owner"}, PERSON)

    def test_additional_properties(self):
        with self.assertRaises(ValidationError) as ctx:
            validate({"name": "Asha", "age": 30, "extra": 1}, PERSON)
        self.assertIn("unexpected property", str(ctx.exception))

    def test_nested_array_items(self):
        with self.assertRaises(ValidationError) as ctx:
            validate({"name": "Asha", "age": 30, "tags": ["ok", 5]}, PERSON)
        self.assertIn("$.tags[1]", str(ctx.exception))

    def test_any_of(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        validate("x", schema)
        validate(3, schema)
        with self.assertRaises(ValidationError):
            validate(3.5, schema)

    def test_is_valid(self):
        self.assertTrue(is_valid({"name": "Asha", "age": 30}, PERSON))
        self.assertFalse(is_valid({}, PERSON))

    def test_pattern(self):
        schema = {"type": "string", "pattern": r"^[a-z]+$"}
        validate("abc", schema)
        with self.assertRaises(ValidationError):
            validate("ABC", schema)


class PromptTests(unittest.TestCase):
    def test_instruction_contains_schema(self):
        text = schema_instruction(PERSON)
        self.assertIn("single JSON", text)
        self.assertIn('"required"', text)

    def test_repair_prompt_lists_errors(self):
        try:
            validate({"name": "Asha"}, PERSON)
        except ValidationError as exc:
            text = repair_prompt(exc, PERSON)
        self.assertIn("missing required property 'age'", text)
        self.assertIn("corrected JSON", text)


class StructuredCallTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "user", "content": "Asha is 30"}]

    def test_first_attempt(self):
        result = structured_call(
            lambda _m: '{"name": "Asha", "age": 30}', self.messages, PERSON
        )
        self.assertEqual(result.value, {"name": "Asha", "age": 30})
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.repaired)

    def test_repairs_after_prose(self):
        replies = iter(["I think she is 30.", '{"name": "Asha", "age": 30}'])
        result = structured_call(lambda _m: next(replies), self.messages, PERSON)
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.repaired)

    def test_repair_message_is_sent_back(self):
        seen = []

        def call_fn(convo):
            seen.append(list(convo))
            return '{"name": "Asha"}' if len(seen) == 1 else '{"name": "Asha", "age": 30}'

        structured_call(call_fn, self.messages, PERSON)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1][-1]["role"], "user")
        self.assertIn("missing required property 'age'", seen[1][-1]["content"])
        self.assertEqual(seen[1][-2]["role"], "assistant")

    def test_gives_up(self):
        with self.assertRaises(SchemaError):
            structured_call(lambda _m: "nope", self.messages, PERSON, max_attempts=2)

    def test_attempt_hook(self):
        events = []
        replies = iter(["bad", '{"name": "Asha", "age": 30}'])
        structured_call(
            lambda _m: next(replies),
            self.messages,
            PERSON,
            on_attempt=lambda attempt, error: events.append((attempt, error is None)),
        )
        self.assertEqual(events, [(1, False), (2, True)])

    def test_unpacking(self):
        data, attempts = structured_call(
            lambda _m: '{"name": "Asha", "age": 30}', self.messages, PERSON
        )
        self.assertEqual(data["name"], "Asha")
        self.assertEqual(attempts, 1)

    def test_custom_validator(self):
        result = structured_call(
            lambda _m: '{"a": 1}',
            self.messages,
            {"type": "object"},
            validator=lambda obj: obj["a"] == 1 or (_ for _ in ()).throw(SchemaError("a must be 1")),
        )
        self.assertEqual(result.value, {"a": 1})

    def test_bad_max_attempts(self):
        with self.assertRaises(ValueError):
            structured_call(lambda _m: "{}", self.messages, PERSON, max_attempts=0)

    def test_original_messages_not_mutated(self):
        replies = iter(["bad", '{"name": "Asha", "age": 30}'])
        structured_call(lambda _m: next(replies), self.messages, PERSON)
        self.assertEqual(self.messages, [{"role": "user", "content": "Asha is 30"}])


if __name__ == "__main__":
    unittest.main()
