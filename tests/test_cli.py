import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from agentguard.cli import main
from agentguard.trace import Tracer


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class ValidateCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.schema_path = os.path.join(self.tmp.name, "schema.json")
        self.data_path = os.path.join(self.tmp.name, "data.json")
        with open(self.schema_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}},
                handle,
            )

    def tearDown(self):
        self.tmp.cleanup()

    def _write_data(self, payload):
        with open(self.data_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def test_valid(self):
        self._write_data({"name": "Asha"})
        code, out, _ = run("validate", self.schema_path, self.data_path)
        self.assertEqual(code, 0)
        self.assertIn("valid", out)

    def test_invalid(self):
        self._write_data({"nome": "Asha"})
        code, _, err = run("validate", self.schema_path, self.data_path)
        self.assertEqual(code, 1)
        self.assertIn("INVALID", err)
        self.assertIn("missing required property 'name'", err)

    def test_quiet(self):
        self._write_data({"name": "Asha"})
        code, out, _ = run("validate", self.schema_path, self.data_path, "--quiet")
        self.assertEqual(code, 0)
        self.assertEqual(out, "")


class CostCommandTests(unittest.TestCase):
    def test_cost_output(self):
        code, out, _ = run("cost", "--model", "gpt-4o", "--input", "1000000", "--output", "1000000")
        self.assertEqual(code, 0)
        self.assertIn("$12.500000", out)

    def test_cost_json(self):
        code, out, _ = run("cost", "--model", "gpt-4o-mini", "--input", "1000", "--output", "500", "--json")
        payload = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(payload["total_tokens"], 1500)
        self.assertAlmostEqual(payload["cost_usd"], 0.00045, places=9)

    def test_cost_multiplied(self):
        code, out, _ = run("cost", "--model", "gpt-4o", "--input", "1000000", "--calls", "10")
        self.assertIn("$25.00", out)

    def test_unknown_model(self):
        code, _, err = run("cost", "--model", "not-a-model", "--input", "10")
        self.assertEqual(code, 2)
        self.assertIn("register_price", err)


class ModelsCommandTests(unittest.TestCase):
    def test_table(self):
        code, out, _ = run("models")
        self.assertEqual(code, 0)
        self.assertIn("gpt-4o-mini", out)

    def test_json(self):
        code, out, _ = run("models", "--json")
        payload = json.loads(out)
        self.assertIn("claude-sonnet-4-5", payload)
        self.assertIn("input", payload["gpt-4o"])


class TraceCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Tracer(self.path) as tracer:
            tracer.event("user_message")
            with tracer.span("llm_call", model="gpt-4o-mini", input_tokens=100, output_tokens=20):
                pass

    def tearDown(self):
        self.tmp.cleanup()

    def test_summary(self):
        code, out, _ = run("trace", self.path)
        self.assertEqual(code, 0)
        self.assertIn("llm_call", out)
        self.assertIn("tokens", out)

    def test_json(self):
        code, out, _ = run("trace", self.path, "--json")
        payload = json.loads(out)
        self.assertEqual(payload["spans"], 1)
        self.assertEqual(payload["total_tokens"], 120)

    def test_timeline(self):
        code, out, _ = run("trace", self.path, "--timeline")
        self.assertIn("timeline", out)
        self.assertIn("user_message", out)

    def test_missing_file(self):
        code, _, err = run("trace", os.path.join(self.tmp.name, "nope.jsonl"))
        self.assertEqual(code, 2)
        self.assertIn("no such trace file", err)


class DoctorTests(unittest.TestCase):
    def test_doctor_passes(self):
        code, out, _ = run("doctor")
        self.assertEqual(code, 0)
        self.assertIn("ok", out)
        self.assertNotIn("FAIL", out)

    def test_version_flag(self):
        with self.assertRaises(SystemExit):
            run("--version")


if __name__ == "__main__":
    unittest.main()
