import os
import tempfile
import unittest

from agentguard.trace import Tracer, load, replay, summary


class TracerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "nested", "run.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_parent_directory(self):
        with Tracer(self.path) as tracer:
            tracer.event("started")
        self.assertTrue(os.path.exists(self.path))

    def test_records_have_sequence_and_run_id(self):
        with Tracer(self.path, run_id="abc123") as tracer:
            tracer.event("one")
            tracer.event("two")
        records = load(self.path)
        self.assertEqual([r["seq"] for r in records], [1, 2])
        self.assertTrue(all(r["run_id"] == "abc123" for r in records))
        self.assertTrue(all("ts" in r for r in records))

    def test_span_duration_and_fields(self):
        with Tracer(self.path) as tracer:
            with tracer.span("llm_call", model="gpt-4o-mini") as span:
                span["input_tokens"] = 100
                span["output_tokens"] = 20
        record = load(self.path)[-1]
        self.assertEqual(record["type"], "span")
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["model"], "gpt-4o-mini")
        self.assertEqual(record["input_tokens"], 100)
        self.assertGreaterEqual(record["duration_ms"], 0)

    def test_span_records_errors_and_reraises(self):
        with Tracer(self.path) as tracer:
            with self.assertRaises(ValueError):
                with tracer.span("bad_call"):
                    raise ValueError("model refused")
        record = load(self.path)[-1]
        self.assertEqual(record["status"], "error")
        self.assertIn("model refused", record["error"])

    def test_metadata_event(self):
        with Tracer(self.path, metadata={"app": "demo", "version": 1}):
            pass
        first = load(self.path)[0]
        self.assertEqual(first["name"], "run_started")
        self.assertEqual(first["app"], "demo")

    def test_append_mode(self):
        with Tracer(self.path) as tracer:
            tracer.event("first run")
        with Tracer(self.path, append=True) as tracer:
            tracer.event("second run")
        self.assertEqual(len(load(self.path)), 2)

    def test_write_mode_truncates(self):
        with Tracer(self.path) as tracer:
            tracer.event("first run")
        with Tracer(self.path) as tracer:
            tracer.event("second run")
        records = load(self.path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["name"], "second run")

    def test_non_serialisable_values_do_not_crash(self):
        with Tracer(self.path) as tracer:
            tracer.event("weird", payload={1, 2, 3})
        self.assertIn("payload", load(self.path)[0])

    def test_repr(self):
        tracer = Tracer(self.path, run_id="xyz")
        try:
            self.assertIn("xyz", repr(tracer))
        finally:
            tracer.close()


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Tracer(self.path, run_id="r1") as tracer:
            tracer.event("user_message")
            with tracer.span("llm_call", input_tokens=100, output_tokens=50, cost_usd=0.001):
                pass
            with tracer.span("tool_call"):
                pass
            try:
                with tracer.span("llm_call", input_tokens=10, output_tokens=5):
                    raise RuntimeError("429 rate limit")
            except RuntimeError:
                pass

    def tearDown(self):
        self.tmp.cleanup()

    def test_counts(self):
        stats = summary(self.path)
        self.assertEqual(stats["events"], 1)
        self.assertEqual(stats["spans"], 3)
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["runs"], ["r1"])

    def test_tokens_and_cost(self):
        stats = summary(self.path)
        self.assertEqual(stats["input_tokens"], 110)
        self.assertEqual(stats["output_tokens"], 55)
        self.assertEqual(stats["total_tokens"], 165)
        self.assertAlmostEqual(stats["cost_usd"], 0.001)

    def test_by_name(self):
        stats = summary(self.path)
        self.assertEqual(stats["by_name"]["llm_call"]["count"], 2)
        self.assertEqual(stats["by_name"]["llm_call"]["errors"], 1)
        self.assertEqual(stats["by_name"]["tool_call"]["count"], 1)

    def test_replay_lines(self):
        text = replay(self.path)
        lines = text.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("llm_call", text)
        self.assertIn("ERROR", text)
        self.assertIn("input_tokens=100", text)

    def test_iter_records_skips_corrupt_lines(self):
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write("this is not json\n\n")
        self.assertEqual(len(load(self.path)), 4)


if __name__ == "__main__":
    unittest.main()
