import unittest

from agentrig.tokens import (
    PRICES,
    CostTracker,
    Usage,
    estimate_tokens,
    fit_context,
    get_price,
    register_price,
    trim_to_tokens,
)


class EstimateTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens(None), 0)

    def test_roughly_four_chars_per_token(self):
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("a" * 400), 100)

    def test_non_string(self):
        self.assertGreater(estimate_tokens({"a": 1}), 0)

    def test_trim_to_tokens(self):
        text = " ".join(["word"] * 100)
        trimmed = trim_to_tokens(text, 20)
        self.assertLess(estimate_tokens(trimmed), estimate_tokens(text))
        self.assertLessEqual(estimate_tokens(trimmed), 25)
        self.assertFalse(trimmed.endswith(" "))
        self.assertEqual(trim_to_tokens("short", 100), "short")
        self.assertEqual(trim_to_tokens("anything", 0), "")


class PriceTests(unittest.TestCase):
    def test_exact_lookup(self):
        self.assertEqual(get_price("gpt-4o-mini"), PRICES["gpt-4o-mini"])

    def test_prefix_lookup_prefers_longest(self):
        self.assertEqual(get_price("gpt-4o-mini-2024-07-18"), PRICES["gpt-4o-mini"])
        self.assertEqual(get_price("gpt-4o-2024-08-06"), PRICES["gpt-4o"])

    def test_unknown_model_is_loud(self):
        with self.assertRaises(KeyError):
            get_price("totally-made-up-model")

    def test_register_price(self):
        register_price("test-model-xyz", 1.0, 2.0)
        try:
            self.assertEqual(get_price("test-model-xyz"), (1.0, 2.0))
        finally:
            PRICES.pop("test-model-xyz", None)

    def test_negative_price_rejected(self):
        with self.assertRaises(ValueError):
            register_price("bad-model", -1, 1)


class UsageTests(unittest.TestCase):
    def test_cost_math(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(usage.cost("gpt-4o"), 2.50 + 10.00, places=6)

    def test_small_call(self):
        usage = Usage(input_tokens=1000, output_tokens=500)
        self.assertAlmostEqual(usage.cost("gpt-4o-mini"), (1000 * 0.15 + 500 * 0.60) / 1e6, places=9)

    def test_cached_tokens_are_cheap(self):
        plain = Usage(input_tokens=1_000_000)
        cached = Usage(cached_input_tokens=1_000_000)
        self.assertLess(cached.cost("gpt-4o"), plain.cost("gpt-4o"))

    def test_add(self):
        total = Usage(1, 2, 3) + Usage(10, 20, 30)
        self.assertEqual((total.input_tokens, total.output_tokens, total.cached_input_tokens), (11, 22, 33))
        self.assertEqual(total.total_tokens, 66)

    def test_to_dict(self):
        self.assertEqual(
            Usage(1, 2).to_dict(),
            {"input_tokens": 1, "output_tokens": 2, "cached_input_tokens": 0, "total_tokens": 3},
        )


class CostTrackerTests(unittest.TestCase):
    def test_totals(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", Usage(1000, 500))
        tracker.record("gpt-4o-mini", input_tokens=2000, output_tokens=1000, label="summarise")
        self.assertEqual(tracker.calls, 2)
        self.assertEqual(tracker.total_tokens, 4500)
        self.assertAlmostEqual(tracker.total_cost, 0.00045 + 0.0009, places=9)

    def test_by_model_and_label(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", Usage(1000, 500), label="a")
        tracker.record("gpt-4o", Usage(1000, 500), label="b")
        self.assertEqual(set(tracker.by_model()), {"gpt-4o-mini", "gpt-4o"})
        self.assertEqual(tracker.by_model()["gpt-4o"]["calls"], 1)
        self.assertEqual(set(tracker.by_label()), {"a", "b"})

    def test_budget(self):
        tracker = CostTracker(budget=0.001)
        tracker.record("gpt-4o-mini", Usage(1000, 1000))
        self.assertLess(tracker.total_cost, 0.001)
        self.assertFalse(tracker.exceeded)
        tracker.record("gpt-4o-mini", Usage(1000, 1000))
        self.assertTrue(tracker.exceeded)
        self.assertLess(tracker.remaining, 0)

    def test_would_exceed(self):
        tracker = CostTracker(budget=0.01)
        self.assertTrue(tracker.would_exceed("gpt-4o", Usage(10_000, 0)))
        self.assertFalse(CostTracker().would_exceed("gpt-4o", Usage(10_000, 0)))

    def test_summary_and_reset(self):
        tracker = CostTracker(budget=1.0)
        tracker.record("claude-sonnet-4-5", Usage(1000, 1000))
        summary = tracker.summary()
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["budget_usd"], 1.0)
        self.assertIn("claude-sonnet-4-5", summary["by_model"])
        tracker.reset()
        self.assertEqual(tracker.calls, 0)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            CostTracker(budget=-1)


class FitContextTests(unittest.TestCase):
    def _messages(self, count, words=50):
        body = " ".join(["word"] * words)
        return [{"role": "system", "content": "You are terse."}] + [
            {"role": "user", "content": f"m{i} {body}"} for i in range(count)
        ]

    def test_keeps_system_and_newest(self):
        messages = self._messages(20)
        kept, dropped = fit_context(messages, max_tokens=200)
        self.assertGreater(dropped, 0)
        self.assertEqual(kept[0]["role"], "system")
        self.assertEqual(kept[-1]["content"], messages[-1]["content"])

    def test_no_drop_when_it_fits(self):
        messages = self._messages(2)
        kept, dropped = fit_context(messages, max_tokens=10_000)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), len(messages))

    def test_marker_inserted(self):
        kept, _ = fit_context(self._messages(20), max_tokens=200)
        self.assertEqual(kept[1]["content"], "[earlier messages omitted]")

    def test_reserve(self):
        messages = self._messages(10)
        kept, dropped = fit_context(messages, max_tokens=100, reserve=80)
        self.assertGreater(dropped, 0)
        self.assertLessEqual(len(kept), 3)

    def test_bad_budget(self):
        with self.assertRaises(ValueError):
            fit_context([], max_tokens=10, reserve=10)

    def test_keep_system_false(self):
        messages = self._messages(5)
        kept, _ = fit_context(messages, max_tokens=10_000, keep_system=False)
        self.assertEqual(len(kept), 6)

    def test_list_content(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}, "world"]},
        ]
        kept, dropped = fit_context(messages, max_tokens=1000)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()
