import random
import unittest

from agentguard.retry import (
    RETRYABLE,
    RetryError,
    RetryPolicy,
    classify_error,
    compute_delay,
    retry_call,
    retryable,
)


class FakeHTTPError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class ClassifyTests(unittest.TestCase):
    def test_status_code_wins(self):
        self.assertEqual(classify_error(FakeHTTPError("boom", 429)), "rate_limit")
        self.assertEqual(classify_error(FakeHTTPError("boom", 401)), "auth")
        self.assertEqual(classify_error(FakeHTTPError("boom", 503)), "overloaded")
        self.assertEqual(classify_error(FakeHTTPError("boom", 504)), "timeout")
        self.assertEqual(classify_error(FakeHTTPError("boom", 413)), "context_overflow")

    def test_message_matching(self):
        self.assertEqual(classify_error(Exception("Rate limit reached for gpt-4o")), "rate_limit")
        self.assertEqual(classify_error(Exception("prompt is too long: 210000 tokens")), "context_overflow")
        self.assertEqual(classify_error(Exception("Invalid API key provided")), "auth")
        self.assertEqual(classify_error(Exception("Server overloaded, try later")), "overloaded")
        self.assertEqual(classify_error(Exception("Connection reset by peer")), "transient")

    def test_builtin_exceptions(self):
        self.assertEqual(classify_error(TimeoutError("slow")), "timeout")
        self.assertEqual(classify_error(ConnectionError("down")), "transient")

    def test_unknown(self):
        self.assertEqual(classify_error(ValueError("odd number")), "unknown")


class DelayTests(unittest.TestCase):
    def test_exponential_no_jitter(self):
        policy = RetryPolicy(base_delay=1.0, multiplier=2.0, max_delay=10.0, jitter=False)
        self.assertEqual([compute_delay(i, policy) for i in (1, 2, 3, 4, 5)], [1.0, 2.0, 4.0, 8.0, 10.0])

    def test_jitter_within_bounds(self):
        policy = RetryPolicy(base_delay=2.0, multiplier=2.0, jitter=True)
        rng = random.Random(0)
        for _ in range(20):
            self.assertGreaterEqual(compute_delay(3, policy, rng), 0.0)
            self.assertLessEqual(compute_delay(3, policy, rng), 8.0)


class RetryCallTests(unittest.TestCase):
    def setUp(self):
        self.sleeps = []

    def _sleep(self, seconds):
        self.sleeps.append(seconds)

    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise FakeHTTPError("rate limit exceeded", 429)
            return "ok"

        policy = RetryPolicy(max_attempts=5, jitter=False, base_delay=0.01)
        self.assertEqual(retry_call(flaky, policy=policy, sleep=self._sleep), "ok")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(len(self.sleeps), 2)

    def test_auth_error_not_retried(self):
        calls = {"n": 0}

        def unauthorised():
            calls["n"] += 1
            raise FakeHTTPError("invalid api key", 401)

        with self.assertRaises(FakeHTTPError):
            retry_call(unauthorised, policy=RetryPolicy(max_attempts=4), sleep=self._sleep)
        self.assertEqual(calls["n"], 1)

    def test_exhaustion_raises_retry_error(self):
        def always_down():
            raise FakeHTTPError("service unavailable", 503)

        policy = RetryPolicy(max_attempts=3, jitter=False, base_delay=0.01)
        with self.assertRaises(RetryError) as ctx:
            retry_call(always_down, policy=policy, sleep=self._sleep)
        self.assertEqual(ctx.exception.attempts, 3)
        self.assertEqual(ctx.exception.kind, "overloaded")
        self.assertIsInstance(ctx.exception.last_error, FakeHTTPError)

    def test_passes_arguments(self):
        self.assertEqual(retry_call(lambda a, b=1: a + b, 2, b=3), 5)

    def test_on_retry_hook(self):
        seen = []
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise FakeHTTPError("timeout", 504)
            return "ok"

        policy = RetryPolicy(
            max_attempts=3,
            jitter=False,
            base_delay=0.01,
            on_retry=lambda attempt, exc, kind, delay: seen.append((attempt, kind)),
        )
        retry_call(flaky, policy=policy, sleep=self._sleep)
        self.assertEqual(seen, [(1, "timeout")])

    def test_custom_retry_if(self):
        calls = {"n": 0}

        def weird():
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("custom problem")
            return "ok"

        policy = RetryPolicy(max_attempts=3, retry_if=lambda exc: isinstance(exc, ValueError), jitter=False)
        self.assertEqual(retry_call(weird, policy=policy, sleep=self._sleep), "ok")

    def test_default_policy_is_retryable_set(self):
        self.assertIn("rate_limit", RETRYABLE)
        self.assertNotIn("auth", RETRYABLE)
        self.assertTrue(RetryPolicy().should_retry(Exception("x"), "transient"))
        self.assertFalse(RetryPolicy().should_retry(Exception("x"), "auth"))


class DecoratorTests(unittest.TestCase):
    def test_retryable_decorator(self):
        calls = {"n": 0}

        @retryable(RetryPolicy(max_attempts=3, jitter=False, base_delay=0.01), sleep=lambda _s: None)
        def flaky(x):
            calls["n"] += 1
            if calls["n"] == 1:
                raise FakeHTTPError("429 rate limit", 429)
            return x * 2

        self.assertEqual(flaky(21), 42)
        self.assertEqual(flaky.__name__, "flaky")


if __name__ == "__main__":
    unittest.main()
