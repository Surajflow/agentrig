"""Cost tracking, budgets, and context fitting -- before the invoice arrives.

Run:  python examples/03_cost_tracking.py
"""

from agentguard import CostTracker, Usage, estimate_tokens, fit_context, get_price, normalize_usage

# A realistic mix: a big model for planning, a cheap one for the bulk work.
RUNS = [
    ("claude-sonnet-4-5", "plan", {"input_tokens": 3200, "output_tokens": 900}),
    ("gpt-4o-mini", "extract", {"prompt_tokens": 18000, "completion_tokens": 2400}),
    ("gpt-4o-mini", "extract", {"prompt_tokens": 17500, "completion_tokens": 2200}),
    ("gpt-4o-mini", "summarise", {"prompt_tokens": 9000, "completion_tokens": 800}),
    ("gemini-2.5-flash", "classify", {"promptTokenCount": 12000, "candidatesTokenCount": 400}),
]

PROVIDER_OF = {
    "claude-sonnet-4-5": "anthropic",
    "gpt-4o-mini": "openai",
    "gemini-2.5-flash": "gemini",
}


def main():
    tracker = CostTracker(budget=0.05)

    for model, label, raw_usage in RUNS:
        usage = normalize_usage(PROVIDER_OF[model], raw_usage)
        entry = tracker.record(model, usage, label=label)
        print(
            f"{label:<10} {model:<18} {usage.total_tokens:>7,} tokens  "
            f"${entry.cost_usd:.6f}  (budget left ${tracker.remaining:.6f})"
        )

    print("\n== summary ==")
    for key, value in tracker.summary().items():
        if key not in ("by_model", "by_label"):
            print(f"  {key:<13} {value}")

    print("\n  by model")
    for model, row in tracker.by_model().items():
        print(f"    {model:<20} {row['calls']:>2} calls  {row['tokens']:>8,} tokens  ${row['cost_usd']:.6f}")

    print(f"\nbudget exceeded: {tracker.exceeded}")

    print("\n== would the next call fit the budget? ==")
    next_call = Usage(input_tokens=50_000, output_tokens=2_000)
    for model in ("gpt-4o-mini", "gpt-4o"):
        price_in, price_out = get_price(model)
        verdict = "no" if tracker.would_exceed(model, next_call) else "yes"
        print(
            f"  {model:<14} 50k in / 2k out -> ${next_call.cost(model):.4f} "
            f"(${price_in}/{price_out} per 1M)  fits: {verdict}"
        )

    print("\n== context fitting ==")
    messages = [{"role": "system", "content": "You are a support agent. Be terse."}]
    for i in range(30):
        messages.append({"role": "user", "content": f"Message {i}: " + "detail " * 60})
        messages.append({"role": "assistant", "content": f"Acknowledged message {i}."})

    before = sum(estimate_tokens(m["content"]) for m in messages)
    fitted, dropped = fit_context(messages, max_tokens=1200, reserve=300)
    after = sum(estimate_tokens(m["content"]) for m in fitted)
    print(f"  before: {len(messages)} messages, ~{before:,} tokens")
    print(f"  after:  {len(fitted)} messages, ~{after:,} tokens, dropped {dropped}")
    print(f"  system prompt kept: {fitted[0]['role'] == 'system'}")
    print(f"  newest message kept: {fitted[-1] == messages[-1]}")


if __name__ == "__main__":
    main()
