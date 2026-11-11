"""Structured output that fixes itself.

Run:  python examples/01_structured_output.py

No API key needed: the "model" is a stub that misbehaves on purpose, once.
Swap `fake_model` for your provider call and nothing else changes.
"""

from agentrig import SchemaError, schema_instruction, structured_call

INVOICE = {
    "type": "object",
    "required": ["vendor", "total", "currency"],
    "properties": {
        "vendor": {"type": "string", "minLength": 1},
        "total": {"type": "number", "minimum": 0},
        "currency": {"enum": ["INR", "USD", "EUR"]},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "amount"],
                "properties": {
                    "label": {"type": "string"},
                    "amount": {"type": "number"},
                },
            },
        },
    },
}

DOCUMENT = """
Invoice from Brightline Traders, dated 4 Nov 2026.
Consulting (8 hours)  Rs 40,000
Travel reimbursement Rs 2,500
Total due: Rs 42,500
"""


def fake_model(messages):
    """Stands in for a real LLM. Returns bad output the first time."""
    is_repair = any("could not be used" in str(m.get("content", "")) for m in messages)
    if is_repair:
        # Second attempt: the model has seen the exact validation errors.
        return """{"vendor": "Brightline Traders", "total": 42500, "currency": "INR",
                   "line_items": [{"label": "Consulting", "amount": 40000},
                                  {"label": "Travel reimbursement", "amount": 2500}]}"""
    # First attempt: chatty prose and a field the schema forbids.
    return """Sure! Here's what I found:

```json
{"vendor": "Brightline Traders", "total": "42,500", "currency": "Rs", "confidence": 0.9}
```"""


def main():
    messages = [
        {"role": "system", "content": "Extract invoice data.\n\n" + schema_instruction(INVOICE)},
        {"role": "user", "content": DOCUMENT},
    ]

    try:
        result = structured_call(
            fake_model,
            messages,
            INVOICE,
            on_attempt=lambda attempt, error: print(
                f"attempt {attempt}: " + ("ok" if error is None else f"rejected -> {error}")
            ),
        )
    except SchemaError as exc:
        print("model never produced valid JSON:", exc)
        return

    print(f"\nparsed in {result.attempts} attempt(s), repaired={result.repaired}")
    print(f"vendor   {result.value['vendor']}")
    print(f"total    {result.value['total']:,.2f} {result.value['currency']}")
    for item in result.value["line_items"]:
        print(f"  - {item['label']:<22} {item['amount']:>10,.2f}")


if __name__ == "__main__":
    main()
