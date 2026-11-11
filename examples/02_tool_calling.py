"""Plain Python functions -> tool schemas for three providers, plus safe dispatch.

Run:  python examples/02_tool_calling.py

Type hints become the schema, the docstring becomes the description, and every
failure mode comes back as a JSON error the model can recover from.
"""

import json
from typing import Literal, Optional

from agentrig import ToolRegistry, extract_tool_calls, tool

INR_PER_USD = 88.0

ORDERS = {
    "ORD-1042": {"status": "shipped", "eta": "2026-11-14", "total_inr": 2499},
    "ORD-1043": {"status": "processing", "eta": "2026-11-17", "total_inr": 899},
}


@tool
def get_order(order_id: str) -> dict:
    """Look up one order by its id.

    Args:
        order_id: Order id such as "ORD-1042".
    """
    order = ORDERS.get(order_id.upper())
    if order is None:
        raise LookupError(f"no order {order_id}")
    return {"order_id": order_id.upper(), **order}


@tool
def refund_estimate(order_id: str, reason: Optional[Literal["damaged", "late", "changed_mind"]] = None) -> dict:
    """Estimate a refund for an order.

    Args:
        order_id: Order id to refund.
        reason: Why the customer wants the refund. Damaged orders get full value.
    """
    order = ORDERS[order_id.upper()]
    ratio = 1.0 if reason == "damaged" else 0.8 if reason == "late" else 0.5
    return {"order_id": order_id.upper(), "refund_inr": round(order["total_inr"] * ratio), "reason": reason}


@tool
def convert_currency(amount: float, to: Literal["INR", "USD"] = "INR") -> dict:
    """Convert an amount between INR and USD at a fixed demo rate.

    Args:
        amount: Amount to convert.
        to: Target currency.
    """
    if to == "USD":
        return {"amount": round(amount / INR_PER_USD, 2), "currency": "USD"}
    return {"amount": round(amount * INR_PER_USD, 2), "currency": "INR"}


def main():
    registry = ToolRegistry([get_order, refund_estimate, convert_currency])

    print("== what each provider sees ==\n")
    for provider in ("openai", "anthropic", "gemini"):
        spec = registry.specs(provider)[1]
        print(f"{provider}: {json.dumps(spec, indent=2)[:340]}...\n")

    print("== dispatch, including the failure modes ==\n")
    cases = [
        ("get_order", '{"order_id": "ORD-1042"}'),
        ("get_order", '{"order_id": "ORD-9999"}'),        # tool raises
        ("get_order", '{"orderid": "ORD-1042"}'),         # wrong argument name
        ("get_order", '{"order_id": '),                   # truncated JSON
        ("refund_estimate", {"order_id": "ORD-1042", "reason": "damaged"}),
        ("convert_currency", {"amount": 2499, "to": "USD"}),
        ("delete_everything", {}),                        # tool does not exist
    ]
    for name, arguments in cases:
        result = registry.dispatch(name, arguments)
        print(f"{name:<17} {result}")

    print("\n== reading a provider response ==\n")
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_order", "arguments": '{"order_id": "ORD-1043"}'},
                        }
                    ]
                }
            }
        ]
    }
    for call in extract_tool_calls("openai", response):
        print(f"model asked for {call['name']}({call['arguments']})")
        print("result:", registry.dispatch(call["name"], call["arguments"]))


if __name__ == "__main__":
    main()
