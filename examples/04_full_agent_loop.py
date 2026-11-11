"""A complete agent loop: tools + retry + structured output + cost + tracing.

Run:  python examples/04_full_agent_loop.py

The model is scripted, so this runs offline. Replace `ScriptedModel.chat` with
one HTTP call to your provider and the rest of the loop is unchanged.
"""

import json
from typing import Literal

from agentrig import (
    CostTracker,
    RetryPolicy,
    SchemaError,
    ToolRegistry,
    Tracer,
    Usage,
    retry_call,
    schema_instruction,
    structured_call,
    tool,
)

TICKETS = {
    "T-1": {"subject": "Refund not received", "age_days": 9, "amount_inr": 2499},
    "T-2": {"subject": "App crashes on login", "age_days": 2, "amount_inr": 0},
    "T-3": {"subject": "Wrong item delivered", "age_days": 21, "amount_inr": 899},
}

TRIAGE = {
    "type": "object",
    "required": ["priority", "category", "needs_human"],
    "properties": {
        "priority": {"enum": ["low", "medium", "high"]},
        "category": {"enum": ["billing", "bug", "logistics", "other"]},
        "needs_human": {"type": "boolean"},
        "reason": {"type": "string", "maxLength": 200},
    },
}


@tool
def get_ticket(ticket_id: str) -> dict:
    """Fetch one support ticket.

    Args:
        ticket_id: Ticket id such as "T-1".
    """
    ticket = TICKETS.get(ticket_id.upper())
    if ticket is None:
        raise LookupError(f"no ticket {ticket_id}")
    return {"ticket_id": ticket_id.upper(), **ticket}


@tool
def escalate(ticket_id: str, reason: str, severity: Literal["low", "high"] = "high") -> str:
    """Hand a ticket to a human agent.

    Args:
        ticket_id: Ticket to escalate.
        reason: One line explaining the escalation.
        severity: How urgent it is.
    """
    return f"escalated {ticket_id} ({severity}): {reason}"


class ScriptedModel:
    """Deterministic stand-in for an LLM that calls tools and then answers."""

    def __init__(self):
        self.turn = 0
        self.fail_once = True

    def chat(self, messages):
        """Returns (text, usage, tool_call|None)."""
        self.turn += 1
        if self.fail_once:
            self.fail_once = False
            raise ConnectionError("502 bad gateway")  # exercises the retry layer
        if self.turn == 2:
            return None, Usage(420, 60), {"name": "get_ticket", "arguments": {"ticket_id": "T-1"}}
        if self.turn == 3:
            return None, Usage(560, 80), {"name": "escalate", "arguments": {"ticket_id": "T-1", "reason": "refund 9 days overdue"}}
        if self.turn == 4:
            return '{"priority": "high", "category": "billing", "needs_human": true, "reason": "Refund overdue by 9 days"}', Usage(700, 90), None
        return "done", Usage(10, 5), None


def main():
    registry = ToolRegistry([get_ticket, escalate])
    model = ScriptedModel()
    costs = CostTracker(budget=0.01)
    history = [
        {"role": "system", "content": "Triage support tickets.\n\n" + schema_instruction(TRIAGE)},
        {"role": "user", "content": "Ticket T-1: customer says the refund never arrived."},
    ]

    with Tracer("runs/agent_loop.jsonl") as tracer:
        tracer.event("run_started", model="scripted", tools=registry.names())

        for step in range(6):
            with tracer.span("llm_call", step=step, model="scripted") as span:
                text, usage, call = retry_call(
                    lambda: model.chat(history),
                    policy=RetryPolicy(max_attempts=4, jitter=False, base_delay=0.01),
                    sleep=lambda _seconds: None,
                    )
                span["input_tokens"] = usage.input_tokens
                span["output_tokens"] = usage.output_tokens
                span["cost_usd"] = round(usage.cost("gpt-4o-mini"), 6)
            costs.record("gpt-4o-mini", usage, label=f"step-{step}")

            if call is not None:
                with tracer.span("tool_call", tool=call["name"]) as tool_span:
                    result = registry.dispatch(call["name"], call["arguments"])
                    tool_span["result_chars"] = len(result)
                print(f"tool {call['name']}({json.dumps(call['arguments'])}) -> {result}")
                history.append({"role": "assistant", "content": f"called {call['name']}"})
                history.append({"role": "user", "content": f"Tool result: {result}"})
                continue

            if text is None:
                continue

            if text.strip().startswith("{"):
                try:
                    result = structured_call(lambda _m, _reply=text: _reply, history, TRIAGE)
                except SchemaError as exc:
                    print("triage failed:", exc)
                    break
                print("\ntriage:", json.dumps(result.value, indent=2))
                tracer.event("triage_complete", priority=result.value["priority"])
                break

            print("model said:", text)
            break

        tracer.event("run_finished", cost_usd=round(costs.total_cost, 6))

    print("\ncost:", json.dumps(costs.summary()["cost_usd"]))
    print("trace: runs/agent_loop.jsonl  (inspect with: agentrig trace runs/agent_loop.jsonl --timeline)")


if __name__ == "__main__":
    main()
