# Quickstart

Five minutes from `pip install` to a hardened agent loop. Everything here runs
against any provider, and every snippet works offline if you swap the model call
for a stub.

## Install

```bash
pip install agentguard
agentguard doctor     # verifies JSON extraction, the repair loop and pricing
```

## Step 1 — wrap your model call

`agentguard` never talks to a provider for you. It hands you the conversation and
takes back text. That is the whole contract, so wrap whatever you already use:

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def call_model(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content
```

For Anthropic the same function is `response.content[0].text`; for Gemini,
`response.candidates[0].content.parts[0].text`. `agentguard.providers.extract_text`
does all three if you prefer to pass raw response dicts.

## Step 2 — get JSON you can trust

```python
from agentguard import structured_call, schema_instruction

TICKET = {
    "type": "object",
    "required": ["priority", "category"],
    "properties": {
        "priority": {"enum": ["low", "medium", "high"]},
        "category": {"enum": ["billing", "bug", "logistics", "other"]},
        "needs_human": {"type": "boolean"},
    },
}

messages = [
    {"role": "system", "content": "Triage support tickets.\n\n" + schema_instruction(TICKET)},
    {"role": "user", "content": "Refund still not received after 9 days."},
]

result = structured_call(call_model, messages, TICKET)
print(result.value["priority"], result.attempts)
```

Two details worth knowing:

- `structured_call` appends the failed reply **and** a repair prompt naming every
  violation, so the model can see its own mistake. That is why the second attempt
  usually works.
- The schema is also your documentation. If a field can be one of three strings,
  say `enum` — the validator will enforce it and the repair prompt will quote it.

## Step 3 — give the model tools

```python
from agentguard import ToolRegistry, tool

@tool
def get_ticket(ticket_id: str) -> dict:
    """Fetch one support ticket.

    Args:
        ticket_id: Ticket id such as "T-1".
    """
    return {"ticket_id": ticket_id, "age_days": 9}

registry = ToolRegistry([get_ticket])
tools = registry.specs("openai")     # pass to client.chat.completions.create(tools=...)
```

In your loop, feed the model's tool call straight back in:

```python
from agentguard import extract_tool_calls

for call in extract_tool_calls("openai", response.model_dump()):
    output = registry.dispatch(call["name"], call["arguments"])
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
```

`dispatch` returns an error string rather than raising, so a hallucinated tool
name costs one turn instead of the whole run.

## Step 4 — survive the network

```python
from agentguard import RetryPolicy, retry_call

reply = retry_call(
    lambda: call_model(messages),
    policy=RetryPolicy(max_attempts=5, base_delay=0.5, max_delay=30, jitter=True),
)
```

Auth and bad-request errors are raised immediately — retrying them only wastes
quota. Rate limits, overloads, timeouts and connection resets get exponential
backoff with jitter.

## Step 5 — see what it cost and what happened

```python
from agentguard import CostTracker, Tracer, normalize_usage

costs = CostTracker(budget=0.50)

with Tracer("runs/triage.jsonl") as tracer:
    with tracer.span("llm_call", model="gpt-4o-mini") as span:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        usage = normalize_usage("openai", response.usage.model_dump())
        span["input_tokens"] = usage.input_tokens
        span["output_tokens"] = usage.output_tokens
    costs.record("gpt-4o-mini", usage, label="triage")

print(costs.summary())
```

```bash
agentguard trace runs/triage.jsonl --timeline
```

## Putting it together

`examples/04_full_agent_loop.py` is the whole pattern in one file: a retried
model call, two tools, structured triage output, a cost ledger and a trace — all
offline. Read it, then replace the scripted model with your own.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `structured_call` never succeeds | the schema is stricter than the task, e.g. `maxLength` too small | read the last error; loosen the constraint |
| Repair loop burns tokens | `max_attempts` too high for a cheap model | 2–3 is plenty; log `on_attempt` to see why |
| `KeyError: unknown model` | model id not in the price table | `agentguard.register_price("your-model", input, output)` |
| Cost looks 10x too low | usage counted as cached | check `normalize_usage` input, not your arithmetic |
| Tool calls silently do nothing | arguments passed as a dict where a string was expected, or vice versa | both are accepted — check `registry.names()` matches the model's name |
