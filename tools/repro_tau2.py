"""Reproduce tau2's empty-response bug using a REAL tau2 prompt.

Requires tau2 installed in a separate venv and the data dir cloned.
The script uses tau2's own modules to build the first-turn text that
the green agent would send to our purple agent over A2A, then runs
that through our `_chat_loop` to see what we return.

Setup (one time):
    /opt/homebrew/bin/uv venv /tmp/repro-tau2 --python 3.12
    /tmp/repro-tau2/bin/python -m pip install "tau2 @ git+https://github.com/sierra-research/tau2-bench"
    git clone --depth 1 https://github.com/sierra-research/tau2-bench /tmp/tau2-bench-src

Run:
    export OPENROUTER_API_KEY=sk-or-v1-...
    export MODEL_NAME=z-ai/glm-5
    export JSON_OUTPUT=true              # match the leaderboard config
    /tmp/repro-tau2/bin/python tools/repro_tau2.py
"""

import asyncio
import json
import os
import sys

os.environ.setdefault("TAU2_DATA_DIR", "/tmp/tau2-bench-src/data")

# We use tau2 to build the prompt, but we run our agent's LLM call
# via the local agentswe-general source.
THIS_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(THIS_REPO, "src"))


def build_tau2_first_turn_text() -> str:
    """Reconstruct exactly what tau2-agentbeats sends on the first A2A turn.

    Mirrors RemoteA2AAgent.agent_prompt + the "Now here are the user
    messages:" suffix from src/agent.py:155-160 of tau2-agentbeats.
    """
    from tau2.domains.airline.environment import get_environment
    from tau2.registry import registry

    env = get_environment()
    tools = env.get_tools()
    tools_json = json.dumps([t.openai_schema for t in tools], indent=2)

    RESPOND_ACTION_NAME = "respond"
    respond_schema = json.dumps({
        "type": "function",
        "function": {
            "name": RESPOND_ACTION_NAME,
            "description": "Respond directly to the user with a message instead of calling a tool.",
            "parameters": {
                "properties": {
                    "content": {
                        "description": "The message content to send to the user.",
                        "title": "Content",
                        "type": "string",
                    }
                },
                "required": ["content"],
                "title": "parameters",
                "type": "object",
            },
        },
    }, indent=2)

    example1 = json.dumps({"name": "find_user_id_by_name_zip", "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip_code": "19122"}}, indent=2)
    example2 = json.dumps({"name": RESPOND_ACTION_NAME, "arguments": {"content": "Hello, how can I help you today?"}}, indent=2)

    agent_prompt = f"""{env.policy}

Here's a list of tools you can use (you can use at most one tool at a time):
{tools_json}

Additionally, you can respond to the user with the following call:

{respond_schema}


Please respond in JSON format.
The JSON should contain:
- "name": the tool call function name.
- "arguments": the arguments for the tool call.

You should only use one tool at a time!
You cannot respond to user and use a tool at the same time!
Tool calls are cheap and you should not hesitate to use them when necessary.
Most tasks will require you to use tools and respond to the user as part of the optimal solution.

Examples of responses:
{example1}

{example2}
"""

    tasks = registry.get_tasks_loader("airline")()
    task = tasks[0]
    # First user message from the user simulator — for repro we use the
    # persona instructions as a stand-in (real production would have the
    # user simulator generate this, but the instructions show the intent).
    initial_user = "Hi, I'd like to cancel my reservation EHGLP3, please."

    return f"{agent_prompt}\n\nNow here are the user messages:\n{initial_user}"


async def main():
    from llm import json_output, make_client, model_name

    print("=" * 70)
    print(f"MODEL_NAME : {os.environ.get('MODEL_NAME', '(unset)')}")
    print(f"JSON_OUTPUT: {os.environ.get('JSON_OUTPUT', '(unset)')}")
    print(f"OPENROUTER_API_KEY set: {bool(os.environ.get('OPENROUTER_API_KEY'))}")
    print("=" * 70)

    text = build_tau2_first_turn_text()
    print(f"\nPrompt length: {len(text):,} chars")
    print(f"Contains literal word 'JSON': {('JSON' in text or 'json' in text)}")
    print(f"\nPrompt head (first 200 chars):\n{text[:200]}")
    print(f"\nPrompt tail (last 300 chars):\n{text[-300:]}")

    client = make_client()
    kwargs = {
        "model": model_name(),
        "messages": [{"role": "user", "content": text}],
    }
    if json_output():
        kwargs["response_format"] = {"type": "json_object"}

    print(f"\nCalling chat.completions.create with kwargs keys: {sorted(kwargs.keys())}")
    print(f"Model resolved to: {kwargs['model']}")

    try:
        resp = await client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"\nLLM CALL FAILED: {type(e).__name__}: {e}")
        return

    print("\n" + "=" * 70)
    print("RAW RESPONSE")
    print("=" * 70)
    print(f"id            : {resp.id}")
    print(f"model (routed): {resp.model}")
    print(f"finish_reason : {resp.choices[0].finish_reason}")
    print(f"usage         : {resp.usage}")

    msg = resp.choices[0].message
    print(f"\nmsg.content (type={type(msg.content).__name__}, len={len(msg.content or '')}):")
    print(repr((msg.content or "")[:800]))
    print(f"\nmsg.tool_calls: {msg.tool_calls}")

    final = msg.content or ""
    print("\n" + "=" * 70)
    print("WHAT _chat_loop WOULD RETURN")
    print("=" * 70)
    print(f"final_text len: {len(final)}")
    print(f"first 500 chars: {repr(final[:500])}")


if __name__ == "__main__":
    asyncio.run(main())
