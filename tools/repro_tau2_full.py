"""Full multi-turn tau2 reproduction against our local agent.

Spins up our purple agent on port 9011, drives ONE airline task end-to-end
using tau2-agentbeats's own RemoteA2AAgent + tau2's Orchestrator. Every
A2A round-trip is logged: outgoing prompt (truncated for readability),
raw response from our agent, and what tau2's parser made of it.

Prereqs (one-time):
  /opt/homebrew/bin/uv venv /tmp/repro-tau2 --python 3.12
  /opt/homebrew/bin/uv pip install --python /tmp/repro-tau2/bin/python \\
      "tau2 @ git+https://github.com/sierra-research/tau2-bench" \\
      "openai>=1.55.0" "httpx>=0.28.1" nest-asyncio \\
      "a2a-sdk[http-server]==0.3.20"
  git clone --depth 1 https://github.com/RDI-Foundation/tau2-agentbeats /tmp/tau2-agentbeats-src
  git clone --depth 1 https://github.com/sierra-research/tau2-bench /tmp/tau2-bench-src

Run:
  export OPENROUTER_API_KEY=sk-or-...   # for OUR agent's LLM calls
  export OPENAI_API_KEY=sk-...           # for tau2's user simulator (gpt-4o-mini)
  export MODEL_NAME=z-ai/glm-5
  export TOOLS_ENABLED=false
  export JSON_OUTPUT=true
  /tmp/repro-tau2/bin/python tools/repro_tau2_full.py [task_id]
"""

import json
import logging
import os
import subprocess
import sys
import time

THIS_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TAU2_DATA_DIR", "/tmp/tau2-bench-src/data")
sys.path.insert(0, "/tmp/tau2-agentbeats-src/src")

PORT = 9011
TASK_ID = sys.argv[1] if len(sys.argv) > 1 else "10"
DOMAIN = "airline"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def start_agent_server():
    """Launch src/server.py in the background, return the process."""
    print(f"[runner] starting agent server on :{PORT} ...")
    proc = subprocess.Popen(
        [
            "/opt/homebrew/bin/uv", "run", "python", "src/server.py",
            "--host", "127.0.0.1", "--port", str(PORT),
        ],
        cwd=THIS_REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    import httpx
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"agent server died:\n{out}")
        try:
            r = httpx.get(f"http://127.0.0.1:{PORT}/.well-known/agent-card.json", timeout=2)
            if r.status_code == 200:
                print(f"[runner] agent server ready")
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("agent server didn't come up in 60s")


def patch_messenger_for_logging():
    """Wrap Messenger.talk_to_agent so we log every send/receive."""
    from messenger import Messenger
    original = Messenger.talk_to_agent
    turn = {"n": 0}

    async def logged(self, message, url, new_conversation=False, timeout=300):
        turn["n"] += 1
        n = turn["n"]
        print(f"\n{'=' * 70}\nTURN {n} → SEND TO AGENT ({len(message)} chars)\n{'=' * 70}")
        head = message[:600]
        tail = message[-600:] if len(message) > 1200 else ""
        print(head + (f"\n... [truncated {len(message) - 1200} chars] ...\n{tail}" if tail else ""))
        try:
            response = await original(self, message, url, new_conversation, timeout)
        except Exception as e:
            print(f"\nTURN {n} ← talk_to_agent RAISED: {type(e).__name__}: {e}")
            raise
        print(f"\n{'=' * 70}\nTURN {n} ← RECEIVED FROM AGENT ({len(response or '')} chars)\n{'=' * 70}")
        print(repr(response)[:1200])
        return response

    Messenger.talk_to_agent = logged


def run_one_task():
    """Use tau2-agentbeats' wiring + tau2's orchestrator to run one task."""
    # tau2-agentbeats was written against an older tau2 layout; alias the
    # symbols it expects at tau2.agent.base back from tau2.agent.
    import tau2.agent as _tau2_agent
    import tau2.agent.base as _tau2_agent_base
    _tau2_agent_base.BaseAgent = _tau2_agent.BaseAgent
    _tau2_agent_base.ValidAgentInputMessage = _tau2_agent.ValidAgentInputMessage

    from agent import RemoteA2AAgent  # tau2-agentbeats' green-side wrapper
    from messenger import Messenger
    from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.run import get_tasks
    from tau2.user.user_simulator import UserSimulator

    tasks = get_tasks(task_set_name=DOMAIN, task_split_name="base", task_ids=[TASK_ID])
    if not tasks:
        raise RuntimeError(f"task {TASK_ID} not found in {DOMAIN}")
    task = tasks[0]
    print(f"\n[runner] task {task.id}, persona instructions:")
    print(str(task.user_scenario)[:500])

    env = registry.get_env_constructor(DOMAIN)(solo_mode=False)
    agent = RemoteA2AAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        messenger=Messenger(),
        agent_url=f"http://127.0.0.1:{PORT}",
    )
    user = UserSimulator(
        tools=env.get_user_tools() if env.user_tools else None,
        instructions=str(task.user_scenario),
        llm="openai/gpt-4o-mini",
        llm_args={"temperature": 1.0},
    )
    orchestrator = Orchestrator(
        domain=DOMAIN,
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=200,  # matches tau2-agentbeats README default
        max_errors=10,
        seed=42,
        solo_mode=False,
        validate_communication=False,
    )

    print(f"\n[runner] running orchestrator ...")
    simulation = orchestrator.run()
    print(f"\n[runner] orchestrator terminated: {simulation.termination_reason}")

    try:
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.ACTION,
            solo_mode=False,
            domain=DOMAIN,
        )
        print(f"[runner] reward: {reward_info.reward}")
    except Exception as e:
        print(f"[runner] evaluation failed: {e}")


def main():
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        if not os.environ.get(key):
            print(f"WARNING: {key} is not set")

    proc = start_agent_server()
    try:
        patch_messenger_for_logging()
        run_one_task()
    finally:
        print("\n[runner] shutting down agent server ...")
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        if out:
            print("\n[runner] agent server stderr/stdout (last 2000 chars):")
            print(out[-2000:])


if __name__ == "__main__":
    main()
