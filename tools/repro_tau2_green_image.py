"""Run the REAL tau2 green agent docker image against our local purple.

This is the most production-faithful local repro possible:
- pulls/uses ghcr.io/rdi-foundation/tau2-agentbeats:latest
- starts our agent server on the host
- runs the green container, points it at host.docker.internal:9011
- POSTs the same EvalRequest shape the leaderboard uses
- reads the artifact response (per-task rewards + summary)

Usage:
  export OPENROUTER_API_KEY=...   # for our purple agent's LLM
  export OPENAI_API_KEY=...        # for tau2 user simulator (gpt-4o-mini)
  /opt/homebrew/bin/uv run python tools/repro_tau2_green_image.py [task_id ...]
"""

import json
import os
import subprocess
import sys
import time
import uuid

import httpx

THIS_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PURPLE_PORT = 9011
GREEN_PORT = 9009
GREEN_CONTAINER = "tau2-green-repro"
GREEN_IMAGE = "ghcr.io/rdi-foundation/tau2-agentbeats:latest"
TASK_IDS = sys.argv[1:] or ["0"]


def start_purple_server():
    print(f"[runner] starting purple server on :{PURPLE_PORT} ...", flush=True)
    proc = subprocess.Popen(
        ["/opt/homebrew/bin/uv", "run", "python", "src/server.py",
         "--host", "0.0.0.0", "--port", str(PURPLE_PORT),
         # Agent card must advertise an address reachable from inside
         # the green container; 0.0.0.0 isn't routable from there.
         "--card-url", f"http://192.168.65.254:{PURPLE_PORT}/"],
        cwd=THIS_REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"purple server died:\n{out}")
        try:
            r = httpx.get(
                f"http://127.0.0.1:{PURPLE_PORT}/.well-known/agent-card.json",
                timeout=2,
            )
            if r.status_code == 200:
                print("[runner] purple server ready", flush=True)
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("purple server didn't come up in 60s")


def start_green_container():
    print(f"[runner] starting green container ({GREEN_IMAGE}) ...", flush=True)
    subprocess.run(
        ["/usr/local/bin/docker", "rm", "-f", GREEN_CONTAINER],
        capture_output=True,
    )
    env_args = []
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"):
        v = os.environ.get(key)
        if v:
            env_args += ["-e", f"{key}={v}"]
    if not env_args:
        raise RuntimeError(
            "Need OPENAI_API_KEY (or GEMINI/DEEPSEEK) for the user simulator"
        )
    result = subprocess.run(
        ["/usr/local/bin/docker", "run", "-d",
         "--name", GREEN_CONTAINER,
         "-p", f"{GREEN_PORT}:{GREEN_PORT}",
         "--add-host=host.docker.internal:host-gateway",
         *env_args,
         GREEN_IMAGE],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker run failed: {result.stderr}")
    container_id = result.stdout.strip()
    print(f"[runner] green container started ({container_id[:12]})", flush=True)

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            r = httpx.get(
                f"http://127.0.0.1:{GREEN_PORT}/.well-known/agent-card.json",
                timeout=2,
            )
            if r.status_code == 200:
                print("[runner] green container ready", flush=True)
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("green container didn't come up in 60s")


def send_eval_request():
    body = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": uuid.uuid4().hex,
                "parts": [
                    {"kind": "text", "text": json.dumps({
                        "participants": {
                            # Docker Desktop Mac NATs container-to-host as
                            # 192.168.65.254 (host.docker.internal resolves
                            # to this); use it directly to avoid any DNS
                            # quirks inside the green's httpx client.
                            "agent": f"http://192.168.65.254:{PURPLE_PORT}",
                        },
                        "config": {
                            "domain": "airline",
                            "task_ids": TASK_IDS,
                            "max_steps": 200,  # tau2-agentbeats README default
                        },
                    })}
                ],
            },
        },
    }
    print(f"[runner] POSTing EvalRequest to green for tasks {TASK_IDS} ...", flush=True)
    with httpx.Client(timeout=600) as client:
        r = client.post(f"http://127.0.0.1:{GREEN_PORT}/", json=body)
    print(f"[runner] HTTP {r.status_code}", flush=True)
    print("[runner] response JSON:")
    try:
        print(json.dumps(r.json(), indent=2)[:5000])
    except Exception:
        print(r.text[:5000])


def dump_green_logs():
    print("\n[runner] --- last 200 lines of green container logs ---", flush=True)
    result = subprocess.run(
        ["/usr/local/bin/docker", "logs", "--tail", "200", GREEN_CONTAINER],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("(stderr)", result.stderr)


def cleanup_green():
    subprocess.run(
        ["/usr/local/bin/docker", "rm", "-f", GREEN_CONTAINER],
        capture_output=True,
    )


def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY not set (purple LLM will 401)")
    proc = start_purple_server()
    try:
        start_green_container()
        try:
            send_eval_request()
        finally:
            dump_green_logs()
            cleanup_green()
    finally:
        print("\n[runner] shutting down purple server ...", flush=True)
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        if out:
            print("[runner] purple server stdout (last 2000 chars):")
            print(out[-2000:])


if __name__ == "__main__":
    main()
