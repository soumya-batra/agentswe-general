#!/usr/bin/env bash
# Run a single terminal-bench task against our local agent.
#
# Usage:
#   source /tmp/agentswe-keys.sh
#   ./run_task.sh dna-insert
#
# Prereqs:
#   - docker (Desktop on macOS)
#   - OPENROUTER_API_KEY in env
#
# Output: traces in ./output/traces/, green logs in stdout.

set -euo pipefail
TASK="${1:-dna-insert}"

cd "$(dirname "$0")"
mkdir -p output/traces

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY must be set"
    exit 1
fi

echo "[repro_tb] cleaning previous run ..."
docker compose down -v 2>/dev/null || true
rm -rf output/traces && mkdir -p output/traces

echo "[repro_tb] starting green + agent stack ..."
docker compose up -d green agent

echo "[repro_tb] waiting for both services healthy ..."
for i in $(seq 1 30); do
    g=$(docker inspect --format='{{.State.Health.Status}}' tb-green-repro 2>/dev/null || echo "missing")
    a=$(docker inspect --format='{{.State.Health.Status}}' tb-agent-repro 2>/dev/null || echo "missing")
    echo "[repro_tb] green=$g agent=$a (try $i)"
    if [ "$g" = "healthy" ] && [ "$a" = "healthy" ]; then
        break
    fi
    sleep 3
done

echo "[repro_tb] sending EvalRequest for task=$TASK ..."
# Green's A2A server expects an EvalRequest as the message payload.
# We send via the standard A2A message/send JSON-RPC envelope.
# The green parses EvalRequest from the TEXT part via
# EvalRequest.model_validate_json(get_message_text(message)).
REQ=$(TASK="$TASK" python3 <<'PY'
import json, os
eval_req = json.dumps({
    "participants": {"agent": "http://agent:9010"},
    "config": {"task": os.environ["TASK"], "oracle": False},
})
print(json.dumps({
    "jsonrpc": "2.0", "id": "1", "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "messageId": f"evalreq-{os.getpid()}",
            "parts": [{"kind": "text", "text": eval_req}],
        }
    }
}))
PY
)

curl -sS -X POST http://127.0.0.1:9019/ \
    -H "Content-Type: application/json" \
    -d "$REQ" | tee output/green_response_${TASK}.json
echo

echo "[repro_tb] writing logs ..."
docker logs tb-green-repro > output/green_${TASK}.log 2>&1 || true
docker logs tb-agent-repro > output/agent_${TASK}.log 2>&1 || true

echo "[repro_tb] traces in:"
ls -la output/traces/ || true

echo "[repro_tb] tearing down ..."
docker compose down -v
