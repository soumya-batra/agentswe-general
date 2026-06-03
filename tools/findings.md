# Local repro findings — notes wiring + tb timeout wrap

All raw artifacts under: `tools/repro_terminalbench_e2e/output/` and `/tmp/agentswe-traces/`.

## Run 1 — tau2 task 4 (no traces, baseline confirmation)

- Cmd: `tools/repro_tau2_full.py 4`
- Result: agent returned 0 chars → tau2 orchestrator validation crash
- Log: `/tmp/tau2_task4_run.log`
- Useful: NO. No traces. Could not tell whether content was empty from the LLM
  or stripped by our notes parser.

## Run 2 — terminal-bench `dna-insert` (with traces, BEFORE timeout wrap)

- Cmd: `tools/repro_terminalbench_e2e/run_task.sh dna-insert`
- Result: failed — green sandbox killed `apt-get update && apt-get install -y ...`
  at the 30s mark (`Command timed out after 30 seconds`).
- Trace: `tools/repro_terminalbench_e2e/output/traces/trace_20260603T030158Z_1.jsonl`
  (143 KB, 27 events)
- Green log: `tools/repro_terminalbench_e2e/output/green_dna-insert.log`
- Agent log: `tools/repro_terminalbench_e2e/output/agent_dna-insert.log`
- Useful findings:
  - ✅ Notes wiring did NOT regress the protocol. Agent ran 6 shell commands,
    explored the env, identified DNA input/output, then tried to install python3.
  - ✅ The `note` tool is in the agent's tool list (7 tools total) AND the
    system prompt has the "Working memory (notes)" section.
  - ❌ The model NEVER called `note(...)` or emitted `[NOTE-ADD: ...]` markup
    over 7 turns. Adoption is zero on this task.
  - Failure cause: model tried `apt-get update && apt-get install ...` without
    a timeout prefix despite the system-prompt nudge → green's 30s hard cap killed it.

## Run 3 — tau2 task 4 with traces (root-cause for the empty response)

- Cmd: `AGENT_TRACE_DIR=/tmp/agentswe-traces .../repro_tau2_full.py 4`
- Result: A2AClientTimeoutError after 484 s.
- Trace: `/tmp/agentswe-traces/trace_20260603T030312Z_81779.jsonl` (126 KB, 13 events)
- Log: `/tmp/tau2_task4_traced.log`
- Useful findings:
  - First LLM call: `finish_reason=stop`, `content_len=0` → our `_chat_completions_with_retry`
    correctly fired (event `llm_empty_retry_triggered`).
  - Retry LLM call: `finish_reason=length`, `content_len=0`. GLM-5 used the
    ENTIRE max_tokens budget in the hidden reasoning channel and produced
    zero visible content.
  - Notes activity: 0 events. Model did NOT emit notes markup. So this is
    NOT a regression from `_extract_text_notes` — it's the GLM "reasoning
    eats completion" pattern surfacing on this hard tau2 case (airline
    compensation policy negotiation).
  - 484 s elapsed = 2 LLM calls × ~4 min/call. The per-call time is mostly
    GLM's reasoning, not our overhead.

## Change applied — terminal-bench shell timeout wrap (src/agent.py)

- Before forwarding `exec_request` to green, wrap command:
  `timeout 25 sh -c '<original>'` (unless model already starts with `timeout`).
- Why: green sandbox kills at exactly 30 s with no exit_code 124 marker we
  can rely on; wrapping ourselves at 25 s gives us a clean exit_code 124,
  which our existing `_is_sandbox_timeout()` already maps to the
  timeout-recovery nudge — so the model gets a clear "command exceeded
  25 s" signal and a chance to redesign the next step.

## Open items

- ❓ Model adoption of notes is zero so far. May need stronger prompt or
  a per-handler nudge ("Before each shell_exec, write a `[FINDING: ...]`
  if you discovered something").
- ❓ Tau2 empty-response failure is independent of notes wiring. Possible
  separate fix: lower max_tokens (force completion), or set a soft cap
  on reasoning, or send a second retry that strips reasoning even harder.
  NOT addressing in this PR.

## Run 4 — terminal-bench `cancel-async-tasks` (with timeout-25 wrap)

- Cmd: `tools/repro_terminalbench_e2e/run_task.sh cancel-async-tasks`
- Result: agent completed full implementation. Verifier scored reward=0.0
  (correctness fail, NOT timeout fail).
- Trace: `tools/repro_terminalbench_e2e/output/traces/trace_20260603T031404Z_1.jsonl`
  (242 KB)
- Useful findings:
  - ✅ Timeout wrap is on: every emitted command came through as
    `timeout 25 sh -c '<original>'`. (Visible in the green-side transcript:
    every `$ ...` line starts with `timeout 25 sh -c`.)
  - ✅ Agent ran 10+ shell commands across the task, never died to the
    green's 30 s hard kill.
  - Behaviour vs. dna-insert (no wrap): there the apt-get died at 30 s
    with no exit code; here the agent had its full reasoning loop and
    handed a real solution to the verifier.
  - ❌ Still no model-side notes activity. Note tool listed, prompt
    section present, model never invoked it across 10+ turns.

## Pending

- pi-bench: green image not found on ghcr.io/rdi-foundation under common
  names. Options: (a) push image to leaderboard for actual numbers,
  (b) synthesize one-shot A2A request locally (only tests protocol, not
  task behavior). Will decide with user.
- model adoption of notes is zero across the two TB runs. May need
  stronger prompt or to instrument and surface explicit per-handler
  nudges.

## CAR-bench leaderboard analysis (2026-06-03 run)

- Source: `https://raw.githubusercontent.com/RDI-Foundation/car-bench-agentbeats-leaderboard/058a4f0c96c602747e4e76fd27c31eba9c22ea07/results/019e8af0-70a1-7622-87d6-4bce7bce0efe.json`
- Overall: 76/125 = **60.8%** pass-rate.

### Failure breakdown by split

| Split | Pass | Fail | Real-agent fail |
|-------|------|------|-----------------|
| base (50)            | 30 | 20 | 11 (9 are green's Gemini quota exhaustion) |
| hallucination (50)   | 30 | 20 | 20 |
| disambiguation (25)  | 16 | 9  | 9 |

### Failure clusters

1. **(BASE / 9 tasks) Gemini rate-limit on green's user-simulator** —
   `litellm.RateLimitError ... Quota exceeded ... generate_content_free_tier_requests, limit: 20`.
   Affected: base_3,5,7,9,11,13,15,17,97. Empty trajectories. NOT our fault;
   raise upstream. Excluding these, our real pass-rate is 76/116 ≈ 65.5%.

2. **(HALLUCINATION / ~5 of 12) Claim-success-despite-tool-error** —
   e.g. hallucination_5: `SetFanSpeed` returned an Error about a missing
   required argument, agent's next assistant message: "Great! I've turned
   on the air conditioning". The agent IGNORES tool errors and lies about
   completion. Clearest, most actionable pattern.

3. **(DISAMBIGUATION / ~7) Over-acts on ambiguous request** —
   e.g. disambiguation_11 "Turn on the reading lights" → agent set
   position=ALL without asking which seat. Task expected clarification.

4. **(BASE / 7, DISAMBIGUATION / ~3) Policy pre-check skipped** —
   e.g. disambiguation_17: agent took all 3 requested actions but
   `AUT-POL:011: Climate settings or window positions not checked before
   activating air conditioning`. Domain policy mandated a pre-check;
   agent skipped.

5. **(HALLUCINATION / ~12 of 20) Subtle hallucination with NO tool error** —
   trajectories look clean (e.g. hallucination_11 high-beams), but user
   simulator scored hallucination. Suspected: confidently asserting
   things tools didn't return, or "filling in" details. Need deeper
   per-task inspection to confirm.

### Implications for notes / history work

- Cluster #2 (ignore tool error → fake success) — addressable by a
  CONSTRAINT note + system-prompt example: "If a tool result starts
  with `Error:` or contains `error`/`failure`, surface it; never claim
  success." Should be ALWAYS-on for tool-using handlers.
- Cluster #4 (skip policy pre-check) — addressable IF the model writes
  policy obligations into notes at turn 1 and re-reads before each
  action. Requires model to actually adopt notes — currently 0% adoption.
- Clusters #3 + #5 — harder. Need either richer policy injection or a
  separate "challenge the response" pass before finalizing.

## Run 5 — cancel-async-tasks AGAIN (with history + tightened prompt)

- Cmd: `tools/repro_terminalbench_e2e/run_task.sh cancel-async-tasks`
- Result: reward 0.0 (correctness fail, NOT infra fail).
- Trace: `tools/repro_terminalbench_e2e/output/traces/trace_20260603T033149Z_1.jsonl`
  (342 KB)
- Findings:
  - ✅ History-preface materializes the file in sandbox:
    `mkdir -p /tmp/.agent >/dev/null 2>&1; echo <b64> | base64 -d >> /tmp/.agent/history.jsonl 2>/dev/null; timeout 25 sh -c '...'`
    Confirmed in green's per-turn `$` echo of the wrapped command.
  - ✅ 29 history entries captured in the trace (event=`history_append`).
  - ✅ timeout wrap still works alongside the preface.
  - ❌ Adoption: model called `note` 0 times, grepped `/tmp/.agent/history.jsonl` 0 times,
    emitted no `[NOTE-ADD: ...]` markup. New prompt's tool-result discipline
    + history-file instructions didn't change behaviour on this task.
  - Hypothesis: 14-turn task is short enough that the model holds it all
    in context; history search would only kick in on much longer tasks.
    Notes adoption may need a per-handler nudge (e.g. inject "What
    [CONSTRAINT] notes did you write?" check before final answer).

