"""General-purpose A2A purple agent.

Receives A2A messages and dispatches to a handler based on the SHAPE of
the incoming payload. Each handler runs an LLM loop using GLM (or any
OpenAI-compatible model) and returns an A2A artifact.

Handler dispatch is by message *format*, not by benchmark identity:

  - JSON with protocol == "terminal-bench-shell-v1"  -> terminal_shell
  - JSON with {instance_id, problem_statement,
               docker_image}                          -> swe_bench
  - JSON with {messages: [...], tools: [...]}         -> openai_passthrough
  - anything else (plain text or arbitrary JSON)      -> generic

This is fair game per AgentBeats rules: protocols ARE the API contract.
What's forbidden is hardcoding "if benchmark == 'OfficeQA': ..." style
lookups. We don't do that.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

import openai


# ---------------------------------------------------------------------------
# Trace logging — when AGENT_TRACE_DIR is set, every LLM call, A2A
# in/out, and notes mutation is appended as one JSON object per line to
# a session file in that dir. Off by default. Use only for local debug.
_TRACE_DIR = os.environ.get("AGENT_TRACE_DIR", "").strip() or None
_TRACE_PATH: str | None = None
if _TRACE_DIR:
    try:
        os.makedirs(_TRACE_DIR, exist_ok=True)
        _TRACE_PATH = os.path.join(
            _TRACE_DIR,
            f"trace_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}.jsonl",
        )
    except Exception as _e:
        print(f"[trace] disabled — failed to create {_TRACE_DIR}: {_e}")
        _TRACE_PATH = None


def _trace(event: str, **fields: Any) -> None:
    """Append a structured trace record. No-op if AGENT_TRACE_DIR unset."""
    if not _TRACE_PATH:
        return
    rec: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "t": time.monotonic(),
        "event": event,
    }
    rec.update(fields)
    try:
        with open(_TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
    except Exception as _e:
        # Never break the agent because tracing failed
        print(f"[trace] write failed: {_e}")


def _trace_msg_summary(msg: Any) -> dict[str, Any]:
    """Compact dict-form of an OpenAI ChatCompletionMessage."""
    out: dict[str, Any] = {}
    try:
        out["role"] = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        out["content_len"] = len(content) if content else 0
        out["content"] = content
        tc = getattr(msg, "tool_calls", None)
        if tc:
            out["tool_calls"] = [
                {
                    "id": t.id,
                    "name": t.function.name,
                    "arguments": t.function.arguments,
                }
                for t in tc
            ]
    except Exception as _e:
        out["error"] = str(_e)
    return out
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import new_agent_parts_message, new_agent_text_message

from llm import (
    json_output,
    make_client,
    model_name,
    reasoning_enabled,
    retrieval_enabled,
    tools_enabled,
)
from messenger import Messenger
from tools import (
    GENERIC_TOOL_SCHEMAS,
    RETRIEVE_DOCUMENTS,
    DockerWorkdir,
    LocalWorkdir,
    Workdir,
    dispatch as tool_dispatch,
)


def _args_dict(raw: str | None) -> dict[str, Any]:
    """Parse a tool_call's `arguments` field to a dict.

    The OpenAI SDK gives us `tc.function.arguments` as a string. Models
    occasionally emit a non-object JSON (list, scalar, null) — when that
    hits a downstream `args.get(...)` we get
        'list' object has no attribute 'get'
    and the whole task fails. Defensively coerce anything non-dict to
    an empty dict.
    """
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# Generic recovery instructions injected as a follow-up user turn
# whenever a tool result contains "timed out". The model frequently
# retries the same synchronous long-running command and gets killed
# again; this nudge tells it to background the work and poll on
# later turns. Applies to any handler that ingests tool results
# (terminal-bench's per-command sandbox limit, our own LocalWorkdir
# / DockerWorkdir timeouts, etc.) — not benchmark-specific.
_TIMEOUT_RECOVERY_NUDGE = (
    "[execution-time-limit] The previous command exceeded the "
    "execution time limit and was killed. Do NOT retry the same "
    "command synchronously — it will time out again.\n\n"
    "For long-running work, background it and poll across separate "
    "turns:\n\n"
    "    nohup CMD > /tmp/job.log 2>&1 &\n"
    "    echo $!         # capture the PID\n\n"
    "Then on a LATER turn check progress:\n"
    "    tail -n 50 /tmp/job.log\n"
    "    ps -p <PID> > /dev/null && echo running || echo done\n"
    "    cat /tmp/job.log   # full output when done\n\n"
    "Re-running the same synchronous command is forbidden. Background "
    "it, then poll progress on the next turn."
)


def _is_exec_timeout(result: str) -> bool:
    """Deterministic check: did OUR shell_exec / python_exec / DockerWorkdir
    just report a timeout? Matches the exact format strings produced in
    tools.py (LocalWorkdir.shell_exec returns the bracketed form,
    DockerWorkdir / python_exec route through _format_result which puts
    the message in the stderr block). Won't false-positive on log file
    content because we require the specific phrasing we ourselves emit.
    """
    if not result:
        return False
    return (
        "[error: timed out after" in result
        or "stderr:\ntimed out after" in result
    )


def _is_sandbox_timeout(exit_code: Any, stderr: str) -> bool:
    """Deterministic check: did terminal-bench's green sandbox kill the
    command at its per-command time limit? We trust two signals:
      - the green's grader prints 'Command timed out after N seconds'
        into the exec_result's stderr — this is the exact string that
        ends up in the task_rewards error field
      - the GNU `timeout` exit code (124) is the standard convention
        for any sandbox that wraps the child in `timeout N ...`
    """
    if stderr and "Command timed out" in stderr:
        return True
    try:
        if int(exit_code) == 124:
            return True
    except (TypeError, ValueError):
        pass
    return False


_TOOL_OUTPUT_CHAR_BUDGET = 20_000


def _truncate_tool_output(text: str, budget: int = _TOOL_OUTPUT_CHAR_BUDGET) -> str:
    """Cap a single tool result so one oversized stdout (e.g. cat of a
    build log, find / on a huge tree) doesn't blow past the model's
    context window. Keeps a small head and a larger tail because exit
    codes, errors, and final output usually appear at the bottom.
    """
    if len(text) <= budget:
        return text
    head_keep = budget // 4
    tail_keep = budget - head_keep
    dropped = len(text) - budget
    return (
        text[:head_keep]
        + f"\n\n[... truncated {dropped:,} chars of tool output ...]\n\n"
        + text[-tail_keep:]
    )


async def _chat_completions_with_retry(client, *, max_attempts: int = 4, **kwargs):
    """OpenRouter sometimes routes to providers that 5xx or return a
    non-JSON body (HTML error page → JSONDecodeError in the SDK). One
    bad pick shouldn't kill the whole task.

    Also handles the GLM-5 "reasoning eats the completion" bug: the
    upstream returns a 200 with finish_reason=stop, no tool_calls,
    AND empty content (all tokens spent in the hidden reasoning
    channel). On that pattern, retry ONCE with reasoning forcibly
    disabled — even if it was already off, the explicit override
    nudges OpenRouter / the provider to actually drop it. Production
    tau2 saw this fire on ~4% of tasks even with reasoning_enabled
    already false; this defensive retry recovers them.
    """

    async def _one_call(call_kwargs):
        delay = 1.0
        for attempt in range(max_attempts):
            try:
                return await client.chat.completions.create(**call_kwargs)
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
                json.JSONDecodeError,
            ) as e:
                _trace(
                    "llm_call_retry",
                    attempt=attempt,
                    error=f"{type(e).__name__}: {e}",
                )
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2

    _trace(
        "llm_call_request",
        model=kwargs.get("model"),
        n_messages=len(kwargs.get("messages", [])),
        messages=kwargs.get("messages"),
        tools=kwargs.get("tools"),
        tool_choice=kwargs.get("tool_choice"),
        response_format=kwargs.get("response_format"),
        seed=kwargs.get("seed"),
        extra_body=kwargs.get("extra_body"),
        parallel_tool_calls=kwargs.get("parallel_tool_calls"),
    )
    resp = await _one_call(kwargs)
    choice = resp.choices[0]
    msg = choice.message
    _trace(
        "llm_call_response",
        finish_reason=choice.finish_reason,
        model_routed=getattr(resp, "model", None),
        usage=getattr(resp, "usage", None),
        msg=_trace_msg_summary(msg),
    )
    empty_content = not (msg.content or "").strip()
    no_tool_calls = not msg.tool_calls
    finished_clean = choice.finish_reason == "stop"
    if empty_content and no_tool_calls and finished_clean:
        _trace("llm_empty_retry_triggered")
        retry_kwargs = dict(kwargs)
        extra = dict(retry_kwargs.get("extra_body") or {})
        extra["reasoning"] = {"enabled": False}
        retry_kwargs["extra_body"] = extra
        try:
            retry_resp = await _one_call(retry_kwargs)
        except Exception as e:
            _trace("llm_empty_retry_failed", error=f"{type(e).__name__}: {e}")
            return resp  # original at least has a parseable shape
        retry_msg = retry_resp.choices[0].message
        _trace(
            "llm_empty_retry_response",
            finish_reason=retry_resp.choices[0].finish_reason,
            msg=_trace_msg_summary(retry_msg),
        )
        if (retry_msg.content or "").strip() or retry_msg.tool_calls:
            return retry_resp
    return resp


# ---------------------------------------------------------------------------
# Message inspection


def extract_payload(message: Message) -> tuple[Any, str]:
    """Return (parsed_payload, raw_text).

    parsed_payload is:
      - the dict from a DataPart if present, OR
      - a JSON-decoded dict if the text content parses, OR
      - None for plain text.
    raw_text is the concatenated text content (useful even when JSON parsed).
    """
    text_chunks: list[str] = []
    data_obj: Any = None
    for part in message.parts:
        if isinstance(part.root, DataPart) and data_obj is None:
            data_obj = part.root.data
        elif isinstance(part.root, TextPart):
            text_chunks.append(part.root.text)
    text = "\n".join(text_chunks)

    _trace(
        "a2a_incoming",
        message_id=getattr(message, "message_id", None),
        context_id=getattr(message, "context_id", None),
        n_parts=len(message.parts),
        data_payload=data_obj,
        raw_text=text,
    )

    if data_obj is not None:
        return data_obj, text

    stripped = text.strip()
    if stripped and stripped[0] in "{[":
        try:
            return json.loads(stripped), text
        except (json.JSONDecodeError, ValueError):
            pass
    return None, text


def classify(payload: Any, raw_text: str = "") -> str:
    if isinstance(payload, dict):
        if payload.get("protocol") == "terminal-bench-shell-v1":
            return "terminal_shell"
        if (
            "instance_id" in payload
            and "problem_statement" in payload
            and "docker_image" in payload
        ):
            return "swe_bench"
        if isinstance(payload.get("messages"), list):
            return "openai_passthrough"
        # CAR-bench: green sends a DataPart containing tool schemas or
        # tool_results, paired with a TextPart for the user utterance.
        # Distinguish from pi-bench by the absence of a `messages` field.
        # NOTE: we deliberately do NOT also fall back to a raw-text
        # "System:\\n...\\n\\nUser:" marker. CAR-bench always emits the
        # DataPart with `tools` on first turn (verified against the
        # reference baseline), so the dict check is sufficient. A text
        # fallback risks false-positive routing of tau2 first turns
        # that happen to contain both phrases in their policy text.
        if "tools" in payload or "tool_results" in payload:
            return "car_bench"
    return "generic"


# ---------------------------------------------------------------------------
# Agent


GENERIC_SYSTEM_PROMPT = """\
You are a general-purpose AI agent participating in benchmark evaluations
over the A2A protocol. You may receive tasks of any kind: question
answering, coding, web research, multi-turn dialogue, policy decisions.

Guidelines:
- Read the task carefully. If a specific OUTPUT FORMAT is requested
  (XML tags, JSON, a patch/diff, a tool call), follow it EXACTLY.
- If you have tools available, use them when you don't know something
  or need to act on the environment; otherwise rely on your own
  knowledge and the task's own instructions/tools.
- Be concise. Stop calling tools once you have enough information.
- For multi-turn dialogues, treat each incoming message as the next turn.
- Each shell command must complete quickly (often within 30 seconds in
  sandboxed environments). For long-running operations (builds, large
  recursive searches, package installs), narrow the scope, set explicit
  timeouts (`timeout 25 ...`), or run in the background (`nohup ... &`)
  and check results in a later step.

## Tool-result discipline (CRITICAL)
NEVER claim an action succeeded if the tool result indicates failure.
If a tool's response contains `"Error"`, `"error"`, `status: "FAILURE"`,
or is missing the expected fields, you MUST:
  1. Acknowledge the failure to the user in plain language.
  2. Either retry with corrected arguments, ask the user for missing
     info, or honestly state the limitation.
Do NOT proceed to the next step or to a confident success message.
Example of WRONG behavior — DO NOT do this:
  tool returns: `Error: SetFanSpeed.invoke() missing required arg 'level'`
  assistant says: "Great! I've turned on the air conditioning for you."
This is a hallucination and will fail evaluation.

## Policy obligations
When the task includes a policy document or domain rules (often in the
system prompt or benchmark_context), treat its pre-condition checks as
HARD requirements. Common pattern: "before activating X, check the
state of Y". If you skip the check and go straight to the action, even
if the action succeeds, the task fails. When in doubt, run the check.

## Ambiguous user requests
If a user request could refer to multiple distinct entities (e.g. "the
reading light" when there are several seats, "the window" when several
are open), ASK which one rather than guessing or applying to all.
Over-acting on ambiguity is a common failure mode.

## Conversation history file
Your previous turns (your tool calls and their results) are appended to
`/tmp/.agent/history.jsonl` inside your sandbox shell. When context
gets long and you need to recall an earlier finding, grep / cat that
file instead of re-running discovery commands. Example:
  grep -i "window_driver_position" /tmp/.agent/history.jsonl
  tail -n 5 /tmp/.agent/history.jsonl
Each line is a JSON object with fields `turn`, `kind`
(`task` | `tool_call` | `tool_result`), `name`, and `text`.

## Working memory (notes)
You have a working memory that PERSISTS across turns and is injected
at the TOP of every turn under "# Working notes". Use it to commit
short one-line entries you'll need to reference later. Suggested
prefixes (use what fits, invent your own only if needed):
  [CONSTRAINT: ...]    — a rule/policy that must be obeyed
  [FINDING: ...]       — a discovery (file path, value, fact)
  [DECISION: ...]      — a decision you made + the reason
  [PLAN: ...]          — your next step(s)
  [BLOCKED-ON: ...]    — what's blocking you
  [DONE: ...]          — a completed step

How to add notes:
- If you have a `note` tool available, call it: `note(action="add", text="...")`.
- Otherwise, emit `[NOTE-ADD: <text>]` anywhere in your response. It
  will be stripped before downstream parsing.
To remove: tool `note(action="remove", text="n3")` or `[NOTE-REMOVE: n3]`.

BEFORE each tool call or final answer, scan the existing notes. Do not
contradict a CONSTRAINT, repeat a DONE step, or re-discover a FINDING.
Keep notes SHORT (one line). Only commit things you will need later —
don't narrate the obvious.
"""


_NOTES_TOOL_INSTRUCTION_TEXT = (
    "## Working memory (notes)\n"
    "You have a working memory that persists across turns and is "
    "injected at the TOP of every turn under '# Working notes'. Use "
    "it to commit short one-line entries you will need later.\n\n"
    "Suggested prefixes (use what fits): [CONSTRAINT: ...], "
    "[FINDING: ...], [DECISION: ...], [PLAN: ...], [BLOCKED-ON: ...], "
    "[DONE: ...].\n\n"
    "To ADD a note: emit `[NOTE-ADD: <text>]` anywhere in your "
    "response. It will be stripped from the response your caller "
    "sees but stored in your working memory.\n"
    "To REMOVE a note: emit `[NOTE-REMOVE: n3]` (use the n<id> shown "
    "next to each note).\n\n"
    "BEFORE each action or final answer, scan your existing notes. "
    "Do not contradict a CONSTRAINT, repeat a DONE step, or "
    "re-discover a FINDING. Keep notes SHORT (one line). Only commit "
    "things you will reference later."
)


_NOTES_TAG_ADD = re.compile(r"\[NOTE-ADD:\s*(.*?)\]", re.DOTALL)
_NOTES_TAG_REMOVE = re.compile(r"\[NOTE-REMOVE:\s*n(\d+)\s*\]")
_NOTES_SOFT_CAP = 50


class Agent:
    """Per-context conversation state. Created once per A2A context_id."""

    def __init__(self) -> None:
        self.messenger = Messenger()
        self.client = make_client()
        self.handler: str | None = None
        # OpenAI-format conversation history (system + turns)
        self.history: list[dict[str, Any]] = [
            {"role": "system", "content": GENERIC_SYSTEM_PROMPT}
        ]
        # Per-conversation workdir (lazily created). A handler may swap
        # in a different Workdir implementation (e.g. DockerWorkdir for
        # SWE-bench) before any tool calls happen.
        self._workdir: Workdir | None = None
        # When a protocol handler emits a tool call as A2A and waits for
        # the next turn to bring the result, the tool_call_id sits here.
        self.pending_protocol_tool_id: str | None = None
        # CAR-bench state: tool schemas from the green (set on first
        # turn) and the previous turn's emitted tool_calls so we can
        # map incoming tool_results back to their tool_call_id by name.
        self._car_bench_tools: list[dict[str, Any]] | None = None
        self._car_bench_pending_tool_calls: list[dict[str, Any]] | None = None
        # Working memory — persistent notes across turns. Each note is
        # a short string; survives history truncation. Soft-capped at
        # _NOTES_SOFT_CAP; when over, oldest gets dropped.
        self._notes: list[tuple[int, str]] = []
        self._next_note_id: int = 1
        # Per-conversation history log — append-only JSONL, surfaced to
        # the model via a file in its shell (terminal-bench) or via a
        # file in its workdir (sweb / generic). One entry per turn /
        # tool call / tool result.
        self._history_log: list[dict[str, Any]] = []

    # ---- working notes -------------------------------------------------

    def _note_add(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "[note: empty text, not added]"
        nid = self._next_note_id
        self._next_note_id += 1
        self._notes.append((nid, text))
        if len(self._notes) > _NOTES_SOFT_CAP:
            # Drop the oldest to stay within budget.
            self._notes = self._notes[-_NOTES_SOFT_CAP:]
        return f"[note added: n{nid}]"

    def _note_remove(self, key: str) -> str:
        key = (key or "").strip().lstrip("n")
        try:
            target_id = int(key)
        except (TypeError, ValueError):
            return f"[note: bad id {key!r}, not removed]"
        before = len(self._notes)
        self._notes = [(i, t) for (i, t) in self._notes if i != target_id]
        return (
            f"[note n{target_id} removed]"
            if len(self._notes) < before
            else f"[note: no n{target_id} to remove]"
        )

    def _notes_system_message(self) -> dict[str, str] | None:
        if not self._notes:
            return None
        lines = "\n".join(f"[n{i}] {t}" for i, t in self._notes)
        return {
            "role": "system",
            "content": (
                "# Working notes (carry-forward across turns):\n"
                + lines
                + "\nScan these BEFORE your next action; do not "
                "contradict, repeat, or rediscover them."
            ),
        }

    # ---- history log --------------------------------------------------

    def _record_history(
        self,
        kind: str,
        text: str,
        *,
        name: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one entry to the conversation history log."""
        entry: dict[str, Any] = {
            "turn": len(self._history_log),
            "kind": kind,
            # Cap each entry at ~80 KB so a single huge tool output
            # doesn't push the JSONL past sandbox shell limits.
            "text": (text or "")[:80_000],
        }
        if name:
            entry["name"] = name
        if extra:
            entry.update(extra)
        self._history_log.append(entry)
        _trace("history_append", entry=entry)
        return entry

    def _history_jsonl_preface(self) -> str:
        """Shell command that writes the LATEST history entry to the
        sandbox's /tmp/.agent/history.jsonl, used by terminal-bench
        before each exec_request. Empty string if there's nothing to
        write. Failsafe — never errors the parent command."""
        if not self._history_log:
            return ""
        # Encode as base64 to dodge ALL quoting hazards (newlines,
        # backticks, $ expansions, single quotes inside JSON).
        import base64
        line = json.dumps(self._history_log[-1])
        b64 = base64.b64encode((line + "\n").encode("utf-8")).decode("ascii")
        return (
            "mkdir -p /tmp/.agent >/dev/null 2>&1; "
            f"echo {b64} | base64 -d >> /tmp/.agent/history.jsonl 2>/dev/null; "
        )

    def _extract_text_notes(self, content: str | None) -> str:
        """Pull [NOTE-ADD: ...] / [NOTE-REMOVE: nK] markup out of
        free-form text content, mutate self._notes, return the cleaned
        content with the markup stripped (so the downstream consumer
        — green's parser, judge, etc. — doesn't see it)."""
        if not content:
            return content or ""
        for m in _NOTES_TAG_ADD.findall(content):
            self._note_add(m)
        for m in _NOTES_TAG_REMOVE.findall(content):
            self._note_remove(m)
        cleaned = _NOTES_TAG_ADD.sub("", content)
        cleaned = _NOTES_TAG_REMOVE.sub("", cleaned)
        # Tidy up leftover whitespace from removed tags.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _handle_note_tool(self, args: dict[str, Any]) -> str:
        """Process a `note` tool call. args = {action, text}."""
        action = str(args.get("action", "")).strip().lower()
        text = str(args.get("text", ""))
        if action == "add":
            return self._note_add(text)
        if action == "remove":
            return self._note_remove(text)
        if action == "list":
            if not self._notes:
                return "[no notes]"
            return "\n".join(f"[n{i}] {t}" for i, t in self._notes)
        return f"[note: unknown action {action!r}; use add | remove | list]"

    def workdir(self) -> Workdir:
        if self._workdir is None:
            self._workdir = LocalWorkdir(tempfile.mkdtemp(prefix="agentswe-"))
        return self._workdir

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        payload, raw_text = extract_payload(message)

        # Sticky handler: classify on first message, keep using it.
        if self.handler is None:
            self.handler = classify(payload, raw_text)
        _trace("handler_dispatch", handler=self.handler)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[handler={self.handler}] working..."),
        )

        if self.handler == "openai_passthrough":
            await self._openai_passthrough(payload, updater)
        elif self.handler == "terminal_shell":
            await self._terminal_shell(payload, raw_text, updater)
        elif self.handler == "swe_bench":
            await self._swe_bench(payload, updater)
        else:
            # car_bench is dispatched by the executor BEFORE we get here
            # (it uses a Message-only response model, no task lifecycle).
            await self._generic(payload, raw_text, updater)

    # ---- handlers ------------------------------------------------------

    async def _generic(
        self,
        payload: Any,
        raw_text: str,
        updater: TaskUpdater,
    ) -> None:
        """Free-form LLM loop with web tools. Handles OfficeQA, BrowseComp+,
        tau2-bench, and any other plain-text or unknown-JSON task."""
        # Compose the user turn. If the green sent JSON, embed it; otherwise
        # pass the raw text through.
        if payload is not None and not isinstance(payload, str):
            user_content = (
                "Task (JSON):\n```json\n"
                + json.dumps(payload, indent=2)
                + "\n```"
            )
        else:
            user_content = raw_text or "(empty message)"

        # When retrieval is enabled (e.g. OfficeQA deployment), pre-retrieve
        # top-k passages for this turn's question and prepend them as a
        # system message — the model gets the corpus context up front
        # instead of having to decide whether to retrieve. It can STILL
        # call `retrieve_documents` again with a refined query if the
        # initial set isn't specific enough.
        if retrieval_enabled():
            try:
                from tools import retrieve_documents as _retrieve
                query_text = raw_text or user_content
                retrieved = await _retrieve(query=query_text)
                self.history.append({
                    "role": "system",
                    "content": (
                        "# Retrieved corpus context\n"
                        "The following passages were retrieved from the "
                        "baked-in corpus based on the user's question. "
                        "Use them as your primary source. You may call "
                        "`retrieve_documents` again with a refined query "
                        "if these aren't specific enough.\n\n"
                        "# Output format\n"
                        "The judge for this benchmark extracts your final "
                        "answer from `<FINAL_ANSWER>...</FINAL_ANSWER>` "
                        "tags via regex. Your response MUST end with the "
                        "answer wrapped in those exact tags, e.g. "
                        "`<FINAL_ANSWER>42</FINAL_ANSWER>` or "
                        "`<FINAL_ANSWER>-118255.5</FINAL_ANSWER>`. "
                        "Without the tags the answer is marked WRONG "
                        "even if it is numerically correct. Show your "
                        "reasoning before the tag; only the contents "
                        "between the tags are scored.\n\n"
                        "# Arithmetic precision\n"
                        "Preserve the operand ORDER and SIGN exactly as "
                        "the question asks. 'Difference between A and B' "
                        "means A - B (NOT B - A) — this changes the "
                        "sign of the answer. 'Ratio of A to B' means "
                        "A / B. 'A minus B' means A - B. When the "
                        "result is negative, include the leading minus "
                        "sign. Double-check the final sign before "
                        "wrapping in the FINAL_ANSWER tags. For any "
                        "non-trivial arithmetic, use `python_exec` "
                        "rather than mental math.\n\n"
                        + retrieved
                    ),
                })
            except Exception as e:  # noqa: BLE001
                # Never let a retrieval failure kill the whole task.
                print(
                    f"[retrieval] pre-retrieve failed: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

        self.history.append({"role": "user", "content": user_content})

        # When TOOLS_ENABLED=off (e.g. tau2 deployment), the task's own
        # prompt has already given the model whatever tool surface it
        # needs as inline text; we must not expose our own OpenAI-format
        # tools or the model will mix surfaces.
        tools = list(GENERIC_TOOL_SCHEMAS) if tools_enabled() else []
        # Even when generic tools are off, expose retrieve_documents
        # when the corpus is baked in — it's the right tool for the job.
        if retrieval_enabled():
            tools.append(RETRIEVE_DOCUMENTS)

        final_text, paused = await self._chat_loop(
            tools=tools,
            updater=updater,
            workdir=self.workdir(),
            max_steps=200,
        )
        # Generic mode doesn't pause; paused should always be None here.

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=final_text))],
            name="response",
        )

    async def _openai_passthrough(
        self,
        payload: dict[str, Any],
        updater: TaskUpdater,
    ) -> None:
        """Pi-Bench-style: green sends {messages, tools, benchmark_context}
        in OpenAI chat-completion format. We forward to GLM and wrap the
        response as an A2A DataPart with the same shape pi-bench expects.
        """
        messages = payload.get("messages") or []
        tools = payload.get("tools") or []
        benchmark_context = payload.get("benchmark_context") or []

        # If the green sent benchmark_context (e.g. pi-bench's policy /
        # task notes), prepend it as a system message so the model has
        # the operating context the benchmark expects. Strip any
        # incoming system messages first to avoid duplication.
        if benchmark_context:
            ctx_blocks = []
            for node in benchmark_context:
                if not isinstance(node, dict):
                    continue
                content = str(node.get("content", "")).strip()
                if not content:
                    continue
                kind = str(node.get("kind", "context")).strip() or "context"
                title = kind.replace("_", " ").title()
                ctx_blocks.append(f"### {title}\n{content}")
            if ctx_blocks:
                system_content = "## Benchmark Context\n" + "\n\n".join(ctx_blocks)
                messages = [
                    {"role": "system", "content": system_content},
                    *(m for m in messages if isinstance(m, dict) and m.get("role") != "system"),
                ]

        # Inject working notes (if any) and a one-time tools-disabled
        # notes instruction. The benchmark's own tool surface stays
        # primary; this just gives the model a private persistent
        # scratch it can write to via [NOTE-ADD: ...] markup.
        notes_msg = self._notes_system_message()
        notes_instr = {"role": "system", "content": _NOTES_TOOL_INSTRUCTION_TEXT}
        prefix: list[dict[str, Any]] = []
        if notes_msg:
            prefix.append(notes_msg)
        prefix.append(notes_instr)
        messages = prefix + list(messages)

        kwargs: dict[str, Any] = {
            "model": model_name(),
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_output():
            kwargs["response_format"] = {"type": "json_object"}
        if not reasoning_enabled():
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        # pi-bench passes a seed for reproducibility; forward when present.
        if "seed" in payload and payload["seed"] is not None:
            kwargs["seed"] = payload["seed"]

        resp = await _chat_completions_with_retry(self.client, **kwargs)
        choice = resp.choices[0]
        msg = choice.message

        # Strip and store any [NOTE-ADD/-REMOVE] markup the model
        # emitted in content BEFORE forwarding to pi-bench's parser.
        cleaned_content = self._extract_text_notes(msg.content)

        # Build the response payload pi-bench's purple_adapter understands.
        # Its _part_to_pi_msg expects tool_calls as a flat list of
        # {id, name, arguments} (NOT the OpenAI SDK's nested
        # {id, type, function: {name, arguments}} shape).
        data: dict[str, Any] = {}
        if msg.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in msg.tool_calls
            ]
        if cleaned_content:
            data["content"] = cleaned_content

        _trace(
            "a2a_outgoing",
            handler="openai_passthrough",
            raw_content=msg.content,
            cleaned_content=cleaned_content,
            tool_calls=data.get("tool_calls"),
            notes_state=list(self._notes),
        )

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=data))],
            name="openai_response",
        )

    async def run_car_bench(
        self,
        message: Message,
        event_queue: Any,
        context_id: str,
    ) -> None:
        """CAR-bench protocol (in-car voice assistant) — Message mode.

        Per turn the green sends a mix of TextPart and DataPart:
        - TextPart on first turn: 'System:\\n<policy>\\n\\nUser:\\n<utterance>'
        - TextPart on later turns: just the new user utterance
        - DataPart on first turn: {"tools": [...openai-style schemas...]}
        - DataPart on later turns: {"tool_results": [{"tool_name", "content"}]}

        We reply with a Message (NOT a Task update) containing a list
        of Parts:
        - TextPart with content (for user-facing replies)
        - DataPart with {"tool_calls": [{"tool_name", "arguments": {...}}]}
          (flat shape — CAR-bench identifies by NAME, not id)

        Why Message mode: the green's tool_provider asserts
        `status == "completed"` AND keeps reusing the same task_id
        across turns. Those two requirements are mutually exclusive in
        a Task lifecycle (terminal tasks reject new sends). Message
        responses have no task lifecycle, so the green never sees a
        terminal task; each turn the conversation continues cleanly.

        State across turns lives on self: history (OpenAI-format),
        _car_bench_tools (cached schemas), and _car_bench_pending_tool_calls
        (so we can map next turn's tool_results back to ids by name).
        """
        payload, raw_text = extract_payload(message)
        # ── parse incoming TextPart ─────────────────────────────────
        user_text: str | None = None
        if raw_text:
            if "System:" in raw_text and "\n\nUser:" in raw_text:
                sys_part, user_part = raw_text.split("\n\nUser:", 1)
                sys_prompt = sys_part.replace("System:", "", 1).strip()
                user_text = user_part.strip()
                # Replace our generic system prompt with the green's
                # benchmark-specific one. Subsequent turns will keep it.
                if self.history and self.history[0].get("role") == "system":
                    self.history[0]["content"] = sys_prompt
                else:
                    self.history.insert(
                        0, {"role": "system", "content": sys_prompt}
                    )
            else:
                user_text = raw_text

        # ── parse incoming DataPart ─────────────────────────────────
        incoming_tools: list[dict[str, Any]] | None = None
        incoming_tool_results: list[dict[str, Any]] | None = None
        if isinstance(payload, dict):
            if isinstance(payload.get("tools"), list):
                incoming_tools = payload["tools"]
            if isinstance(payload.get("tool_results"), list):
                incoming_tool_results = payload["tool_results"]

        if incoming_tools is not None:
            self._car_bench_tools = incoming_tools

        # ── ingest tool_results from prior turn ─────────────────────
        if (
            self._car_bench_pending_tool_calls
            and incoming_tool_results is not None
        ):
            # Map results back to ids by NAME (CAR-bench's convention).
            # Pop matched entries so duplicate tool names get distinct ids.
            calls_by_name: dict[str, list[dict[str, Any]]] = {}
            for tc in self._car_bench_pending_tool_calls:
                calls_by_name.setdefault(
                    tc["function"]["name"], []
                ).append(tc)
            for tr in incoming_tool_results:
                name = tr.get("tool_name", "")
                matched = calls_by_name.get(name) or []
                if matched:
                    tc = matched.pop(0)
                    tool_call_id = tc["id"]
                else:
                    tool_call_id = f"unknown_{name}"
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": _truncate_tool_output(
                        str(tr.get("content", ""))
                    ),
                })
            self._car_bench_pending_tool_calls = None
        elif user_text:
            self.history.append({"role": "user", "content": user_text})

        # ── LLM call with CAR-bench's tools ─────────────────────────
        # Inject working-notes (if any) + a one-time tools-disabled
        # notes instruction so the model knows to use [NOTE-ADD: ...]
        # markup. Notes go right after the existing system prompt.
        notes_msg = self._notes_system_message()
        notes_instr = {"role": "system", "content": _NOTES_TOOL_INSTRUCTION_TEXT}
        if self.history and self.history[0].get("role") == "system":
            prefix: list[dict[str, Any]] = [self.history[0]]
            if notes_msg:
                prefix.append(notes_msg)
            prefix.append(notes_instr)
            messages = prefix + self.history[1:]
        else:
            prefix = []
            if notes_msg:
                prefix.append(notes_msg)
            prefix.append(notes_instr)
            messages = prefix + list(self.history)

        kwargs: dict[str, Any] = {
            "model": model_name(),
            "messages": messages,
        }
        if self._car_bench_tools:
            kwargs["tools"] = self._car_bench_tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = False
        if not reasoning_enabled():
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}

        resp = await _chat_completions_with_retry(self.client, **kwargs)
        msg = resp.choices[0].message

        # ── record assistant turn in history (full OpenAI shape) ────
        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
        }
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        self.history.append(assistant_entry)

        # ── build the outgoing Parts list ───────────────────────────
        parts: list[Part] = []
        if msg.content:
            parts.append(Part(root=TextPart(text=msg.content)))

        if msg.tool_calls:
            flat_calls: list[dict[str, Any]] = []
            for tc in msg.tool_calls:
                args = _args_dict(tc.function.arguments)
                flat_calls.append({
                    "tool_name": tc.function.name,
                    "arguments": args,
                })
            parts.append(
                Part(root=DataPart(data={"tool_calls": flat_calls}))
            )
            # Stash the openai-shape calls so next turn's tool_results can
            # be matched by name -> id.
            self._car_bench_pending_tool_calls = assistant_entry["tool_calls"]
        else:
            self._car_bench_pending_tool_calls = None

        if not parts:
            # Defensive: model returned nothing at all (e.g. reasoning
            # eat-the-completion bug). Send an empty TextPart so the
            # green's validator doesn't blow up on missing parts.
            parts.append(Part(root=TextPart(text="")))

        _trace(
            "a2a_outgoing",
            handler="car_bench",
            raw_content=msg.content,
            text_parts=[
                p.root.text for p in parts if isinstance(p.root, TextPart)
            ],
            data_parts=[
                p.root.data for p in parts if isinstance(p.root, DataPart)
            ],
            notes_state=list(self._notes),
        )

        # Emit a Message event directly — no Task creation, no
        # add_artifact, no update_status. The A2A SDK request handler
        # turns this into a Message-kind JSON-RPC response; the green's
        # sync_client picks it up as `raw_message` and reads parts off
        # it. Because there is no task, there is no task lifecycle to
        # collide with the green's next-turn reuse of any task_id.
        await event_queue.enqueue_event(
            new_agent_parts_message(parts, context_id=context_id)
        )

    async def _terminal_shell(
        self,
        payload: Any,
        raw_text: str,
        updater: TaskUpdater,
    ) -> None:
        """terminal-bench-shell-v1 state machine.

        Same system prompt and same toolset as every other handler. The
        only difference is I/O routing: when the model calls shell_exec,
        we pause the loop, emit an exec_request as the A2A response, and
        wait for the green agent to ship back an exec_result on the next
        A2A turn.
        """
        kind = payload.get("kind") if isinstance(payload, dict) else None

        if kind == "task":
            instruction = payload.get("instruction") or raw_text
            self.history.append({"role": "user", "content": instruction})
            self._record_history("task", instruction)
        elif kind == "exec_result":
            # Examine the raw fields BEFORE we merge them, so a log file
            # echoed via stdout that contains "timed out" doesn't
            # accidentally trigger our timeout recovery nudge.
            raw_exit_code = payload.get("exit_code")
            raw_stderr = payload.get("stderr", "") or ""
            timed_out = _is_sandbox_timeout(raw_exit_code, raw_stderr)
            result_text = _truncate_tool_output(
                f"exit_code: {raw_exit_code}\n"
                f"stdout:\n{payload.get('stdout', '')}\n"
                f"stderr:\n{raw_stderr}"
            )
            self._record_history(
                "tool_result",
                result_text,
                extra={"exit_code": raw_exit_code, "timed_out": timed_out},
            )
            if self.pending_protocol_tool_id is None:
                # Out-of-band exec_result; treat as user content so the
                # conversation can still continue.
                self.history.append(
                    {"role": "user", "content": result_text}
                )
            else:
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": self.pending_protocol_tool_id,
                        "content": result_text,
                    }
                )
                self.pending_protocol_tool_id = None
            if timed_out:
                self.history.append({
                    "role": "user",
                    "content": _TIMEOUT_RECOVERY_NUDGE,
                })
        else:
            await self._send_protocol_message(
                updater,
                {
                    "kind": "final",
                    "error": f"unknown protocol kind: {kind!r}",
                },
            )
            return

        tools = GENERIC_TOOL_SCHEMAS if tools_enabled() else []

        # Both shell_exec AND python_exec must run inside the benchmark's
        # sandbox, not in our local agent container — so we pause on both
        # and translate python_exec into `python3 -c <code>` before
        # emitting the exec_request.
        final_text, paused = await self._chat_loop(
            tools=tools,
            updater=updater,
            workdir=self.workdir(),
            max_steps=200,
            pause_on={"shell_exec", "python_exec"},
        )

        if paused:
            tc = paused[0]
            args = _args_dict(tc.get("arguments"))
            self.pending_protocol_tool_id = tc["id"]
            if tc["name"] == "python_exec":
                command = f"python3 -c {shlex.quote(args.get('code', ''))}"
            else:
                command = args.get("command", "")
            # Record this tool call in our history log so the model can
            # grep /tmp/.agent/history.jsonl in subsequent turns.
            self._record_history(
                "tool_call",
                command,
                name=tc["name"],
                extra={"tool_call_id": tc["id"]},
            )
            # Prepend a heartbeat that writes the latest history entry
            # into the sandbox. base64-safe; failure does not break
            # the user-visible command.
            preface = self._history_jsonl_preface()
            # Cap each command at 25s so the green's hard 30s sandbox
            # kill never fires — `timeout 25` exits with code 124, which
            # is what _is_sandbox_timeout already recognizes, so the
            # model gets the existing timeout-recovery nudge. Skip if
            # the model already added `timeout` themselves.
            stripped = command.lstrip()
            if not stripped.startswith("timeout "):
                command = f"timeout 25 sh -c {shlex.quote(command)}"
            # History preface runs FIRST, then the user command (with
            # its own timeout wrap). Using ; not && — the preface is
            # best-effort, not a gate.
            if preface:
                command = preface + command
            await self._send_protocol_message(
                updater,
                {
                    "kind": "exec_request",
                    "command": command,
                },
            )
        else:
            await self._send_protocol_message(
                updater,
                {"kind": "final", "summary": final_text},
            )

    async def _send_protocol_message(
        self, updater: TaskUpdater, payload: dict[str, Any]
    ) -> None:
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=json.dumps(payload)))],
            name="protocol_message",
        )

    async def _swe_bench(
        self,
        payload: dict[str, Any],
        updater: TaskUpdater,
    ) -> None:
        """Coding-bench-style: green ships a problem statement + a Docker
        image containing the repo, and expects a unified diff back.

        We pull the image, start the container, let the LLM investigate
        and edit files in it via shell_exec/read_file/write_file, then
        run `git diff` ourselves to extract the patch. The LLM never
        has to format a diff — it just edits.
        """
        instance_id = str(payload.get("instance_id", "instance"))
        problem = payload.get("problem_statement", "")
        image = payload.get("docker_image", "")
        base_commit = payload.get("base_commit", "")
        repo = payload.get("repo", "")
        hints = payload.get("hints", "") or ""

        if not image:
            await updater.add_artifact(
                parts=[Part(root=TextPart(
                    text="```diff\n# no docker_image in payload\n```"
                ))],
                name="patch",
            )
            return

        docker_wd = DockerWorkdir(image=image, name_hint=f"swe-{instance_id}")
        self._workdir = docker_wd

        try:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"pulling {image}..."),
            )
            try:
                await docker_wd.start()
            except RuntimeError as e:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(
                        text=f"```diff\n# docker setup failed: {e}\n```"
                    ))],
                    name="patch",
                )
                return

            repo_path = docker_wd.repo_path or "/"

            user_message = (
                f"You are working in a Linux container. The repository "
                f"is checked out at {repo_path} (base commit {base_commit}, "
                f"repo {repo!r}).\n\n"
                f"Problem to solve:\n{problem}\n\n"
                + (f"Hints:\n{hints}\n\n" if hints else "")
                + "Investigate the repo with shell_exec / read_file, then "
                "use write_file (or shell_exec) to apply your fix. "
                "When you're satisfied your fix is complete, stop calling "
                "tools — your changes will be captured as a unified diff "
                "via `git diff` and returned as the answer."
            )
            self.history.append({"role": "user", "content": user_message})

            tools = GENERIC_TOOL_SCHEMAS if tools_enabled() else []
            await self._chat_loop(
                tools=tools,
                updater=updater,
                workdir=docker_wd,
                max_steps=200,
            )

            diff = await docker_wd.diff()
            patch_text = (
                f"```diff\n{diff}\n```" if diff.strip() else "```diff\n```"
            )
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=patch_text))],
                name="patch",
            )
        finally:
            try:
                await docker_wd.cleanup()
            except Exception:  # noqa: BLE001
                pass

    # ---- shared LLM loop ----------------------------------------------

    async def _chat_loop(
        self,
        tools: list[dict[str, Any]],
        updater: TaskUpdater,
        workdir: Workdir | None,
        max_steps: int,
        pause_on: set[str] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Run a tool-use loop until either:
          (a) the model returns plain content (no tool_calls), or
          (b) the model calls a tool whose name is in pause_on.

        In case (a), returns (text, None).
        In case (b), returns ("", [tool_call_dict, ...]) — the caller is
        responsible for emitting those tool calls over the wire and
        ingesting their results on a subsequent invocation. The paused
        tool_call assistant entry is already appended to self.history.

        Appends every turn (assistant + tool results) to self.history.
        """
        pause_on = pause_on or set()
        for step in range(max_steps):
            # Inject working notes as a transient system message at the
            # top of context (NOT mutated into self.history — we want
            # the freshest version every turn, including any notes the
            # model just wrote in the previous step).
            notes_msg = self._notes_system_message()
            messages = (
                [self.history[0], notes_msg] + self.history[1:]
                if notes_msg and self.history
                else self.history
            )
            kwargs: dict[str, Any] = {
                "model": model_name(),
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
                # Discourage parallel calls so a paused call doesn't
                # leave siblings without tool results in history.
                kwargs["parallel_tool_calls"] = False
            if json_output():
                kwargs["response_format"] = {"type": "json_object"}
            if not reasoning_enabled():
                kwargs["extra_body"] = {"reasoning": {"enabled": False}}

            resp = await _chat_completions_with_retry(self.client, **kwargs)
            msg = resp.choices[0].message
            # Parse any [NOTE-ADD/-REMOVE] markers out of msg.content.
            # The model could use either the tool OR text markup; we
            # accept both. Cleaned content goes into history so the
            # markers don't pollute the model's view of its own past.
            raw_content_before_notes = msg.content
            if msg.content:
                msg.content = self._extract_text_notes(msg.content)
            if raw_content_before_notes != msg.content:
                _trace(
                    "chat_loop_notes_stripped",
                    step=step,
                    raw=raw_content_before_notes,
                    cleaned=msg.content,
                    notes_state=list(self._notes),
                )

            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content,
            }
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            self.history.append(assistant_entry)

            if not msg.tool_calls:
                _trace(
                    "chat_loop_final",
                    step=step,
                    final_text=msg.content or "",
                    notes_state=list(self._notes),
                )
                return msg.content or "", None

            paused_calls = [
                tc for tc in msg.tool_calls
                if tc.function.name in pause_on
            ]
            if paused_calls:
                # Strip non-paused siblings (if any) to keep history
                # consistent — every assistant.tool_calls entry must be
                # followed by a tool result for each id before the next
                # assistant turn.
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in paused_calls
                ]
                return "", [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in paused_calls
                ]

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"step {step + 1}: calling "
                    + ", ".join(tc.function.name for tc in msg.tool_calls)
                ),
            )

            for tc in msg.tool_calls:
                args = _args_dict(tc.function.arguments)
                # Record the tool call in the searchable history log.
                self._record_history(
                    "tool_call",
                    json.dumps(args, ensure_ascii=False)[:2000],
                    name=tc.function.name,
                    extra={"tool_call_id": tc.id},
                )
                # Intercept the `note` tool here (needs Agent state,
                # not available inside tool_dispatch).
                if tc.function.name == "note":
                    result = self._handle_note_tool(args)
                else:
                    result = await tool_dispatch(
                        tc.function.name, args, workdir=workdir
                    )
                truncated = _truncate_tool_output(result)
                self._record_history(
                    "tool_result",
                    truncated,
                    name=tc.function.name,
                    extra={"tool_call_id": tc.id},
                )
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": truncated,
                    }
                )
                # Only nudge when an execution tool produced our own
                # timeout marker — never on text content tools where
                # "timed out" could appear in legitimate output.
                if (
                    tc.function.name in ("shell_exec", "python_exec")
                    and _is_exec_timeout(result)
                ):
                    self.history.append({
                        "role": "user",
                        "content": _TIMEOUT_RECOVERY_NUDGE,
                    })

        return (
            "[ran out of tool-use steps; returning best partial answer]\n"
            + (self.history[-1].get("content") or ""),
            None,
        )
