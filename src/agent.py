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
import shlex
import tempfile
from typing import Any

import openai
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
    """
    delay = 1.0
    for attempt in range(max_attempts):
        try:
            return await client.chat.completions.create(**kwargs)
        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
            json.JSONDecodeError,
        ):
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2


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
"""


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

    def workdir(self) -> Workdir:
        if self._workdir is None:
            self._workdir = LocalWorkdir(tempfile.mkdtemp(prefix="agentswe-"))
        return self._workdir

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        payload, raw_text = extract_payload(message)

        # Sticky handler: classify on first message, keep using it.
        if self.handler is None:
            self.handler = classify(payload, raw_text)

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
        if msg.content:
            data["content"] = msg.content

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
        kwargs: dict[str, Any] = {
            "model": model_name(),
            "messages": self.history,
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
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
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
        elif kind == "exec_result":
            result_text = _truncate_tool_output(
                f"exit_code: {payload.get('exit_code')}\n"
                f"stdout:\n{payload.get('stdout', '')}\n"
                f"stderr:\n{payload.get('stderr', '')}"
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
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            self.pending_protocol_tool_id = tc["id"]
            if tc["name"] == "python_exec":
                command = f"python3 -c {shlex.quote(args.get('code', ''))}"
            else:
                command = args.get("command", "")
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
            kwargs: dict[str, Any] = {
                "model": model_name(),
                "messages": self.history,
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
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await tool_dispatch(
                    tc.function.name, args, workdir=workdir
                )
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _truncate_tool_output(result),
                    }
                )

        return (
            "[ran out of tool-use steps; returning best partial answer]\n"
            + (self.history[-1].get("content") or ""),
            None,
        )
