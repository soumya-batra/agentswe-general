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
from a2a.utils import new_agent_text_message

from llm import json_output, make_client, model_name, tools_enabled
from messenger import Messenger
from tools import (
    GENERIC_TOOL_SCHEMAS,
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


def classify(payload: Any) -> str:
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

    def workdir(self) -> Workdir:
        if self._workdir is None:
            self._workdir = LocalWorkdir(tempfile.mkdtemp(prefix="agentswe-"))
        return self._workdir

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        payload, raw_text = extract_payload(message)

        # Sticky handler: classify on first message, keep using it.
        if self.handler is None:
            self.handler = classify(payload)

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

        self.history.append({"role": "user", "content": user_content})

        # When TOOLS_ENABLED=off (e.g. tau2 deployment), the task's own
        # prompt has already given the model whatever tool surface it
        # needs as inline text; we must not expose our own OpenAI-format
        # tools or the model will mix surfaces.
        tools = GENERIC_TOOL_SCHEMAS if tools_enabled() else []

        final_text, paused = await self._chat_loop(
            tools=tools,
            updater=updater,
            workdir=self.workdir(),
            max_steps=12,
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
            max_steps=8,
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
                max_steps=20,
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
