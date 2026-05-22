"""Generic tools available to the LLM loop.

The tool surface is the same regardless of which benchmark is on the
other end of A2A. What differs is the WORKDIR the tools operate in:

  - LocalWorkdir   — subprocess in a /tmp scratch dir. Used for plain
                     Q&A, browsing, dialogue, etc.
  - DockerWorkdir  — `docker exec` into a long-lived container. Used
                     for SWE-bench, where the green agent ships a
                     proprietary repo as a Docker image and expects a
                     patch.

Each workdir implements the same three methods: shell_exec, read_file,
write_file. The tool dispatcher just forwards to whichever workdir
the current handler set up.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)


WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web. Returns a list of titles, URLs, and snippets. "
            "Use this when you need to find information you don't already know."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
}

WEB_FETCH = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "Fetch the contents of a URL as text. Truncates very long pages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
    },
}

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the working environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
}

WRITE_FILE = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file in the working environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
}

SHELL_EXEC = {
    "type": "function",
    "function": {
        "name": "shell_exec",
        "description": (
            "Run a shell command in the working environment. Returns "
            "stdout, stderr, and exit code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_s": {"type": "integer"},
            },
            "required": ["command"],
        },
    },
}


GENERIC_TOOL_SCHEMAS = [WEB_SEARCH, WEB_FETCH, SHELL_EXEC, READ_FILE, WRITE_FILE]


# ---------------------------------------------------------------------------
# Workdir abstraction


class Workdir(Protocol):
    async def shell_exec(self, command: str, timeout_s: int = 60) -> str: ...
    async def read_file(self, path: str, max_bytes: int = 200_000) -> str: ...
    async def write_file(self, path: str, content: str) -> str: ...
    async def cleanup(self) -> None: ...


def _format_result(exit_code: int, stdout: str, stderr: str) -> str:
    return (
        f"exit_code: {exit_code}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


class LocalWorkdir:
    """A scratch directory on the agent's local filesystem."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(path, exist_ok=True)

    async def shell_exec(self, command: str, timeout_s: int = 60) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self.path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            return f"[error: timed out after {timeout_s}s]"
        return _format_result(
            proc.returncode or 0,
            stdout_b.decode("utf-8", "replace"),
            stderr_b.decode("utf-8", "replace"),
        )

    def _resolve(self, path: str) -> str | None:
        full = os.path.abspath(os.path.join(self.path, path))
        base = os.path.abspath(self.path)
        if full != base and not full.startswith(base + os.sep):
            return None
        return full

    async def read_file(self, path: str, max_bytes: int = 200_000) -> str:
        full = self._resolve(path)
        if full is None:
            return f"[error: path escapes workdir: {path}]"
        try:
            with open(full, "rb") as f:
                data = f.read(max_bytes + 1)
        except FileNotFoundError:
            return f"[error: file not found: {path}]"
        except IsADirectoryError:
            return "\n".join(sorted(os.listdir(full)))
        truncated = len(data) > max_bytes
        text = data[:max_bytes].decode("utf-8", errors="replace")
        if truncated:
            text += f"\n[...truncated at {max_bytes} bytes]"
        return text

    async def write_file(self, path: str, content: str) -> str:
        full = self._resolve(path)
        if full is None:
            return f"[error: path escapes workdir: {path}]"
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[wrote {len(content)} chars to {path}]"

    async def cleanup(self) -> None:
        # Leave local workdirs in place; the container is ephemeral.
        return None


class DockerWorkdir:
    """A long-lived `docker run`+`docker exec` workdir.

    Pulls the image, starts a container with `sleep infinity` so we can
    exec into it across many tool calls, and tears it down on cleanup.

    Requires the host's Docker socket to be reachable at
    /var/run/docker.sock (see Amber's framework.docker mount).
    """

    def __init__(self, image: str, name_hint: str = "agentswe") -> None:
        self.image = image
        self.name = f"{name_hint}-{os.urandom(4).hex()}"
        self.container_id: str | None = None
        self.repo_path: str | None = None  # set by _discover_repo

    async def _docker(
        self, *args: str, timeout_s: int = 300, input_bytes: bytes | None = None
    ) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "docker", *args,
            stdin=asyncio.subprocess.PIPE if input_bytes else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=input_bytes), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return -1, "", f"timed out after {timeout_s}s"
        return (
            proc.returncode or 0,
            stdout_b.decode("utf-8", "replace"),
            stderr_b.decode("utf-8", "replace"),
        )

    async def start(self) -> None:
        rc, _, err = await self._docker("pull", self.image, timeout_s=900)
        if rc != 0:
            raise RuntimeError(f"docker pull failed: {err}")
        rc, out, err = await self._docker(
            "run", "-d",
            "--name", self.name,
            "--entrypoint", "/bin/sh",
            self.image, "-c", "sleep infinity",
        )
        if rc != 0:
            raise RuntimeError(f"docker run failed: {err}")
        self.container_id = out.strip()
        await self._discover_repo()

    async def _discover_repo(self) -> None:
        rc, out, _ = await self._docker(
            "exec", self.name, "sh", "-c",
            "find / -name .git -type d 2>/dev/null | head -1",
            timeout_s=60,
        )
        if rc == 0 and out.strip():
            git_dir = out.strip().splitlines()[0]
            self.repo_path = git_dir.removesuffix("/.git") or "/"

    async def shell_exec(self, command: str, timeout_s: int = 60) -> str:
        if not self.container_id:
            return "[error: docker workdir not started]"
        cwd = self.repo_path or "/"
        rc, stdout, stderr = await self._docker(
            "exec", "-w", cwd, self.name, "sh", "-c", command,
            timeout_s=timeout_s,
        )
        return _format_result(rc, stdout, stderr)

    async def read_file(self, path: str, max_bytes: int = 200_000) -> str:
        if not self.container_id:
            return "[error: docker workdir not started]"
        # Resolve relative paths against the repo.
        target = path if path.startswith("/") else f"{self.repo_path or '/'}/{path}"
        rc, out, err = await self._docker(
            "exec", self.name, "sh", "-c",
            f"head -c {max_bytes + 1} -- {_shquote(target)}",
            timeout_s=60,
        )
        if rc != 0:
            return f"[error reading {path}: {err.strip() or 'exit ' + str(rc)}]"
        if len(out) > max_bytes:
            out = out[:max_bytes] + f"\n[...truncated at {max_bytes} bytes]"
        return out

    async def write_file(self, path: str, content: str) -> str:
        if not self.container_id:
            return "[error: docker workdir not started]"
        target = path if path.startswith("/") else f"{self.repo_path or '/'}/{path}"
        rc, _, err = await self._docker(
            "exec", "-i", self.name, "sh", "-c",
            f"mkdir -p $(dirname {_shquote(target)}) && cat > {_shquote(target)}",
            input_bytes=content.encode("utf-8"),
            timeout_s=60,
        )
        if rc != 0:
            return f"[error writing {path}: {err.strip()}]"
        return f"[wrote {len(content)} chars to {target}]"

    async def diff(self) -> str:
        """Return `git diff` over the repo (used by SWE-bench)."""
        if not self.repo_path:
            return ""
        rc, out, _ = await self._docker(
            "exec", "-w", self.repo_path, self.name,
            "git", "diff",
            timeout_s=60,
        )
        return out if rc == 0 else ""

    async def cleanup(self) -> None:
        if self.container_id:
            await self._docker("rm", "-f", self.name, timeout_s=30)
            self.container_id = None


def _shquote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
# Web tools (don't need a workdir)


async def web_search(query: str, max_results: int = 5) -> str:
    import httpx
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return (
            "[web_search disabled: TAVILY_API_KEY not configured. "
            "Answer from your own knowledge if possible.]"
        )
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    lines: list[str] = []
    for r in data.get("results", []):
        lines.append(
            f"## {r.get('title', '(untitled)')}\n"
            f"{r.get('url', '')}\n"
            f"{r.get('content', '')}"
        )
    return "\n\n".join(lines) or "[no results]"


async def web_fetch(url: str) -> str:
    import httpx
    async with httpx.AsyncClient(
        timeout=30, follow_redirects=True
    ) as client:
        resp = await client.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; agentswe-general/0.1; "
                    "+https://agentbeats.dev)"
                ),
            },
        )
        resp.raise_for_status()
        text = resp.text
    cap = 100_000
    if len(text) > cap:
        text = text[:cap] + f"\n[...truncated, full length {len(text)}]"
    return text


# ---------------------------------------------------------------------------
# Tool dispatcher


async def dispatch(
    name: str,
    arguments: dict[str, Any],
    workdir: Workdir | None = None,
) -> str:
    try:
        if name == "web_search":
            return await web_search(**arguments)
        if name == "web_fetch":
            return await web_fetch(**arguments)

        if workdir is None:
            return f"[error: {name} requires a workdir; none available]"

        if name == "shell_exec":
            return await workdir.shell_exec(**arguments)
        if name == "read_file":
            return await workdir.read_file(**arguments)
        if name == "write_file":
            return await workdir.write_file(**arguments)
        return f"[error: unknown tool '{name}']"
    except TypeError as e:
        return f"[error: bad arguments for {name}: {e}]"
    except Exception as e:  # noqa: BLE001
        return f"[error: {type(e).__name__}: {e}]"
