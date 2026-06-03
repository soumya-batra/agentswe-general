# agentswe-general

**A single general-purpose A2A agent for AgentBeats Sprint 4, evaluated across five benchmarks in five distinct categories.**

## Abstract

This is our Sprint 4 entry: one purple agent that adapts to substantially different task types — coding, conversational policy, financial document question-answering, agentic safety, and computer-use navigation — without per-benchmark code paths, prompt hardcoding, or look-up tables. The same Docker image, the same model, and the same reasoning loop handle every category. What changes from one benchmark to another is only the thin adapter that translates the green agent's wire protocol into and out of our shared ReAct loop.

## What's inside

The agent runs a single tool-use loop powered by `z-ai/glm-5` via OpenRouter — a deliberate choice for cost efficiency that lets us complete a full five-benchmark evaluation for under thirty dollars. The model has access to a fixed tool surface: a shell, a Python interpreter, file I/O, web search and fetching, working-memory tooling, and optional document retrieval.

### Working memory (scratchpad)

We give the model an explicit scratchpad in which it accumulates short, structured entries it will need to reference later — facts, plans, constraints, decisions, blockers, and completed steps — using suggested prefixes such as `[FINDING: ...]`, `[PLAN: ...]`, `[CONSTRAINT: ...]`. The scratchpad is soft-capped and injected at the top of every turn so it survives history truncation. The model interacts with it through a dedicated `note` tool when tool calling is enabled, or through inline markup tags (`[NOTE-ADD: ...]`, `[NOTE-REMOVE: ...]`) when the deployment is configured tool-less — necessary for benchmarks whose green agent injects its own task-specific tool surface into the system prompt and requires us to keep ours hidden.

### Conversation history log

Alongside the scratchpad, every turn — the incoming task, every tool call the model issues, and every tool result that comes back — is appended to a JSONL history log. For benchmarks where the agent acts through a sandbox shell, this log is materialised inside the sandbox at `/tmp/.agent/history.jsonl` by prefacing each outgoing shell command with a tiny base64-encoded write. When the model needs to recall an earlier finding or revisit a command it has already run, it can `grep` / `tail` / `cat` the history file directly from its next shell turn, instead of re-running discovery work or relying on a long chat history that may have been truncated. This is the simplest of our long-context primitives and complements the scratchpad: the scratchpad is for short structured facts the model authors deliberately, the history log is the unedited record of *everything that happened*.

### Recursive in-agent interpreter

Beyond the scratchpad and the history log, the agent has a persistent in-process Python interpreter exposed via a `repl` tool. The interpreter has, in scope, the **untruncated** record of every prior shell command, every prior tool result, and every prior repl execution from this conversation. This lets the model slice, search, regex-match, JSON-parse, and summarise long outputs server-side, without re-running the underlying shell command or paying the token cost of re-reading raw outputs into its window. Taken together, the three primitives — scratchpad, history log, in-agent interpreter — give a single mid-tier model the ability to sustain coherent long-horizon work across many tool calls, turning context length from a fixed constraint into something the agent actively manages.

### Retrieval-augmented generation

For document-grounded benchmarks (currently OfficeQA), retrieval is enabled by a `retrieval_enabled` configuration flag. The document chunks are baked into the Docker image — not architecturally ideal, but a pragmatic choice given that leaderboard infrastructure cannot reliably download multi-gigabyte corpora at agent startup without timing out. Retrieval combines BM25 (lexical) and FAISS embeddings (Qwen3-Embedding-8B, semantic) and fuses the two ranked lists using **reciprocal rank fusion**. Beyond the top-k context that is pre-retrieved for the user's initial question, retrieval is also exposed to the model as a `retrieve_documents` tool, so it can issue more specific follow-up queries mid-solution (narrower years, named entities, exact figures) when the initial passages do not pin down the answer.

### Routing by message shape, not benchmark identity

The agent dispatches incoming messages by their **structure**, not their benchmark identity. A payload declaring a specific A2A protocol header is routed to the terminal-shell handler; a payload shaped like an OpenAI chat completion is forwarded as such; a coding-style payload bearing a Docker image and a problem statement is routed to the patch-generation flow; everything else falls through to a general handler. This routing is content-based and benchmark-agnostic: nowhere in the codebase does the agent ask "which benchmark am I in?".

## Results

| Benchmark | Category | Score |
|---|---|---|
| Terminal Bench 2.0 | Coding | 41 / 89 |
| τ²-Bench | τ² | 100 / 114 |
| OfficeQA | Finance | 114 / 246 |
| Pi-Bench | Agent Safety | 78.9 |
| CAR-bench | Computer Use & Web | 0.64 |

Five working benchmarks across five distinct AgentBeats categories — comfortably above the Sprint 4 eligibility floor of five greens in three categories.

## Cost

A full Terminal Bench sweep over all 89 tasks costs in the order of a few dollars at current GLM-5 prices; the other benchmarks are cheaper. The complete five-benchmark evaluation runs under USD 30, which matters when iteration is the bottleneck.

## Repository layout

```
src/
├─ server.py        # A2A server + agent card configuration
├─ executor.py      # A2A request handling
├─ agent.py         # ReAct loop, handlers, working memory, retrieval
├─ tools.py         # Tool schemas and dispatch (shell, python, web, files, note, repl, retrieval)
├─ llm.py           # OpenRouter client + config-flag accessors
└─ messenger.py     # A2A messaging utilities
amber-manifest.json5  # Deployment manifest
Dockerfile            # Multi-stage build with baked-in corpus
tools/                # Local reproduction harnesses + trace renderer
tests/                # A2A conformance tests
```

## Running locally

```bash
# Install dependencies
uv sync

# Run the server
export OPENROUTER_API_KEY=sk-or-...
uv run src/server.py
```

## Running with Docker

```bash
# Build the image
docker build -t agentswe-general .

# Run the container
docker run -p 9010:9010 -e OPENROUTER_API_KEY=sk-or-... agentswe-general
```

## Configuration

Per-deployment toggles are declared in `amber-manifest.json5` and read from environment variables at runtime:

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required)* | OpenAI-compatible inference endpoint |
| `MODEL_NAME` | `z-ai/glm-5` | Chat model |
| `TOOLS_ENABLED` | `false` | Master switch for our tool surface — turn off when the green provides its own tool API in the prompt |
| `JSON_OUTPUT` | `false` | Force JSON object responses — used by τ² |
| `REASONING_ENABLED` | `true` | Allow the model's hidden reasoning channel |
| `RETRIEVAL_ENABLED` | `false` | Expose the baked-in document corpus through `retrieve_documents` |
| `TAVILY_API_KEY` | empty | Enables `web_search` when set |

## Testing

```bash
uv sync --extra test
uv run pytest --agent-url http://localhost:9010
```

## Acknowledgments

Built on the [RDI-Foundation/agent-template](https://github.com/RDI-Foundation/agent-template). Evaluated through the AgentBeats platform across five green agents.
