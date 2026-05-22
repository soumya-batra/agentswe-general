import argparse

import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--card-url", type=str)
    args = parser.parse_args()

    skill = AgentSkill(
        id="general_purpose",
        name="General-purpose agent",
        description=(
            "Handles tasks across many AgentBeats benchmark categories: "
            "Q&A, web research, multi-turn dialogue, coding (via a "
            "Docker-mounted workspace), terminal-driven tasks, and "
            "policy/decision tasks."
        ),
        tags=[
            "general",
            "coding",
            "research",
            "qa",
            "dialogue",
            "policy",
            "agentbeats",
        ],
        examples=[
            "Answer a question grounded in a public document",
            "Fix a bug in a Docker-shipped repository and return a diff",
            "Solve a Linux terminal task via shell commands",
            "Make a policy-compliant decision for a given scenario",
        ],
    )

    agent_card = AgentCard(
        name="agentswe-general",
        description=(
            "General-purpose purple agent for AgentBeats Sprint 4. "
            "Dispatches incoming A2A messages by FORMAT (not benchmark "
            "name) to handlers that share a single system prompt and "
            "toolset, differing only in I/O translation."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="0.1.0",
        default_input_modes=["text", "application/json"],
        default_output_modes=["text", "application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
