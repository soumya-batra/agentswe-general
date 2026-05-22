FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Docker CLI is needed by the SWE-bench handler. The daemon is the
# host's, reached via /var/run/docker.sock mounted by Amber.
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos "" agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml README.md ./
COPY src src

RUN --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9010"]
EXPOSE 9010
