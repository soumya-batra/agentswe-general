FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Docker CLI is needed by the SWE-bench handler. The daemon is the
# host's, reached via /var/run/docker.sock mounted by Amber.
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*

# Run as root so Amber's framework.docker mount can bind a Unix socket
# at /var/run/docker.sock inside the container. Container isolation is
# already broken by the host docker socket mount itself.
WORKDIR /root

COPY pyproject.toml README.md ./
COPY src src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9010"]
EXPOSE 9010
