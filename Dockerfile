# Stage 1: pull the baked-in OfficeQA FAISS index + Treasury Bulletin
# corpus from our existing specialized agent image. Multi-GB blobs that
# can't be reliably fetched at runtime — bake them into the image.
FROM --platform=linux/amd64 ghcr.io/soumya-batra/officeqa-agent:latest AS officeqa-data

# Stage 2: the real agent image.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Docker CLI is needed by the SWE-bench handler. The daemon is the
# host's, reached via /var/run/docker.sock mounted by Amber.
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*

# Copy the OfficeQA corpus + FAISS index. Paths match what the
# manifest sets via CORPUS_DIR and FAISS_INDEX_DIR.
COPY --from=officeqa-data /app/corpus /app/corpus
COPY --from=officeqa-data /app/faiss_index /app/faiss_index

# Run as root so Amber's framework.docker mount can bind a Unix socket
# at /var/run/docker.sock inside the container. Container isolation is
# already broken by the host docker socket mount itself.
WORKDIR /root

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Use the baked venv's python directly — skips `uv run`'s per-launch
# venv revalidation (saves ~1-2s of cold start). Some leaderboards
# have tight readiness windows; faster startup = fewer false fails.
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/root/.venv/bin/python", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9010"]
EXPOSE 9010
