"""Quick sanity test for OfficeQA retrieval integration.

Runs INSIDE the just-built agentswe-general docker image so the baked
faiss_index + corpus + Python deps are all in place. Calls our
`retrieve_documents` with a sample Treasury Bulletin question and
prints the top passages.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  ./tools/repro_officeqa_retrieval.py "<question>"
  (defaults to a sample question if you don't pass one)
"""

import os
import subprocess
import sys

IMAGE = "ghcr.io/soumya-batra/agentswe-general:latest"
QUERY = sys.argv[1] if len(sys.argv) > 1 else (
    "What was the total Treasury debt held by foreign and international "
    "investors at the end of fiscal year 1985?"
)


def main():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Mount our LOCAL src/ over the image's so we test the same code
    # that's on disk (not just whatever was baked into :latest).
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cmd = [
        "/usr/local/bin/docker", "run", "--rm", "--platform", "linux/amd64",
        "-e", f"OPENROUTER_API_KEY={key}",
        "-e", "RETRIEVAL_TOP_K=5",  # keep output short for the test
        "-e", "RETRIEVAL_ENABLED=true",
        "-v", f"{repo}/src:/root/src:ro",
        "--entrypoint", "sh", IMAGE,
        "-c",
        f'''cd /root && uv run python -c "
import asyncio, sys
sys.path.insert(0, 'src')
from tools import retrieve_documents
query = {QUERY!r}
print(f'Query: {{query}}')
print('=' * 70)
out = asyncio.run(retrieve_documents(query=query))
print(out)
print()
print('=' * 70)
print(f'Result length: {{len(out)}} chars')
"
'''
    ]
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
