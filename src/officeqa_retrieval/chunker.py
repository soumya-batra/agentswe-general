import re
import json
import hashlib
from pathlib import Path
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS_PROSE = 1200
MAX_TOKENS_TABLE = 3500
OVERLAP_TOKENS   = 150
MIN_CHARS        = 200

TABLE_LINE_RE = re.compile(r"^\|.+\|$")
JUNK          = {"TREASURY DEPARTMENT", "LIBRARY ROOM", "DATE LOANED", "BORROWER'S NAME"}  # fix 3


# ── utils ─────────────────────────────────────────────────────────────────────

def tok(text: str) -> int:
    return len(enc.encode(text))

def short_hash(text: str) -> str:
    return hashlib.md5(
        re.sub(r"\s+", " ", text).strip().lower().encode()
    ).hexdigest()

def split_tokens(text: str, max_tok: int = MAX_TOKENS_PROSE,
                 overlap: int = OVERLAP_TOKENS) -> list[str]:
    tokens = enc.encode(text)
    if len(tokens) <= max_tok:
        return [text]
    chunks = []
    start  = 0
    while start < len(tokens):
        end = min(start + max_tok, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        start += max_tok - overlap
    return chunks


# ── header / noise detection ──────────────────────────────────────────────────

def is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if re.match(r"^\d{1,3}$", s):
        return True
    if re.match(r"^[-\d\s\u2014]+$", s):  # ← add \u2014 for em-dash
        return True
    if any(j in s for j in JUNK):
        return True
    return False

def header_level(line: str) -> int | None:
    s = line.strip()
    if not s or s in JUNK:
        return None
    if len(s) > 80:
        return None
    if s.isupper() and 6 < len(s) <= 80:
        return 1
    if re.match(r"^[A-Z][a-zA-Z]+(?:\s+[A-Za-z]+){1,5}$", s) and len(s) <= 60:
        return 2
    return None

def is_toc(text: str) -> bool:
    lines = [l for l in text.split("\n") if l.strip()]
    hits  = sum(1 for l in lines if re.search(r"\d{1,3}(-\d{1,3})?$", l.strip()))
    return len(lines) >= 2 and hits >= 2

def is_caption_only(text: str) -> bool:      # fix 1: drop orphaned captions
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    # a caption-only block has all short lines with no sentence-ending punctuation
    return all(len(l) <= 120 and not re.search(r"[.!?]", l) for l in lines)


# ── section stack ─────────────────────────────────────────────────────────────

class SectionStack:
    def __init__(self):
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, text: str):
        self._stack = [(l, t) for l, t in self._stack if l < level]
        self._stack.append((level, text))

    def current(self) -> str:
        return " > ".join(t for _, t in self._stack) if self._stack else "UNKNOWN"


# ── table utilities ───────────────────────────────────────────────────────────

def _dedup_table(table_text: str) -> str:
    lines = table_text.split("\n")
    mid   = len(lines) // 2
    if mid > 4 and re.match(r"^\| ?---", lines[mid].strip()):
        return "\n".join(lines[:mid])
    return table_text

def _parse_markdown_table(table_text: str) -> dict:
    lines = [l for l in table_text.split("\n")
             if l.strip().startswith("|")]
    if not lines:
        return {}
    headers = [
        c.strip() for c in lines[0].split("|")
        if c.strip() and c.strip() != "---"
    ]
    rows = []
    for line in lines[2:6]:
        if re.match(r"^\|[-| ]+\|$", line):
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells:
            rows.append(cells)
    return {"headers": headers, "sample_rows": rows}

def _clean_header(h: str) -> str:
    parts = [p for p in h.split(">") if "unnamed" not in p.lower()]
    return " > ".join(p.strip() for p in parts[-2:]) if parts else h

def summarise_table(table_text: str, section: str) -> str:
    info = _parse_markdown_table(table_text)
    if not info:
        return f"Table in section: {section}"

    col_str  = ", ".join(_clean_header(h) for h in info["headers"][:8])
    examples = []
    for row in info["sample_rows"][:3]:
        if row:
            label = row[0]
            vals  = row[1:4]
            if any(re.search(r"\d", v) for v in vals):
                examples.append(f"{label}: {', '.join(vals)}")

    summary = f"Table columns: {col_str}."
    if examples:
        summary += " Sample values — " + "; ".join(examples) + "."
    return summary


# ── context capture ───────────────────────────────────────────────────────────

def context_before(lines: list[str], start: int, max_chars: int = 400) -> str:
    buf = []
    j   = start - 1
    while j >= 0 and sum(len(l) for l in buf) < max_chars:
        l = lines[j].strip()
        if not l:
            j -= 1
            continue
        if any(k in l for k in JUNK):
            break
        if header_level(l) is not None:
            break
        if TABLE_LINE_RE.match(l):
            break
        buf.insert(0, l)
        j -= 1
    return "\n".join(buf)

def context_after(lines: list[str], start: int, max_chars: int = 200) -> str:
    buf = []
    j   = start
    while j < len(lines) and sum(len(l) for l in buf) < max_chars:
        l = lines[j].strip()
        if (is_noise(l)
                or TABLE_LINE_RE.match(l)
                or header_level(l) is not None):
            break
        buf.append(l)
        j += 1
    return "\n".join(buf)


# ── main chunker ──────────────────────────────────────────────────────────────

def iter_chunks(content: str) -> list[dict]:
    lines   = content.splitlines()
    chunks: list[dict] = []
    section = SectionStack()
    i       = 0

    while i < len(lines):
        raw      = lines[i]
        stripped = raw.strip()

        if is_noise(stripped):
            i += 1
            continue

        lvl = header_level(stripped)
        if lvl is not None:
            section.push(lvl, stripped)
            i += 1
            continue

        # ── TABLE ─────────────────────────────────────────────────────────────
        if TABLE_LINE_RE.match(stripped):
            table_start = i
            table_lines = []
            while i < len(lines) and TABLE_LINE_RE.match(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1

            table_text = "\n".join(table_lines).strip()
            if len(table_text) < MIN_CHARS:
                continue

            table_text = _dedup_table(table_text)
            before     = context_before(lines, table_start)
            after      = context_after(lines, i)
            sec        = section.current()
            summary    = summarise_table(table_text, sec)

            full = "\n\n".join(filter(None, [
                before,
                "[TABLE]\n" + table_text,
                after,
            ]))

            chunks.append({
                "text":          full,
                "section":       sec,
                "type":          "table",
                "table_summary": summary,
            })
            continue

        # ── PROSE ──────────────────────────────────────────────────────────────
        prose_lines = []
        while i < len(lines):
            s = lines[i].strip()
            if is_noise(s):
                i += 1
                continue
            if header_level(s) is not None or TABLE_LINE_RE.match(s):
                break
            prose_lines.append(s)
            i += 1

        prose = "\n".join(prose_lines).strip()
        if len(prose) < MIN_CHARS or is_toc(prose) or is_caption_only(prose):  # fix 1
            continue

        sec = section.current()
        for part in split_tokens(prose, max_tok=MAX_TOKENS_PROSE):
            chunks.append({
                "text":    part,
                "section": sec,
                "type":    "text",
            })

    return _dedup_chunks(chunks)


def _dedup_chunks(chunks: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in chunks:
        key = short_hash(c["text"])
        if len(c["text"]) < MIN_CHARS or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ── enrichment ────────────────────────────────────────────────────────────────

def enrich(chunk: dict) -> str:
    sec  = chunk["section"]
    text = chunk["text"]

    if chunk["type"] == "table":
        summary = chunk.get("table_summary", f"Table in {sec}")
        result = f"{sec}\n\n{summary}\n\n{text}".strip()
    else:
        result = f"{sec}\n\n{text}".strip()
    
    # hard cap at 8100 tokens to stay safely under the 8191 limit
    tokens = enc.encode(result)
    if len(tokens) > 8100:
        result = enc.decode(tokens[:8100])

    return result


# ── CLI / diagnostics ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path    = Path(sys.argv[1])
    content = path.read_text(encoding="utf-8", errors="ignore")

    lines        = content.splitlines()
    i            = 0
    table_sizes  = []
    while i < len(lines):
        if TABLE_LINE_RE.match(lines[i].strip()):
            tl = []
            while i < len(lines) and TABLE_LINE_RE.match(lines[i].strip()):
                tl.append(lines[i])
                i += 1
            table_sizes.append(tok("\n".join(tl)))
        else:
            i += 1

    if table_sizes:
        print(f"Table count      : {len(table_sizes)}")
        print(f"Max tokens       : {max(table_sizes)}")
        print(f"Mean tokens      : {sum(table_sizes) // len(table_sizes)}")
        print(f"Tables > 1200 tok: {sum(1 for t in table_sizes if t > 1200)}")
        print(f"Tables > 2000 tok: {sum(1 for t in table_sizes if t > 2000)}")
        print(f"Tables > 4500 tok: {sum(1 for t in table_sizes if t > 4500)}")

    raw_chunks = iter_chunks(content)
    enriched   = [enrich(c) for c in raw_chunks]

    seen, final = set(), []
    for e in enriched:
        key = short_hash(e)
        if key not in seen:
            seen.add(key)
            final.append(e)

    print(f"\nRaw chunks : {len(raw_chunks)}")
    print(f"Final      : {len(final)}")

    out = Path("faiss_index/chunks_enriched.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(final, indent=2))
    print(f"Saved to {out}")