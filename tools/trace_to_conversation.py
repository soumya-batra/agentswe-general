"""Convert an agent trace (JSONL from AGENT_TRACE_DIR) into a verbatim
conversation transcript — every message, every tool call, every result,
no summarization, no truncation, in chronological order.

Usage:
  uv run python tools/trace_to_conversation.py <trace.jsonl> > conversation.txt
"""

import json
import sys
from pathlib import Path


def main(path: str) -> None:
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

    print(f"=" * 80)
    print(f"CONVERSATION TRANSCRIPT — verbatim from {Path(path).name}")
    print(f"events: {len(events)}, elapsed: "
          f"{events[-1]['t']-events[0]['t']:.1f}s" if events else "empty")
    print(f"=" * 80)

    for i, e in enumerate(events):
        kind = e.get("event")
        ts = e.get("ts", "")

        if kind == "a2a_incoming":
            print(f"\n\n{'─'*78}")
            print(f"[{i:03d}] {ts}  A2A IN  message_id={e.get('message_id')}")
            print(f"{'─'*78}")
            if e.get("data_payload") is not None:
                print(f"<DataPart>\n{json.dumps(e['data_payload'], indent=2)}")
            if e.get("raw_text"):
                print(f"<TextPart>\n{e['raw_text']}")

        elif kind == "handler_dispatch":
            print(f"\n[{i:03d}] {ts}  HANDLER  → {e.get('handler')}")

        elif kind == "llm_call_request":
            print(f"\n\n{'═'*78}")
            print(f"[{i:03d}] {ts}  LLM CALL REQUEST")
            print(f"  model={e.get('model')}")
            print(f"  n_messages={e.get('n_messages')}")
            tools = e.get("tools") or []
            if tools:
                tool_names = [t.get("function", {}).get("name") for t in tools]
                print(f"  tools=[{', '.join(tool_names)}]  (count={len(tools)})")
                # Dump the full schema for `note` and `history` if present —
                # this is what the user wants to verify.
                for t in tools:
                    name = t.get("function", {}).get("name")
                    if name in ("note", "history", "history_search"):
                        print(f"  tool[{name}] schema:")
                        print(f"    {json.dumps(t, indent=4)}")
            print(f"  response_format={e.get('response_format')}")
            print(f"  extra_body={e.get('extra_body')}")
            print(f"{'═'*78}")
            print(f"FULL MESSAGES SENT TO LLM:")
            for j, m in enumerate(e.get("messages") or []):
                if not isinstance(m, dict):
                    print(f"\n  [msg {j}] (non-dict): {m}")
                    continue
                role = m.get("role", "?")
                content = m.get("content", "")
                tc = m.get("tool_calls")
                tcid = m.get("tool_call_id")
                head = f"\n  [msg {j}] role={role}"
                if tcid:
                    head += f" tool_call_id={tcid}"
                if m.get("name"):
                    head += f" name={m.get('name')}"
                print(head)
                if content:
                    # Word-for-word — no truncation.
                    for line in str(content).splitlines() or [""]:
                        print(f"    │ {line}")
                if tc:
                    for k, t in enumerate(tc):
                        print(f"    │ <tool_call[{k}] id={t.get('id')} "
                              f"name={t.get('function', {}).get('name')}>")
                        args = t.get("function", {}).get("arguments", "")
                        for line in str(args).splitlines() or [""]:
                            print(f"    │   {line}")

        elif kind == "llm_call_response":
            msg = e.get("msg", {})
            print(f"\n\n{'╌'*78}")
            print(f"[{i:03d}] {ts}  LLM RESPONSE  finish={e.get('finish_reason')}")
            print(f"  model_routed={e.get('model_routed')}")
            usage = e.get("usage")
            if usage:
                print(f"  usage={usage}")
            print(f"{'╌'*78}")
            print(f"  content_len={msg.get('content_len')}")
            content = msg.get("content") or ""
            if content:
                for line in content.splitlines() or [""]:
                    print(f"  │ {line}")
            tcs = msg.get("tool_calls") or []
            for k, t in enumerate(tcs):
                print(f"  │ <tool_call[{k}] id={t.get('id')} name={t.get('name')}>")
                args = t.get("arguments", "")
                for line in str(args).splitlines() or [""]:
                    print(f"  │   {line}")

        elif kind == "llm_empty_retry_triggered":
            print(f"\n[{i:03d}] {ts}  ⚠️  EMPTY-RESPONSE RETRY TRIGGERED")

        elif kind == "llm_empty_retry_response":
            msg = e.get("msg", {})
            print(f"[{i:03d}] {ts}  EMPTY-RETRY RESPONSE  finish={e.get('finish_reason')}  content_len={msg.get('content_len')}")
            content = msg.get("content") or ""
            for line in content.splitlines() or [""]:
                print(f"  │ {line}")

        elif kind == "chat_loop_notes_stripped":
            print(f"\n[{i:03d}] {ts}  NOTES STRIPPED  step={e.get('step')}")
            print(f"  raw   = {e.get('raw', '')!r}")
            print(f"  clean = {e.get('cleaned', '')!r}")
            print(f"  notes_state = {e.get('notes_state')}")

        elif kind == "chat_loop_final":
            print(f"\n[{i:03d}] {ts}  CHAT-LOOP FINAL  step={e.get('step')}  notes_state={e.get('notes_state')}")
            text = e.get("final_text", "") or ""
            for line in text.splitlines() or [""]:
                print(f"  │ {line}")

        elif kind == "history_append":
            entry = e.get("entry", {})
            print(f"\n[{i:03d}] {ts}  HISTORY APPEND  turn={entry.get('turn')} kind={entry.get('kind')} name={entry.get('name')}")
            t = entry.get("text", "") or ""
            for line in str(t).splitlines() or [""]:
                print(f"  │ {line}")

        elif kind == "a2a_outgoing":
            print(f"\n[{i:03d}] {ts}  A2A OUT  handler={e.get('handler')}")
            if "raw_content" in e and e["raw_content"]:
                print("  raw_content:")
                for line in str(e["raw_content"]).splitlines() or [""]:
                    print(f"    │ {line}")
            if "cleaned_content" in e and e["cleaned_content"]:
                print("  cleaned_content:")
                for line in str(e["cleaned_content"]).splitlines() or [""]:
                    print(f"    │ {line}")
            if e.get("tool_calls"):
                print(f"  tool_calls: {json.dumps(e['tool_calls'], indent=2)}")
            if e.get("text_parts"):
                for line in (e["text_parts"][0] or "").splitlines() or [""]:
                    print(f"  │ {line}")
            if e.get("data_parts"):
                print(f"  data_parts: {json.dumps(e['data_parts'], indent=2)}")
            if e.get("notes_state"):
                print(f"  notes_state: {e['notes_state']}")

        else:
            # Unknown event — dump verbatim
            print(f"\n[{i:03d}] {ts}  {kind}: {json.dumps({k: v for k, v in e.items() if k not in ('ts', 't', 'event')})}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
