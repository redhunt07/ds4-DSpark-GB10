#!/usr/bin/env python3
"""Run repeated live tool-call/result cycles against OpenAI Chat."""

import argparse
import json

from protocol_contract import request


TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Return the supplied integer unchanged.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--cycles", type=int, default=50)
    args = parser.parse_args()

    elapsed = 0.0
    for cycle in range(1, args.cycles + 1):
        messages = [{
            "role": "user",
            "content": f"Call ping exactly once with value {cycle}; do not answer first.",
        }]
        _, raw, took = request(args.base_url, "POST", "/v1/chat/completions", {
            "model": args.model,
            "messages": messages,
            "tools": [TOOL],
            "tool_choice": {"type": "function", "function": {"name": "ping"}},
            "stream": cycle % 2 == 0,
            "max_tokens": 128,
            "temperature": 0,
        })
        elapsed += took

        if cycle % 2 == 0:
            events = []
            for block in raw.split("\n\n"):
                for line in block.splitlines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        events.append(json.loads(line[6:]))
            fragments = []
            call_id = None
            for event in events:
                delta = event.get("choices", [{}])[0].get("delta", {})
                for call in delta.get("tool_calls", []):
                    call_id = call.get("id") or call_id
                    fragments.append(call.get("function", {}).get("arguments", ""))
            assert call_id and fragments, raw[:1000]
            call = {
                "id": call_id,
                "type": "function",
                "function": {"name": "ping", "arguments": "".join(fragments)},
            }
        else:
            choice = json.loads(raw)["choices"][0]
            assert choice["finish_reason"] == "tool_calls", raw[:1000]
            calls = choice["message"].get("tool_calls", [])
            assert len(calls) == 1, raw[:1000]
            call = calls[0]

        parsed = json.loads(call["function"]["arguments"])
        assert parsed["value"] == cycle, (cycle, parsed)
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"],
             "content": json.dumps({"value": cycle})},
        ])
        _, raw, took = request(args.base_url, "POST", "/v1/chat/completions", {
            "model": args.model,
            "messages": messages,
            "stream": False,
            "max_tokens": 64,
            "temperature": 0,
        })
        elapsed += took
        terminal = json.loads(raw)["choices"][0]
        assert terminal["finish_reason"] != "tool_calls", raw[:1000]
        assert not terminal["message"].get("tool_calls"), raw[:1000]
        if cycle % 10 == 0:
            print(f"completed {cycle}/{args.cycles}", flush=True)

    print(json.dumps({"ok": True, "cycles": args.cycles,
                      "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
