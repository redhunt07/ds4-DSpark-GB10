#!/usr/bin/env python3
"""Black-box DS4 protocol and liveness contract checks.

Uses only the Python standard library so it can run against a canary or the
production service without installing a test framework.
"""

import argparse
import http.client
import json
import time
from urllib.parse import urlparse


def request(base, method, path, body=None, timeout=900):
    url = urlparse(base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    raw = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if raw is not None else {}
    started = time.monotonic()
    conn.request(method, path, raw, headers)
    response = conn.getresponse()
    payload = response.read().decode("utf-8", "replace")
    elapsed = time.monotonic() - started
    conn.close()
    if response.status != 200:
        raise AssertionError(f"{method} {path}: HTTP {response.status}: {payload[:500]}")
    return response.getheader("content-type", ""), payload, elapsed


def check_health(base):
    _, raw, _ = request(base, "GET", "/healthz")
    data = json.loads(raw)
    assert data["status"] == "ok" and data["model_loaded"] is True
    assert data["active"]["phase"] in {
        "queued", "prefill", "decode", "recovery", "serialize",
        "complete", "error", "cancelled",
    }


def check_chat(base, model, stream):
    _, raw, elapsed = request(base, "POST", "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "stream": stream,
        "max_tokens": 32,
        "temperature": 0,
    })
    if stream:
        assert "data: [DONE]" in raw
        assert '"finish_reason"' in raw
    else:
        data = json.loads(raw)
        assert data["choices"][0]["finish_reason"] is not None
        assert not data["choices"][0]["message"].get("tool_calls")
    return elapsed


def check_summary_no_tools(base, model):
    _, raw, elapsed = request(base, "POST", "/v1/chat/completions", {
        "model": model,
        "messages": [
            {"role": "system", "content": "You summarize coding sessions. Tools are unavailable."},
            {"role": "user", "content": "Summarize: changed parser and added tests."},
        ],
        "stream": False,
        "max_tokens": 96,
        "temperature": 0,
    })
    msg = json.loads(raw)["choices"][0]["message"]
    assert not msg.get("tool_calls"), raw[:1000]
    return elapsed


def check_identical_tool_loop_terminates(base, model):
    messages = [{"role": "user", "content": "Run the tests once and report."}]
    args = json.dumps({"command": "npm test"}, separators=(",", ":"))
    for index in range(3):
        call_id = f"loop_call_{index}"
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": args},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": "Tests passed (exit 0).",
        })
    _, raw, elapsed = request(base, "POST", "/v1/chat/completions", {
        "model": model,
        "messages": messages,
        "tools": [{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }],
        "stream": False,
        "max_tokens": 96,
        "temperature": 0,
    })
    choice = json.loads(raw)["choices"][0]
    assert choice["finish_reason"] != "tool_calls", raw[:1000]
    assert not choice["message"].get("tool_calls"), raw[:1000]
    return elapsed


def check_responses(base, model, stream):
    _, raw, elapsed = request(base, "POST", "/v1/responses", {
        "model": model,
        "input": "Reply with exactly OK.",
        "stream": stream,
        "max_output_tokens": 32,
    })
    if stream:
        assert "response.completed" in raw
    else:
        data = json.loads(raw)
        assert data["status"] == "completed"
    return elapsed


def check_anthropic(base, model, stream):
    _, raw, elapsed = request(base, "POST", "/v1/messages", {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "stream": stream,
        "max_tokens": 32,
    })
    if stream:
        assert "event: message_stop" in raw
    else:
        data = json.loads(raw)
        assert data["stop_reason"] is not None
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--quick", action="store_true",
                        help="one non-streaming request per API")
    args = parser.parse_args()

    timings = {}
    check_health(args.base_url)
    for stream in ([False] if args.quick else [False, True]):
        timings[f"chat_stream_{stream}"] = check_chat(args.base_url, args.model, stream)
        timings[f"responses_stream_{stream}"] = check_responses(args.base_url, args.model, stream)
        timings[f"anthropic_stream_{stream}"] = check_anthropic(args.base_url, args.model, stream)
    timings["summary_no_tools"] = check_summary_no_tools(args.base_url, args.model)
    timings["identical_tool_loop"] = check_identical_tool_loop_terminates(args.base_url, args.model)
    check_health(args.base_url)
    print(json.dumps({"ok": True, "timings_seconds": timings}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
