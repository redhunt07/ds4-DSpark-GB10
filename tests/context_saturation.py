#!/usr/bin/env python3
"""Exercise warm-prefix requests near the advertised context limit."""

import argparse
import json
import time
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def post_json(url, payload, timeout=1800):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", "replace")
    return raw, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--fractions", default="0.25,0.75,0.90,0.98")
    args = parser.parse_args()

    models = get_json(args.base_url + "/v1/models")
    context = models["data"][0]["context_length"]
    results = []
    # " context" is one token with the DS4 tokenizer. Keep 512 tokens for
    # rendered role markers and the short completion.
    prefix = ""
    words = 0
    for fraction in [float(item) for item in args.fractions.split(",")]:
        target = max(1, int(context * fraction) - 512)
        if target > words:
            prefix += " context" * (target - words)
            words = target
        raw, elapsed = post_json(args.base_url + "/v1/chat/completions", {
            "model": args.model,
            "messages": [{
                "role": "user",
                "content": prefix + "\nReply with OK and nothing else.",
            }],
            "stream": True,
            "max_tokens": 8,
            "temperature": 0,
        })
        assert "data: [DONE]" in raw and '"finish_reason"' in raw
        health = get_json(args.base_url + "/healthz")
        assert health["active"]["terminal"] is True
        results.append({
            "fraction": fraction,
            "target_tokens": target,
            "elapsed_seconds": elapsed,
            "terminal_phase": health["active"]["phase"],
        })
    print(json.dumps({"ok": True, "context": context, "results": results}, indent=2))


if __name__ == "__main__":
    main()
