#!/usr/bin/env python3
"""Stream a ds4 chat completion with reasoning, separating THINKING from ANSWER.
Run unbuffered (python3 -u) so the output file updates live for `tail -f`."""
import json, sys, time, urllib.request

URL = "http://localhost:8000/v1/chat/completions"
PROBLEM = (
    "There are 2025 lamps in a row, numbered 1 to 2025, all initially OFF. You make "
    "2025 passes. On pass k (for k = 1, 2, ..., 2025), you toggle (flip on<->off) "
    "every lamp whose number is divisible by k OR divisible by (2026 - k). After all "
    "2025 passes, exactly how many lamps are ON, and which lamp number(s)? Prove your "
    "answer rigorously."
)

body = {
    "model": "deepseek-v4-flash",
    "reasoning_effort": "max",
    "temperature": 0.6,
    "max_tokens": 48000,
    "stream": True,
    "messages": [{"role": "user", "content": PROBLEM}],
}

req = urllib.request.Request(
    URL, data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)

print("=" * 78)
print("PROBLEM:", PROBLEM)
print("model=deepseek-v4-flash  reasoning_effort=max  temp=0.6  ctx=1048576")
print("=" * 78, flush=True)

t0 = time.time()
mode = None            # "think" or "answer"
n_think = n_ans = 0
first_tok = None

def switch(new):
    global mode
    if mode != new:
        label = "\n\n################  THINKING  ################\n" if new == "think" \
            else "\n\n################  ANSWER  ################\n"
        sys.stdout.write(label)
        mode = new

with urllib.request.urlopen(req, timeout=900) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = obj.get("choices", [{}])[0].get("delta", {})
        think = delta.get("reasoning_content") or delta.get("reasoning")
        ans = delta.get("content")
        if think:
            if first_tok is None:
                first_tok = time.time() - t0
            switch("think"); sys.stdout.write(think); n_think += len(think)
        if ans:
            if first_tok is None:
                first_tok = time.time() - t0
            switch("answer"); sys.stdout.write(ans); n_ans += len(ans)
        sys.stdout.flush()

dt = time.time() - t0
print("\n\n" + "=" * 78)
print(f"done in {dt:.1f}s  ttft={first_tok:.2f}s  "
      f"reasoning_chars={n_think}  answer_chars={n_ans}", flush=True)
