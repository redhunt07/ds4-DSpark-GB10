#!/usr/bin/env python3
"""GB10 DS4 streaming inference diagnostic.

Measures:
  - streaming TTFT from the first SSE content chunk
  - wall-clock output throughput
  - prompt prefill / decode timing from ds4-server logs
  - basic host/GPU/KV-cache state snapshots

The script is intentionally practical rather than fancy: it uses only the
Python standard library plus curl/journalctl/systemctl/nvidia-smi if present.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_SERVICE = "ds4-server-dspark.service"
DEFAULT_PROMPT = "Rispondi con 128 parole ciao separate da spazi e senza punteggiatura."


def _run(cmd: list[str], check: bool = False) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc.returncode, proc.stdout, proc.stderr


def _which(name: str) -> bool:
    return shutil.which(name) is not None


def _parse_sse_chunk(data: str) -> dict | None:
    if data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _chunk_text(chunk: dict) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0] or {}
    delta = choice.get("delta") or {}
    if isinstance(delta, dict):
        text = delta.get("content")
        if text:
            return text
        text = delta.get("reasoning_content")
        if text:
            return text
    text = choice.get("text")
    if text:
        return text
    message = choice.get("message") or {}
    if isinstance(message, dict):
        text = message.get("content")
        if text:
            return text
    return ""


@dataclass
class StreamResult:
    response_json: dict | None
    content: str
    elapsed_s: float
    ttft_s: float | None
    usage: dict | None
    stderr: str


def stream_chat(url: str, payload: dict, timeout_s: int) -> StreamResult:
    cmd = [
        "curl", "-sS", "-N", "--http1.1",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload, ensure_ascii=False),
        url,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    start = time.perf_counter()
    first_chunk_s = None
    content_parts: list[str] = []
    usage = None
    response_json = None
    try:
        assert proc.stdout is not None
        while True:
            if timeout_s and (time.perf_counter() - start) > timeout_s:
                proc.kill()
                raise TimeoutError(f"stream timed out after {timeout_s}s")
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                break
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            chunk = _parse_sse_chunk(line[6:].strip())
            if chunk is None:
                if line[6:].strip() == "[DONE]":
                    break
                continue
            response_json = chunk
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            text = _chunk_text(chunk)
            if text:
                content_parts.append(text)
                if first_chunk_s is None:
                    first_chunk_s = time.perf_counter() - start
    finally:
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        stderr = proc.stderr.read() if proc.stderr else ""
    return StreamResult(
        response_json=response_json,
        content="".join(content_parts),
        elapsed_s=time.perf_counter() - start,
        ttft_s=first_chunk_s,
        usage=usage,
        stderr=stderr,
    )


def nonstream_chat(url: str, payload: dict) -> dict:
    cmd = [
        "curl", "-sS", "--http1.1",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload, ensure_ascii=False),
        url,
    ]
    _, stdout, stderr = _run(cmd, check=True)
    if stderr.strip():
        pass
    return json.loads(stdout)


def journal_since(unit: str, since_epoch: float) -> str:
    if not _which("journalctl"):
        return ""
    _, stdout, _ = _run([
        "journalctl", "--user", "-u", unit, "--no-pager",
        "--since", f"@{int(since_epoch)}",
    ])
    return stdout


def service_status(unit: str) -> str:
    if not _which("systemctl"):
        return ""
    _, stdout, _ = _run(["systemctl", "--user", "status", unit, "--no-pager", "-l"])
    return stdout


def snapshot() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, cmd in {
        "service": ["systemctl", "--user", "status", DEFAULT_SERVICE, "--no-pager", "-l"],
        "env": ["systemctl", "--user", "show", DEFAULT_SERVICE, "-p", "Environment", "--no-pager"],
        "gpu": ["nvidia-smi", "--query-gpu=name,utilization.gpu,power.draw,temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        "free": ["free", "-h"],
        "vmstat": ["vmstat", "1", "2"],
        "df": ["df", "-h", "/home/redhunt07/ai-stack/ds4/kv-cache"],
    }.items():
        if _which(cmd[0]):
            _, stdout, stderr = _run(cmd)
            out[key] = stdout.strip() or stderr.strip()
    return out


def parse_logs(log_text: str) -> dict[str, object]:
    prompt_done = [float(m.group(1)) for m in re.finditer(r"prompt done ([0-9.]+)s", log_text)]
    finish = [(int(m.group(1)), float(m.group(2)))
              for m in re.finditer(r"gen=(\d+).*?finish=\w+ ([0-9.]+)s", log_text)]
    decode_chunks = [float(m.group(1)) for m in re.finditer(r"decoding chunk=([0-9.]+) t/s avg=([0-9.]+) t/s", log_text)]
    spec_gates = len(re.findall(r"spec gate ", log_text))
    mtp_combined = len(re.findall(r"mtp timing combined", log_text))
    mtp_seq = len(re.findall(r"mtp timing seq", log_text))
    return {
        "prompt_done_s": prompt_done[-1] if prompt_done else None,
        "finish_tokens": finish[-1][0] if finish else None,
        "finish_s": finish[-1][1] if finish else None,
        "decode_chunk_avg_tps": decode_chunks[-1] if decode_chunks else None,
        "spec_gate_lines": spec_gates,
        "mtp_combined_lines": mtp_combined,
        "mtp_seq_lines": mtp_seq,
    }


def human_num(v: float | None, digits: int = 2) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="GB10 DS4 streaming TTFT / throughput diagnostic")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--warmup", action="store_true", help="run one non-stream warmup before the measured request")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": True,
        "thinking": {"type": args.thinking},
    }

    pre = snapshot()
    service_before = service_status(args.service) if _which("systemctl") else ""

    warm_usage = None
    if args.warmup:
        warm_payload = dict(payload)
        warm_payload["stream"] = False
        warm_result = nonstream_chat(args.url, warm_payload)
        warm_usage = warm_result.get("usage")

    start_epoch = time.time()
    result = stream_chat(args.url, payload, args.timeout)
    log_text = journal_since(args.service, start_epoch)
    parsed = parse_logs(log_text)

    prompt_tokens = None
    if isinstance(warm_usage, dict):
        prompt_tokens = warm_usage.get("prompt_tokens")
    elif isinstance(result.response_json, dict):
        usage = result.response_json.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = parsed.get("finish_tokens")
    if completion_tokens is None:
        completion_tokens = args.max_tokens

    ttft_s = result.ttft_s
    prompt_done_s = parsed.get("prompt_done_s")
    finish_s = parsed.get("finish_s")
    decode_s = (finish_s - prompt_done_s) if isinstance(prompt_done_s, (int, float)) and isinstance(finish_s, (int, float)) else None
    prompt_tps = (prompt_tokens / prompt_done_s) if isinstance(prompt_tokens, (int, float)) and isinstance(prompt_done_s, (int, float)) and prompt_done_s > 0 else None
    decode_tps = (completion_tokens / decode_s) if isinstance(completion_tokens, (int, float)) and isinstance(decode_s, (int, float)) and decode_s > 0 else None
    wall_tps = (completion_tokens / result.elapsed_s) if result.elapsed_s > 0 else None

    post = snapshot()
    service_after = service_status(args.service) if _which("systemctl") else ""

    report = {
        "service": args.service,
        "url": args.url,
        "model": args.model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft_s,
        "prompt_done_s": prompt_done_s,
        "finish_s": finish_s,
        "prompt_tps": prompt_tps,
        "decode_tps": decode_tps,
        "wall_tps": wall_tps,
        "measure_elapsed_s": result.elapsed_s,
        "parsed": parsed,
        "snapshot_before": pre,
        "snapshot_after": post,
        "service_before": service_before,
        "service_after": service_after,
        "stream_content": result.content,
        "logs": log_text,
    }

    print("# GB10 DS4 streaming diagnostic")
    print(f"- service: `{args.service}`")
    print(f"- url: `{args.url}`")
    print(f"- model: `{args.model}`")
    print(f"- prompt tokens: {prompt_tokens if prompt_tokens is not None else '-'}")
    print(f"- completion tokens: {completion_tokens}")
    print(f"- TTFT: {human_num(ttft_s)} s")
    print(f"- prompt done: {human_num(prompt_done_s)} s")
    print(f"- finish: {human_num(finish_s)} s")
    print(f"- prompt t/s: {human_num(prompt_tps)}")
    print(f"- decode t/s: {human_num(decode_tps)}")
    print(f"- wall t/s: {human_num(wall_tps)}")
    print(f"- spec gate lines: {parsed['spec_gate_lines']}")
    print(f"- DSpark timing lines: combined={parsed['mtp_combined_lines']} seq={parsed['mtp_seq_lines']}")
    print()
    print("## Snapshot")
    for key in ("gpu", "free", "vmstat", "df"):
        if key in pre:
            print(f"### {key}")
            print("```")
            print(pre[key])
            print("```")
    print("## Service")
    print("```")
    print(service_after or service_before or "(service status unavailable)")
    print("```")
    print("## DS4 logs")
    print("```")
    tail = "\n".join(log_text.strip().splitlines()[-40:])
    print(tail or "(no matching logs)")
    print("```")

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Basic, conservative interpretation for the user.
    print("## Readout")
    if isinstance(decode_tps, (int, float)) and decode_tps < 5:
        print("- output decode is still low; the bottleneck is likely draft acceptance, long-context KV, or a fallback path.")
    if parsed["spec_gate_lines"] == 0:
        print("- DSpark gate was not observed; the server may still be on a non-spec path.")
    elif parsed["mtp_combined_lines"] == 0 and parsed["mtp_seq_lines"] > 0:
        print("- DSpark is running, but only the sequential speculative path was observed; combined fast path is still not paying off.")
    elif parsed["mtp_combined_lines"] > 0:
        print("- combined DSpark timing was observed; this is the path most likely to deliver the promised uplift.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
