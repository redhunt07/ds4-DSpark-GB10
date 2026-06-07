#!/usr/bin/env python3
"""Run a small steering-scale sweep through ds4 (in-process).

Loads the model once via libds4.so and iterates (prompt × scale), toggling
the runtime steering scale per inner iteration so the whole sweep pays the
~5–10s model load cost exactly once instead of N_prompt × N_scale times.

Output mirrors the old subprocess-per-iteration version: a banner per
prompt, a sub-banner per scale, then the greedy continuation streamed to
stdout token by token.
"""

import argparse
import sys
from pathlib import Path

# import the in-process binding from python/ (sibling of dir-steering/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import ds4  # noqa: E402


def read_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    if not prompts:
        raise SystemExit(f"{path}: no prompts found")
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ds4flash.gguf")
    ap.add_argument(
        "--direction",
        required=True,
        help="flat f32 vector file produced by build_direction.py",
    )
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--scales", default="-2,-1,-0.5,0,0.5,1,2")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--attn-scale", type=float, default=0.0)
    ap.add_argument("--nothink", action="store_true")
    args = ap.parse_args()

    prompts = read_prompts(Path(args.prompts))
    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    # Load engine once; the steering file is loaded per session via
    # reload_steering() (engine-level steering_file= is a no-op when initial
    # scales are 0 — the C side gates the actual load on a non-zero scale at
    # session create, so we load explicitly here).
    eng = ds4.Engine(args.model)
    think = ds4.THINK_NONE if args.nothink else ds4.THINK_HIGH

    try:
        for prompt in prompts:
            print("=" * 80, flush=True)
            print(f"PROMPT: {prompt}", flush=True)
            for scale in scales:
                print("-" * 80, flush=True)
                print(f"FFN scale: {scale:g}", flush=True)
                with eng.session(ctx_size=args.ctx) as s:
                    s.reload_steering(args.direction, args.attn_scale, scale)
                    s.sync(eng.encode_chat(prompt, think=think))
                    for _ in range(args.tokens):
                        if s.pos + 1 >= s.ctx:
                            break
                        tok = s.sample(temperature=0.0)  # greedy
                        if tok == eng.eos:
                            break
                        s.eval(tok)
                        sys.stdout.write(eng.token_text(tok))
                        sys.stdout.flush()
                    sys.stdout.write("\n")
                    sys.stdout.flush()
    finally:
        eng.close()


if __name__ == "__main__":
    main()
