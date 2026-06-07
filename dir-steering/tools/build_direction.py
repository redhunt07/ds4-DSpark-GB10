#!/usr/bin/env python3
"""Build a DS4 directional-steering vector from paired prompt sets.

Thin CLI on top of ``ds4_steering.build_direction``: loads the model once
via ``libds4.so``, captures per-layer activations for each (good, bad)
prompt pair, averages and orthogonalizes, then writes the flat f32 vector
file the runtime loads via ``--dir-steering-file``.
"""

import argparse
import sys
from pathlib import Path

# import from python/ (sibling of dir-steering/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import ds4  # noqa: E402
import ds4_steering as steering  # noqa: E402


def read_prompt_file(path: Path) -> list[str]:
    """Read one prompt per non-empty line, ignoring shell-style comments."""
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    if not prompts:
        raise SystemExit(f"{path}: no prompts found")
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ds4flash.gguf", help="GGUF model path")
    ap.add_argument(
        "--good-file", required=True, help="desired/target prompts, one per line"
    )
    ap.add_argument(
        "--bad-file", required=True, help="contrast/control prompts, one per line"
    )
    ap.add_argument(
        "--out",
        default="dir-steering/out/direction.json",
        help="metadata JSON path; .f32 is written next to it",
    )
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument(
        "--component",
        default="ffn_out",
        choices=("ffn_out", "attn_out"),
        help="runtime-editable 4096-wide activation stream",
    )
    ap.add_argument(
        "--think",
        action="store_true",
        help="capture after <think>; default captures direct answers",
    )
    ap.add_argument(
        "--pair-normalize",
        action="store_true",
        help="average normalized per-pair differences",
    )
    ap.add_argument(
        "--no-orthogonalize",
        action="store_true",
        help="do not remove the component parallel to the control mean",
    )
    args = ap.parse_args()

    model = Path(args.model).resolve()
    good = read_prompt_file(Path(args.good_file))
    bad = read_prompt_file(Path(args.bad_file))

    eng = ds4.Engine(str(model))
    try:
        directions = steering.build_direction(
            eng,
            good,
            bad,
            component=args.component,
            system=args.system,
            think=args.think,
            ctx=args.ctx,
            pair_normalize=args.pair_normalize,
            orthogonalize=not args.no_orthogonalize,
        )
    finally:
        eng.close()

    json_path, f32_path = steering.save_direction(
        directions,
        args.out,
        meta={
            "component": args.component,
            "thinking": bool(args.think),
            "pair_normalize": bool(args.pair_normalize),
            "orthogonalize_control_mean": not args.no_orthogonalize,
            "good_file": str(Path(args.good_file)),
            "bad_file": str(Path(args.bad_file)),
            "model": str(model),
            "note": "runtime positive scale suppresses this direction; negative scale amplifies it",
        },
    )
    print(f"wrote {json_path}")
    print(f"wrote {f32_path}")


if __name__ == "__main__":
    main()
