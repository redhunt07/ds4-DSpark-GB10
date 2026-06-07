"""Build a fixed-data FastMTP training corpus from the field-standard datasets
(EAGLE/SpecForge use these): ShareGPT (Aeala/ShareGPT_Vicuna_unfiltered) +
UltraChat (HuggingFaceH4/ultrachat_200k). Extracts ASSISTANT turns (the text we
want the MTP head to learn to draft) and writes them NUL-separated so multi-line
content (code, lists, paragraphs) is preserved for ds4 --mtp-harvest.

Run via uv (datasets isn't a venv dep):
  uv run --with datasets python tools/mtp/build_corpus.py --out tools/mtp/corpus_pol.txt \
      --n-sharegpt 2500 --n-ultrachat 2500
Char filter (120..2400 ~ 30..600 tok) keeps most turns inside the harvester's
16..512-token window. NUL-separated; harvester reads with getdelim('\\0').
"""

import argparse
import sys

from datasets import load_dataset  # ty: ignore[unresolved-import]


def sharegpt_turns(n, lo, hi):
    ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
    out = []
    for row in ds:
        for turn in row.get("conversations", []):
            if turn.get("from") in ("gpt", "assistant"):
                t = (turn.get("value") or "").strip()
                if lo <= len(t) <= hi:
                    out.append(t)
                    if len(out) >= n:
                        return out
    return out


def ultrachat_turns(n, lo, hi):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    out = []
    for row in ds:
        for msg in row.get("messages", []):
            if msg.get("role") == "assistant":
                t = (msg.get("content") or "").strip()
                if lo <= len(t) <= hi:
                    out.append(t)
                    if len(out) >= n:
                        return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sharegpt", type=int, default=2500)
    ap.add_argument("--n-ultrachat", type=int, default=2500)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-chars", type=int, default=2400)
    args = ap.parse_args()

    print(f"pulling ShareGPT ({args.n_sharegpt}) + UltraChat ({args.n_ultrachat}) ...")
    docs = sharegpt_turns(args.n_sharegpt, args.min_chars, args.max_chars)
    print(f"  sharegpt: {len(docs)} turns")
    uc = ultrachat_turns(args.n_ultrachat, args.min_chars, args.max_chars)
    print(f"  ultrachat: {len(uc)} turns")
    docs += uc
    # NUL-separated so internal newlines survive
    with open(args.out, "w") as f:
        for d in docs:
            f.write(d)
            f.write("\0")
    chars = sum(len(d) for d in docs)
    print(f"wrote {len(docs)} docs ({chars / 1e6:.1f}M chars) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
