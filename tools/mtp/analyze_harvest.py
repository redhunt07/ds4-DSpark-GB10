"""CPU-only, read-only health check on packed harvest shards (run mid-harvest).

Validates the self-distill data before we commit a train to it: per-category gen
length + alignment, dist sharpness (entropy -> accept ceiling), and a ZH-coherence
spot check (decode zh-* generations back to text). Touches only shard_*.npz tokens/
p_idx/p_val (NOT the big hc array, NOT the _raw_ dirs the harvest is writing).

  uv run --project tools/mtp python tools/mtp/analyze_harvest.py --shards DIR \
      --classes tools/mtp/prompts_v2.txt.classes.json
"""

import argparse
import glob
import json
import os

import numpy as np

SNAP = ("/home/trevor/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/"
        "snapshots/6c858e71890b508e4f3fd6491f45b325580ba934")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--zh-samples", type=int, default=3, help="zh gens to decode/cat")
    args = ap.parse_args()

    classes = json.load(open(args.classes))["classes"]
    shards = sorted(glob.glob(os.path.join(args.shards, "shard_*.npz")))
    print(f"analyzing {len(shards)} packed shards\n")

    # per-category aggregates
    agg = {}  # cls -> [n_shards, total_pos, align_ok, align_tot, ent_sum, ent_n]
    zh_samples = {}  # cls -> list of (shard, tokens) to decode
    for sp in shards:
        idx = int(os.path.basename(sp)[len("shard_") : -len(".npz")])
        cls = classes[idx] if idx < len(classes) else "?"
        d = np.load(sp)
        tok = d["tokens"]
        pidx, pval = d["p_idx"], d["p_val"]
        a = agg.setdefault(cls, [0, 0, 0, 0, 0.0, 0])
        a[0] += 1
        a[1] += len(tok)
        a[2] += int((pidx[:, 0] == tok).sum())
        a[3] += len(tok)
        # entropy of the top-64 dist per position (p_val sums ~1); mean over positions
        p = np.clip(pval.astype(np.float64), 1e-12, 1.0)
        ent = -(p * np.log(p)).sum(axis=1)  # nats, per position
        a[4] += float(ent.sum())
        a[5] += len(ent)
        if cls.startswith("zh-") and len(zh_samples.get(cls, [])) < args.zh_samples:
            zh_samples.setdefault(cls, []).append((os.path.basename(sp), tok.copy()))

    # report table
    print(f"{'category':22s} {'shards':>6s} {'pos':>8s} {'pos/sh':>6s} "
          f"{'align%':>7s} {'entropy':>7s}")
    eval_cats = {"analytical-qa", "chat-essay", "code-generation",
                 "structured-list", "prose-continuation"}
    for cls in sorted(agg, key=lambda c: (not c.startswith("zh-"), c)):
        n, pos, aok, atot, esum, en = agg[cls]
        tag = " [EVAL]" if cls in eval_cats else (" [ZH]" if cls.startswith("zh-") else "")
        print(f"{cls:22s} {n:6d} {pos:8d} {pos / max(1, n):6.0f} "
              f"{aok / max(1, atot) * 100:6.1f}% {esum / max(1, en):7.3f}{tag}")
    tot_align = sum(a[2] for a in agg.values()) / max(1, sum(a[3] for a in agg.values()))
    print(f"\noverall alignment p_idx[:,0]==token: {tot_align * 100:.2f}% "
          f"(should be ~100%); entropy in nats (higher = flatter = harder to draft)")

    # ZH coherence: decode generations back to text
    if zh_samples:
        print("\n=== ZH coherence (decoded generations) ===")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
        assert tok is not None
        for cls in sorted(zh_samples):
            print(f"\n-- {cls} --")
            for name, ids in zh_samples[cls]:
                text = tok.decode(ids.tolist(), skip_special_tokens=True)
                if not isinstance(text, str):
                    text = " ".join(text)
                snippet = text[:240].replace("\n", " ")
                print(f"  [{name}] {snippet}")


if __name__ == "__main__":
    main()
