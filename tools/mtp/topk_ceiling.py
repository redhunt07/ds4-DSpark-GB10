"""Top-k acceptance ceiling: P(base argmax in head's top-k) per draft-depth + category.

Our deployed accept = P(head argmax == base argmax) (top-1, exact match). This probe
asks: if we drafted the head's TOP-K candidates (tree/multi-candidate) instead of
just top-1, how high could accept go? = the headroom that tree-drafting OR better
training (curriculum, multi-regime) could capture.

  top-1 ~= top-4  -> head dist already as peaked as the base allows -> WALL is
                     fundamental (base predictability); no training method helps.
  top-1 <<  top-4  -> real headroom -> tree-drafting + sharper training justified.

  uv run --project tools/mtp python tools/mtp/topk_ceiling.py \
      --shards /tmp/mtp_shards_v2 --trained tools/mtp/mtp_v2.pt --per-class 12
"""

import argparse

import mtp_model as MM
import torch
from train_mtp import list_shards, load_shard, unroll_steps

KS = [1, 2, 4, 8, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--trained", default="tools/mtp/mtp_v2.pt")
    ap.add_argument("--per-class", type=int, default=12, help="eval shards/category")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--max_seq", type=int, default=320)
    args = ap.parse_args()
    dtype = torch.bfloat16

    head = MM.DeepseekV4MtpHead.from_pt(dtype=dtype).to("cuda")
    blob = torch.load(args.trained, map_location="cuda")
    head.load_state_dict(blob["trainable"], strict=False)
    head.eval()

    shards, classes, _ = list_shards(args.shards)
    by_cls = {}
    for s in shards:
        by_cls.setdefault(classes.get(s, "all"), []).append(s)
    eval_shards = []
    for cls, sl in by_cls.items():
        eval_shards += sl[-args.per_class :]  # held-out tail per class
    print(f"top-k ceiling: {len(eval_shards)} shards, K={args.K}\n")

    # per (depth k, class) -> [hits_at_K for K in KS], total
    agg = {}
    with torch.no_grad():
        for sp in eval_shards:
            toks, hc = load_shard(sp, args.max_seq, "cuda", dtype, head.cfg)
            if toks.shape[0] < args.K + 3:
                continue
            cls = classes.get(sp, "all")
            for k, logits, tgt, _ in unroll_steps(head, toks, hc, args.K, "cuda", dtype):
                # rank of target = #logits strictly greater than the target's logit
                tgt_logit = logits.gather(1, tgt.unsqueeze(1))  # [L,1]
                ranks = (logits > tgt_logit).sum(dim=1)  # [L], 0 = top-1
                a = agg.setdefault((k, cls), [[0] * len(KS), 0])
                for i, K in enumerate(KS):
                    a[0][i] += int((ranks < K).sum())
                a[1] += int(tgt.shape[0])

    # report per depth: aggregate + per-class
    classes_seen = sorted({c for (_, c) in agg})
    for depth in range(1, args.K + 1):
        print(f"=== draft-{depth} (k{depth}): P(base argmax in head top-K) ===")
        hdr = "  " + f"{'category':20s}" + "".join(f"{'top-'+str(K):>8s}" for K in KS)
        print(hdr)
        # aggregate row
        tot = [0] * len(KS)
        totn = 0
        for cls in classes_seen:
            if (depth, cls) in agg:
                h, n = agg[(depth, cls)]
                for i in range(len(KS)):
                    tot[i] += h[i]
                totn += n
        print("  " + f"{'AGGREGATE':20s}" + "".join(f"{tot[i] / max(1, totn) * 100:7.1f}%" for i in range(len(KS))))
        for cls in classes_seen:
            if (depth, cls) in agg:
                h, n = agg[(depth, cls)]
                print("  " + f"{cls:20s}" + "".join(f"{h[i] / max(1, n) * 100:7.1f}%" for i in range(len(KS))))
        print()


if __name__ == "__main__":
    main()
