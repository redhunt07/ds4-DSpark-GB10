"""Pin the leak: untrained head per-k accept, HF mask vs strict-causal mask.

HC is clean (probe_hc), so a high k1 on the UNTRAINED head -> structural forward
leak (attention). Swapping create_sliding_window_causal_mask for a hand-built
triu causal mask: if accept drops, the HF mask helper admits forward attention.
"""

import glob
import sys

import torch

sys.path.insert(0, "/home/trevor/Projects/ds4/tools/mtp")
import mtp_model as MM  # noqa: E402
import train_mtp as T  # noqa: E402

from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

SHARDS = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mtp_shards_pol"
device, dtype = "cuda", torch.bfloat16
head = MM.DeepseekV4MtpHead.from_pt(dtype=dtype).to(device).eval()
shards = sorted(glob.glob(f"{SHARDS}/shard_*.npz"))[:8]


def perk(label):
    acc = {k: [0, 0] for k in range(1, 5)}
    with torch.no_grad():
        for sp in shards:
            toks, hc = T.load_shard(sp, 256, device, dtype, head.cfg)
            if toks.shape[0] < 7:
                continue
            for k, logits, tgt, _ in T.unroll_steps(head, toks, hc, 4, device, dtype):
                acc[k][0] += int((logits.argmax(-1) == tgt).sum())
                acc[k][1] += int(tgt.shape[0])
    print(label, " ".join(f"k{k}={acc[k][0] / acc[k][1]:.3f}" for k in acc))


print("=== untrained head, current (HF sliding-window) mask ===")
perk("HF-mask  ")

# Monkeypatch rope_mask to use a hand-built strict-causal additive mask.
_orig = head.rope_mask


def strict_causal(S, position_ids, device, dtype):
    pe, _ = _orig(S, position_ids, device, dtype)
    m = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    m = torch.triu(m, diagonal=1).view(1, 1, S, S)
    return pe, m


head.rope_mask = strict_causal
print("=== untrained head, strict triu causal mask ===")
perk("triu-mask")
