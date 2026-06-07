"""Leak probe: which token offset does the harvested base HC predict?

hc[i] should be the CAUSAL hidden at position i -> predicts t_{i+1}. If
to_logits(hc[i]) instead best-matches t_{i+2}, the HC is shifted forward and the
MTP head can read the next-next-token target straight off the input (explains the
flat accept=1.0 / loss->0 collapse). Pure forward, no training.
"""

import glob
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/trevor/Projects/ds4/tools/mtp")
import mtp_model as MM  # noqa: E402

from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

SHARDS = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mtp_shards_pol"
device, dtype = "cuda", torch.bfloat16

head = MM.DeepseekV4MtpHead.from_pt(dtype=dtype).to(device).eval()

agree = {off: [0, 0] for off in (-1, 0, 1, 2, 3)}
with torch.no_grad():
    for sp in sorted(glob.glob(f"{SHARDS}/shard_*.npz"))[:8]:
        d = np.load(sp)
        toks = torch.from_numpy(d["tokens"].astype(np.int64)).to(device)
        N = toks.shape[0]
        hc = (
            torch.from_numpy(d["hc"][:N])
            .to(device, dtype)
            .reshape(N, head.cfg.hc_mult, head.cfg.hidden_size)
        )
        pred = head.to_logits(hc.unsqueeze(0))[0].argmax(-1)  # [N] predicted token
        for off in agree:
            # pred[i] vs tokens[i+off], valid where i+off in [0, N)
            lo = max(0, -off)
            hi = min(N, N - off)
            p = pred[lo:hi]
            t = toks[lo + off : hi + off]
            agree[off][0] += int((p == t).sum())
            agree[off][1] += int(p.shape[0])

print("offset  agreement   (pred[i] == tokens[i+offset])")
for off in (-1, 0, 1, 2, 3):
    c, n = agree[off]
    tag = ""
    if off == 1:
        tag = "  <- EXPECTED (causal next-token)"
    if off == 2:
        tag = "  <- LEAK (next-next == MTP target)"
    print(f"  {off:+d}    {c / n:6.3f}  ({c}/{n}){tag}")
