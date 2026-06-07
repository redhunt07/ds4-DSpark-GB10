"""Stage-2 de-risk: run the assembled V4-Flash nextn head FORWARD and check it
against ds4. Training-mode forward (full self-attention over a sequence, no
incremental cache), feeding ds4's dumped per-position prev_hc as conditioning.

Exact bit-match to ds4 is not the goal (ds4 uses an incremental Q8 KV cache; this
is a bf16 full-seq forward) — we check that (a) the forward runs end-to-end and
produces finite logits, and (b) the head's top-1 predictions agree with ds4's
draft argmax / the actual next token at a rate consistent with ds4's measured
accept. That validates the TRAINING forward path FastMTP needs.

Usage: forward_check.py [dump.bin]
"""

import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/trevor/Projects/ds4/tools/mtp")
sys.path.insert(0, "/home/trevor/Projects/llama.cpp-tjs-fork/gguf-py")
from gguf import GGUFReader, GGMLQuantizationType, quants  # noqa: E402
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M  # noqa: E402
from transformers.masking_utils import create_sliding_window_causal_mask  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
import mtp_head as H  # noqa: E402

BASE = "/home/trevor/Projects/ds4/ds4flash.gguf"


def _deq(reader, name):
    for t in reader.tensors:
        if t.name == name:
            return torch.from_numpy(
                np.ascontiguousarray(
                    quants.dequantize(t.data, GGMLQuantizationType(t.tensor_type)),
                    np.float32,
                )
            )
    raise KeyError(name)


def rmsnorm(x, w, eps):
    return (x / torch.sqrt(x.double().pow(2).mean(-1, keepdim=True) + eps)).float() * w


def load_records(path, n):
    data = open(path, "rb").read()
    off, recs = 0, []
    while off < len(data) and len(recs) < n:
        magic, pos, tok, arg, hcd = struct.unpack_from("<IIiiI", data, off)
        off += 20
        assert magic == 0x4D545032
        prev = torch.from_numpy(np.frombuffer(data, np.float32, hcd, off).copy())
        off += hcd * 4
        off += hcd * 4  # skip input_hc
        recs.append((pos, tok, arg, hcd, prev))
    return recs


def main():
    dump = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mtp_dump2.bin"
    dt = torch.bfloat16
    layer, glue, cfg, miss, unexp = H.load_head(dt)
    assert not miss and not unexp
    layer.eval()
    n_embd, hc = cfg.hidden_size, cfg.hc_mult

    hc_head = M.DeepseekV4HyperHead(cfg).to(dt).eval()
    hc_head.load_state_dict(
        {
            "hc_fn": glue["hc_head_fn"],
            "hc_base": glue["hc_head_base"],
            "hc_scale": glue["hc_head_scale"],
        },
        strict=False,
    )
    rotary = M.DeepseekV4RotaryEmbedding(cfg)
    e_proj = glue["e_proj.weight"].float()
    h_proj = glue["h_proj.weight"].float()
    enorm = glue["enorm.weight"].float()
    hnorm = glue["hnorm.weight"].float()
    norm_w = glue["norm.weight"].float()
    base = GGUFReader(BASE)
    embed = _deq(base, "token_embd.weight").reshape(-1, n_embd)
    out_w = _deq(base, "output.weight").reshape(-1, n_embd)  # [vocab, n_embd]
    eps = cfg.rms_norm_eps

    recs = load_records(dump, 12)
    toks = [r[2] for r in recs]  # ds4 draft argmax per record (its prediction)
    # build input_hc sequence from (token, prev_hc)
    rows = []
    for _, tok, _, _hcd, prev in recs:
        e = e_proj @ rmsnorm(embed[tok], enorm, eps)
        ph = prev.reshape(hc, n_embd)
        h = (h_proj @ rmsnorm(ph, hnorm, eps).T).T
        rows.append((e.unsqueeze(0) + h))  # [hc, n_embd]
    S = len(rows)
    x = torch.stack(rows).unsqueeze(0).to(dt)  # [1, S, hc, n_embd]
    ids = torch.tensor([[r[1] for r in recs]])
    pos_ids = torch.arange(S).unsqueeze(0)
    ref = torch.zeros(1, S, n_embd, dtype=dt)
    pe = {
        "main": rotary(ref, position_ids=pos_ids, layer_type="main"),
        "compress": rotary(ref, position_ids=pos_ids, layer_type="compress"),
    }
    mask = create_sliding_window_causal_mask(
        config=cfg,
        inputs_embeds=ref,
        attention_mask=None,
        past_key_values=DynamicCache(config=cfg),
        position_ids=pos_ids,
    )
    with torch.no_grad():
        out = layer(
            x,
            input_ids=ids,
            position_ids=pos_ids,
            position_embeddings=pe,
            attention_mask=mask,
        )
        collapsed = hc_head(out)  # [1, S, n_embd]
        h = rmsnorm(collapsed.float(), norm_w, eps)
        logits = h @ out_w.T  # [1, S, vocab]
        mine = logits.argmax(-1)[0].tolist()
    finite = bool(torch.isfinite(logits).all())
    agree_ds4 = sum(int(m == d) for m, d in zip(mine, toks))
    # next-token accept-style: ds4's accepted next = the next record's token
    nxt = [r[1] for r in recs][1:]
    agree_next = sum(int(mine[i] == nxt[i]) for i in range(len(nxt)))
    print(f"forward ran: finite={finite}  S={S}")
    print(f"  my argmax     : {mine}")
    print(f"  ds4 draft arg : {toks}")
    print(f"  top-1 vs ds4-draft : {agree_ds4}/{S}")
    print(f"  top-1 vs next-token: {agree_next}/{len(nxt)}")
    print("STAGE-2 forward:", "RUNS (finite logits)" if finite else "FAIL (nonfinite)")


if __name__ == "__main__":
    main()
