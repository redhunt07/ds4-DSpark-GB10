#!/usr/bin/env python3
"""De-risk stage 1: validate a torch-free numpy reimpl of the MTP head's INPUT
PROJECTION against ds4's own output, using the DS4_MTP_DUMP records (MTP2).

ds4's projection (metal_graph_eval_mtp_draft_from_hc):
    e   = e_proj @ rmsnorm(embed(token), enorm)        # [N_EMBD]
    e_hc= repeat(e, N_HC)                                # [N_HC, N_EMBD]
    h[r]= h_proj @ rmsnorm(prev_hc[r], hnorm)  for each hc row r
    input_hc = e_hc + h                                  # [N_HC, N_EMBD]

We recompute input_hc from each record's (token, prev_hc) using the gguf weights
ds4 actually runs (e_proj/h_proj Q8_0, enorm/hnorm F32, token_embd F16) and
compare to ds4's dumped input_hc. Match => the head-weight loading + projection
plumbing of a reimpl is faithful (stage 1 of the FastMTP de-risk). No torch.

Usage: validate_projection.py <dump.bin> [--base ds4flash.gguf] [--mtp MTP.gguf]
"""

import argparse
import struct
import sys
import numpy as np

FORK = "/home/trevor/Projects/llama.cpp-tjs-fork/gguf-py"
sys.path.insert(0, FORK)
from gguf import GGUFReader, GGMLQuantizationType, quants  # noqa: E402

MAGIC = 0x4D545032  # 'MTP2'


def deq(reader, name):
    for t in reader.tensors:
        if t.name == name:
            arr = quants.dequantize(t.data, GGMLQuantizationType(t.tensor_type))
            return np.asarray(arr, dtype=np.float32), tuple(int(x) for x in t.shape)
    raise KeyError(name)


def kv_float(reader, *names, default):
    for n in names:
        f = reader.fields.get(n)
        if f is not None:
            try:
                return float(f.contents())
            except Exception:
                pass
    return default


def rmsnorm(x, w, eps):
    # x: [..., D], w: [D]
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)).astype(np.float32) * w


def rel_rms(a, b):
    num = np.sqrt(np.mean((a - b) ** 2))
    den = np.sqrt(np.mean(b**2)) + 1e-12
    return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--base", default="ds4flash.gguf")
    ap.add_argument(
        "--mtp",
        default="/home/trevor/models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf",
    )
    ap.add_argument("--max-records", type=int, default=8)
    args = ap.parse_args()

    base = GGUFReader(args.base)
    mtp = GGUFReader(args.mtp)
    eps = kv_float(
        base,
        "deepseek4.attention.layer_norm_rms_epsilon",
        "deepseek2.attention.layer_norm_rms_epsilon",
        default=1e-6,
    )
    print(f"rms_eps = {eps}")

    embed, _ = deq(base, "token_embd.weight")  # [N_VOCAB, N_EMBD] flat
    enorm, _ = deq(mtp, "mtp.0.enorm.weight")
    hnorm, _ = deq(mtp, "mtp.0.hnorm.weight")
    eproj, eproj_shape = deq(mtp, "mtp.0.e_proj.weight")
    hproj, hproj_shape = deq(mtp, "mtp.0.h_proj.weight")
    n_embd = enorm.shape[0]
    embed = embed.reshape(-1, n_embd)  # [N_VOCAB, N_EMBD]
    print(
        f"shapes: embed={embed.shape} e_proj={eproj_shape} h_proj={hproj_shape} N_EMBD={n_embd}"
    )

    # e_proj/h_proj are square [N_EMBD,N_EMBD]; orientation (W@x vs W.T@x) is
    # ambiguous from shape, so try both and report which matches ds4.
    eW = eproj.reshape(n_embd, n_embd)
    hW = hproj.reshape(n_embd, n_embd)

    records = []
    with open(args.dump, "rb") as f:
        data = f.read()
    off = 0
    while off < len(data) and len(records) < args.max_records:
        magic, pos, tok, arg, hcd = struct.unpack_from("<IIiiI", data, off)
        off += 20
        if magic != MAGIC:
            print(f"bad magic {magic:#x} (expected MTP2 {MAGIC:#x}) — rebuild+redump")
            return 2
        prev = np.frombuffer(data, np.float32, hcd, off).copy()
        off += hcd * 4
        inp = np.frombuffer(data, np.float32, hcd, off).copy()
        off += hcd * 4
        records.append((pos, tok, arg, hcd, prev, inp))

    n_hc = records[0][3] // n_embd
    print(f"records={len(records)} hc_dim={records[0][3]} -> N_HC={n_hc}\n")

    for orient in ("W@x", "W.T@x"):
        eM = eW if orient == "W@x" else eW.T
        hM = hW if orient == "W@x" else hW.T
        errs = []
        for pos, tok, arg, hcd, prev, inp in records:
            e = eM @ rmsnorm(embed[tok], enorm, eps)  # [N_EMBD]
            prev_hc = prev.reshape(n_hc, n_embd)
            h = (hM @ rmsnorm(prev_hc, hnorm, eps).T).T  # [N_HC, N_EMBD]
            mine = (e[None, :] + h).reshape(-1)
            errs.append(rel_rms(mine, inp))
        print(
            f"orientation {orient:7s}: per-record rel_rms = "
            + " ".join(f"{e:.2e}" for e in errs)
        )

    print(
        "\nstage-1 PASS if one orientation gives rel_rms ~1e-3 or below across records."
    )


if __name__ == "__main__":
    sys.exit(main())
