"""Harvest FastMTP training data: drive `ds4 --mtp-harvest` ONCE (model loaded a
single time) over a corpus, then pack the per-doc (base_HC, token) shards to npz.

ds4 --mtp-harvest writes, per doc (one corpus line, raw-tokenized, fresh session
-> clean per-doc HCs): <out>/shard_NNNNN.mhcd (MHCD HC records) + .tok (token ids),
guaranteed 1:1 aligned (same prefill produces both). This packer converts each to
shard_NNNNN.npz {tokens int32[N], hc float32[N, hc_dim]} and removes the raw files.

The trainer forms FastMTP windows (h_i, t_{i+1..i+K}) -> targets t_{i+2..i+K+1},
skipping i=0 (BOS) and the last K positions.

Corpus: one document per line. HC stored fp32 (BOS/massive-activation channels
overflow fp16). Usage: harvest.py --corpus FILE --out DIR [--model ds4flash.gguf].
"""

import argparse
import glob
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = "/home/trevor/Projects/ds4"
DS4 = os.path.join(ROOT, "ds4")
MHCD_MAGIC = 0x4D484344
PDIST_MAGIC = 0x4D504454  # 'MPDT' — LK target dist (self-distill mode)


def pack_shard(mhcd_path):
    """Read shard.mhcd + .tok (+ optional .pdist) -> dict {tokens, hc[N,hc_dim]
    (, p_idx[N,topN], p_val[N,topN])} or None. HC positions are rebased to 0 (the
    self-distill gen-only shard keys HC by absolute pos); .tok and .pdist are
    already 0-based, so they align after the rebase."""
    tok_path = mhcd_path + ".tok"
    if not os.path.exists(tok_path):
        return None
    td = open(tok_path, "rb").read()
    n = struct.unpack_from("<i", td, 0)[0]
    toks = np.array(struct.unpack_from("<%di" % n, td, 4), np.int32)
    data = open(mhcd_path, "rb").read()
    off, recs = 0, {}
    while off < len(data):
        magic, pos, hcd = struct.unpack_from("<III", data, off)
        off += 12
        if magic != MHCD_MAGIC:
            return None
        hc = np.frombuffer(data, np.float32, hcd, off).copy()
        off += hcd * 4
        recs[pos] = hc
    if not recs or len(recs) != n:
        return None
    rebase = min(recs)  # gen-only shards key HC by absolute pos; prefill keys from 0
    out = {"tokens": toks, "hc": np.stack([recs[rebase + i] for i in range(n)])}

    pd_path = mhcd_path + ".pdist"  # LK target dist, keyed 0..n-1 (self-distill mode)
    if os.path.exists(pd_path):
        pd = open(pd_path, "rb").read()
        off, pidx, pval = 0, {}, {}
        while off < len(pd):
            magic, pos, nn = struct.unpack_from("<III", pd, off)
            off += 12
            if magic != PDIST_MAGIC:
                return None
            pidx[pos] = np.frombuffer(pd, np.int32, nn, off).copy()
            off += nn * 4
            pval[pos] = np.frombuffer(pd, np.float32, nn, off).copy()
            off += nn * 4
        if len(pidx) == n:
            out["p_idx"] = np.stack([pidx[i] for i in range(n)]).astype(np.int32)
            out["p_val"] = np.stack([pval[i] for i in range(n)]).astype(np.float32)
    return out


def run_ds4(model, corpus_path, out_dir, gen):
    """One ds4 --mtp-harvest process (model loaded once) over corpus_path -> out_dir."""
    env = dict(os.environ, DS4_CUDA_FAST_VERIFY="1")
    cmd = [
        DS4, "--cuda", "-m", model,
        "--mtp-harvest", os.path.abspath(corpus_path), os.path.abspath(out_dir),
    ]
    if gen > 0:
        cmd += ["--mtp-harvest-gen", str(gen)]
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def pack_into(raw_dir, out_dir, base_idx, classes):
    """Pack shard_*.mhcd in raw_dir -> shard_<global>.npz in out_dir (global =
    base_idx + local), tag by class, drop raw. Returns list of manifest entries."""
    entries = []
    for mh in sorted(glob.glob(os.path.join(raw_dir, "shard_*.mhcd"))):
        shard = pack_shard(mh)
        local = int(os.path.basename(mh)[len("shard_") : -len(".mhcd")])
        gidx = base_idx + local
        if shard is not None:
            name = f"shard_{gidx:06d}.npz"
            np.savez(os.path.join(out_dir, name), **shard)
            entry = {"shard": name, "n": int(shard["hc"].shape[0])}
            if classes and 0 <= gidx < len(classes):
                entry["class"] = classes[gidx]
            entries.append(entry)
        else:
            print(f"  skip {os.path.basename(mh)} (align/parse/empty)")
        for ext in (".mhcd", ".mhcd.tok", ".mhcd.pdist"):
            p = mh[: -len(".mhcd")] + ext
            if os.path.exists(p):
                os.unlink(p)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ds4flash.gguf")
    ap.add_argument(
        "--gen", type=int, default=0,
        help="self-distill: greedily generate N tokens/doc + dump LK target dist",
    )
    ap.add_argument(
        "--chunk-size", type=int, default=0,
        help="docs per ds4 invocation; >0 enables chunked harvest with resume + "
        "inter-chunk cooldown (thermal-safe for multi-hour runs). 0 = single process.",
    )
    ap.add_argument(
        "--cooldown", type=int, default=120,
        help="seconds to idle the GPU between chunks (thermal). nvidia-smi temp is "
        "N/A on GB10, so this is a fixed sleep.",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # class tags: <corpus>.classes.json maps prompt-index -> category -> per-shard tag.
    cls_path = os.path.abspath(args.corpus) + ".classes.json"
    classes = None
    if os.path.exists(cls_path):
        classes = json.load(open(cls_path)).get("classes")
        print(f"harvest: tagging shards with {len(classes)} class labels from sidecar")

    # single-process path (back-comat): one ds4 over the whole corpus.
    if args.chunk_size <= 0:
        print("harvest: running ds4 --mtp-harvest (single model load) ...")
        raw = os.path.join(args.out, "_raw")
        os.makedirs(raw, exist_ok=True)
        if run_ds4(args.model, args.corpus, raw, args.gen) != 0:
            print("harvest: ds4 --mtp-harvest failed", file=sys.stderr)
            return 1
        manifest = pack_into(raw, args.out, 0, classes)
        os.rmdir(raw) if not os.listdir(raw) else None
        return _write_manifest(args.out, manifest)

    # chunked path: split corpus, harvest each chunk, resume via .done markers,
    # cooldown between chunks. A thermal hard-off only loses the in-flight chunk.
    prompts = [p for p in open(args.corpus, "rb").read().split(b"\x00") if p]
    chunks = [prompts[i : i + args.chunk_size]
              for i in range(0, len(prompts), args.chunk_size)]
    print(f"harvest: {len(prompts)} docs in {len(chunks)} chunks of {args.chunk_size}")
    manifest = []
    for ci, chunk in enumerate(chunks):
        base = ci * args.chunk_size
        done = os.path.join(args.out, f".chunk_{ci:04d}.done")
        if os.path.exists(done):
            manifest += json.load(open(done))["entries"]
            print(f"[chunk {ci + 1}/{len(chunks)}] resume: already done ({base})")
            continue
        raw = os.path.join(args.out, f"_raw_{ci:04d}")
        os.makedirs(raw, exist_ok=True)
        cf = os.path.join(args.out, f".chunk_{ci:04d}.corpus")
        with open(cf, "wb") as f:
            f.write(b"\x00".join(chunk) + b"\x00")
        print(f"[chunk {ci + 1}/{len(chunks)}] harvesting docs {base}..{base + len(chunk)}")
        if run_ds4(args.model, cf, raw, args.gen) != 0:
            print(f"harvest: chunk {ci} ds4 failed", file=sys.stderr)
            return 1
        entries = pack_into(raw, args.out, base, classes)
        manifest += entries
        json.dump({"entries": entries}, open(done, "w"))
        os.rmdir(raw) if not os.listdir(raw) else None
        os.unlink(cf)
        if ci < len(chunks) - 1 and args.cooldown > 0:
            print(f"[chunk {ci + 1}/{len(chunks)}] cooldown {args.cooldown}s ...")
            time.sleep(args.cooldown)
    return _write_manifest(args.out, manifest)


def _write_manifest(out_dir, manifest):
    total = sum(e["n"] for e in manifest)
    hc_dim = 16384
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(
            {"shards": manifest, "total_positions": total,
             "hc_dim": hc_dim, "n_docs": len(manifest)},
            f, indent=2,
        )
    print(f"harvest done: {len(manifest)} docs, {total} positions -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
