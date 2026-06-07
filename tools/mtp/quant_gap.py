"""Measure the Q4_K quantization gap AND the deployed training gain on the MTP head.

FastMTP shipped fp16 (train == deploy). We export the head through Q4_K experts +
Q8_K activations and never realign post-quant, so the deployed head carries pure
quantization damage. This probe runs accept_proxy in a 2x2 on the SAME eval shards:

           bf16 (ceiling)      quant (deployed-equiv)
  untrained   U_bf16               U_q
  trained     T_bf16               T_q

Decisive numbers:
  U_bf16 - U_q : pure Q4_K penalty (ceiling on what quant-aware can recover)
  T_bf16 - T_q : deploy gap our bf16-trained head pays at export
  T_q   - U_q  : NET deployed accept gain from training (the real A/B, proxy form)

Usage: uv run --project tools/mtp python tools/mtp/quant_gap.py \
         --shards DIR [--trained tools/mtp/mtp_sd_lk_K2.pt] [--n 150]
"""

import argparse

import mtp_model as MM
import torch
from train_mtp import accept_proxy, list_shards


def _eval(head, shards, classes, K, max_seq):
    head.eval()
    return accept_proxy(head, shards, classes, K, max_seq, "cuda", torch.bfloat16, head.cfg)


def _load_trained(path, dtype):
    head = MM.DeepseekV4MtpHead.from_pt(dtype=dtype).to("cuda")
    blob = torch.load(path, map_location="cuda")
    head.load_state_dict(blob["trainable"], strict=False)  # frozen weights already in
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--trained", default="tools/mtp/mtp_sd_lk_K2.pt")
    ap.add_argument("--n", type=int, default=150, help="eval shards to sample (tail)")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--max_seq", type=int, default=256)
    args = ap.parse_args()

    dtype = torch.bfloat16
    shards, classes, _ = list_shards(args.shards)
    eval_shards = shards[-args.n :]
    print(f"quant-gap 2x2: {len(eval_shards)} eval shards, K={args.K}\n")
    res = {}

    # --- untrained head ---
    head = MM.DeepseekV4MtpHead.from_pt(dtype=dtype).to("cuda")
    res["U_bf16"] = _eval(head, eval_shards, classes, args.K, args.max_seq)
    head.enable_quant_aware()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    res["U_q"] = _eval(head, eval_shards, classes, args.K, args.max_seq)
    del head
    torch.cuda.empty_cache()

    # --- trained head ---
    head = _load_trained(args.trained, dtype)
    res["T_bf16"] = _eval(head, eval_shards, classes, args.K, args.max_seq)
    head.enable_quant_aware()
    res["T_q"] = _eval(head, eval_shards, classes, args.K, args.max_seq)

    def k2(r):
        return r["accept_k2"]

    def k1(r):
        return r["accept_k1"]

    print("              k1       k2")
    for tag in ("U_bf16", "U_q", "T_bf16", "T_q"):
        print(f"  {tag:7s}   {k1(res[tag]):.4f}   {k2(res[tag]):.4f}")
    print("\n  --- k2 deltas ---")
    print(f"  pure Q4_K penalty   U_bf16-U_q = {k2(res['U_bf16']) - k2(res['U_q']):+.4f}")
    print(f"  trained deploy gap  T_bf16-T_q = {k2(res['T_bf16']) - k2(res['T_q']):+.4f}")
    print(f"  NET deployed gain   T_q  -U_q  = {k2(res['T_q']) - k2(res['U_q']):+.4f}  <-- the real A/B")
    print(f"  bf16 train gain     T_bf16-U_bf16 = {k2(res['T_bf16']) - k2(res['U_bf16']):+.4f}")


if __name__ == "__main__":
    main()
