"""Validation gate for quant-aware realignment: dequantize the deployed Q4_K
experts from the gguf, orient them to match build_decoder_state's stacking, and
compare to the bf16 .pt experts the trainer currently uses.

Two things this proves:
  1. ORIENTATION — shapes match build_decoder_state exactly and rel-error is a
     small quant-level number (NOT ~100%, which would mean a transposed/misordered
     load). This is the gate before wiring Q4_K experts into training.
  2. THE GAP — the rel-error IS the systematic train/deploy error the frozen
     experts carry that the conditioning path currently never sees. If it's ~Q4_K
     noise (few %) the realignment premise holds; if it's ~0 the lever is moot.

The gguf-dequant is GROUND TRUTH for deployment (ds4 loads the gguf), so it's the
correct frozen-expert value for quant-aware training regardless of the .pt.
"""

import sys

import torch

sys.path.insert(0, "/home/trevor/Projects/ds4/tools/mtp")
sys.path.insert(0, "/home/trevor/Projects/llama.cpp-tjs-fork/gguf-py")
import mtp_head as H  # noqa: E402

MTP_GGUF = H.MTP_GGUF
build_q4k_experts = H.build_q4k_experts


def relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def main():
    sd = torch.load(H.MTP_PT, map_location="cpu", mmap=True, weights_only=True)
    pt = H.build_decoder_state(sd, dtype=torch.float32)
    pt_gu, pt_dn = pt["mlp.experts.gate_up_proj"], pt["mlp.experts.down_proj"]

    q_gu, q_dn = build_q4k_experts(MTP_GGUF)

    print("=== shapes (must match build_decoder_state) ===")
    print(f"gate_up  pt {tuple(pt_gu.shape)}  q4k {tuple(q_gu.shape)}")
    print(f"down     pt {tuple(pt_dn.shape)}  q4k {tuple(q_dn.shape)}")
    assert pt_gu.shape == q_gu.shape and pt_dn.shape == q_dn.shape, "shape mismatch!"

    print("\n=== the train/deploy gap (rel-error: Q4_K-dequant vs bf16 .pt) ===")
    print(f"gate_up  rel-err {relerr(q_gu, pt_gu):.4f}")
    print(f"down     rel-err {relerr(q_dn, pt_dn):.4f}")
    # per-expert spread (does quant hit some experts much harder?)
    pe = torch.tensor([relerr(q_gu[e], pt_gu[e]) for e in range(q_gu.shape[0])])
    print(
        f"per-expert gate_up rel-err: min {pe.min():.4f}  mean {pe.mean():.4f}  "
        f"max {pe.max():.4f}  (spread => router can route around bad experts)"
    )


if __name__ == "__main__":
    main()
