"""Trainable DeepSeek-V4-Flash MTP head as a single nn.Module, for the FastMTP
fine-tune. Wraps transformers' DeepseekV4DecoderLayer (MLA + 256-expert MoE +
Sinkhorn HC) + the nextn glue (enorm/e_proj, hnorm/h_proj, hc-head, mtp.0.norm)
+ frozen base embed/output. Warm-started from mtp_bf16.pt (full-precision source).

Forward is split into the pieces the K-step recursive unroll needs:
  project(tokens, prev_hc) -> input_hc        # e_proj(enorm(embed)) + h_proj(hnorm(prev))
  decode(input_hc, pos, mask, pe) -> out_hc   # the DecoderLayer (HC stream [.,.,hc,D])
  to_logits(out_hc) -> logits                 # hc_head -> mtp.0.norm -> base output head

freeze_for_finetune() freezes the 256 experts (grad still flows through them) and
the base embed/output; everything else (~165M: attn, projections, norms, the two
HyperConnections, hc_head, router gate) stays trainable. Gradient checkpointing on
the decoder keeps the K-step unroll memory ~1x.
"""

from __future__ import annotations
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/trevor/Projects/ds4/tools/mtp")
sys.path.insert(0, "/home/trevor/Projects/llama.cpp-tjs-fork/gguf-py")  # gguf (runtime)
from gguf import GGUFReader, GGMLQuantizationType, quants  # noqa: E402
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M  # noqa: E402
import mtp_head as H  # build_config, build_decoder_state, GLUE, MTP_PT, BASE_GGUF, MTP_LAYER  # noqa: E402


def _deq_base(base_gguf: str, names: set[str]) -> dict[str, torch.Tensor]:
    r = GGUFReader(base_gguf)
    out = {}
    for t in r.tensors:
        if t.name in names:
            arr = quants.dequantize(t.data, GGMLQuantizationType(t.tensor_type))
            out[t.name] = torch.from_numpy(np.ascontiguousarray(arr, np.float32))
    return out


# ---- quant-aware realignment (QLoRA-style error feedback) ---------------------
# Train the fp conditioning path against the EXACT deployment numerics the frozen
# experts run with: Q4_K weights (dequant constant — no STE, weights are frozen)
# and Q8_K activations (straight-through). See probe_q4k.py for the ~7% gap.
def _q8k_ste(x: torch.Tensor) -> torch.Tensor:
    """Per-256-block dynamic Q8_K fake-quant with straight-through gradient,
    matching ds4's cuda_block_q8_K (d = amax/127, int8 levels). Scale is detached
    (kernel's amax is non-differentiable); STE passes identity grad. With the
    amax/127 scale nothing exceeds the range, so no grad range-clip is needed.
    Last dim must be a multiple of 256 (4096 gate_up-in / 2048 down-in both are)."""
    *lead, hd = x.shape
    xb = x.reshape(*lead, hd // 256, 256)
    d = (xb.detach().abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-12)
    xq = ((xb / d).round().clamp(-127, 127) * d).reshape(*lead, hd)
    return x + (xq - x).detach()


def _qa_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Q8_K-fake-quant the activation, then fp32 matmul against the (already
    dequant-Q4_K) weight. fp32 + TF32-off (enforced by the trainer) reproduces
    dev_dot_q4_K_q8_K to within fp32 reduction-order noise (~1e-6)."""
    xq = _q8k_ste(x)
    return torch.nn.functional.linear(xq.float(), w.float()).to(x.dtype)


def _qa_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Quant-aware twin of DeepseekV4Experts.forward: identical routing/gating,
    but the two expert GEMMs go through _qa_linear (Q8_K activations, fp32 acc).
    The non-diff top-k argmax and the quant STE compose orthogonally (no Gumbel)."""
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts
        ).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(mask[expert_idx])
        gate_up = _qa_linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
        current = self._apply_gate(gate_up)
        current = _qa_linear(current, self.down_proj[expert_idx])
        current = current * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, current.to(final.dtype))
    return final


class DeepseekV4MtpHead(nn.Module):
    embed: torch.Tensor  # frozen base token embedding (registered buffer)
    output_w: torch.Tensor  # frozen base lm-head weight (registered buffer)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_size
        self.decoder = M.DeepseekV4DecoderLayer(cfg, H.MTP_LAYER)
        self.hc_head = M.DeepseekV4HyperHead(cfg)
        self.enorm = M.DeepseekV4RMSNorm(D, eps=cfg.rms_norm_eps)
        self.hnorm = M.DeepseekV4RMSNorm(D, eps=cfg.rms_norm_eps)
        self.norm = M.DeepseekV4RMSNorm(D, eps=cfg.rms_norm_eps)
        self.e_proj = nn.Linear(D, D, bias=False)
        self.h_proj = nn.Linear(D, D, bias=False)
        self.rotary = M.DeepseekV4RotaryEmbedding(cfg)
        # frozen base bits (large; persistent=False so they're not in the ckpt)
        self.register_buffer("embed", torch.zeros(cfg.vocab_size, D), persistent=False)
        self.register_buffer(
            "output_w", torch.zeros(cfg.vocab_size, D), persistent=False
        )

    @classmethod
    def from_pt(
        cls, mtp_pt: str = H.MTP_PT, base_gguf: str = H.BASE_GGUF, dtype=torch.bfloat16
    ):
        cfg = H.build_config()
        head = cls(cfg)
        sd = torch.load(mtp_pt, map_location="cpu", mmap=True, weights_only=True)
        head.decoder.load_state_dict(H.build_decoder_state(sd, dtype), strict=True)

        def g(k):
            return sd["mtp.0." + k].to(dtype).clone()

        head.enorm.weight.data = g("enorm.weight")
        head.hnorm.weight.data = g("hnorm.weight")
        head.norm.weight.data = g("norm.weight")
        head.e_proj.weight.data = g("e_proj.weight")
        head.h_proj.weight.data = g("h_proj.weight")
        head.hc_head.load_state_dict(
            {
                "hc_fn": g("hc_head_fn"),
                "hc_base": g("hc_head_base"),
                "hc_scale": g("hc_head_scale"),
            },
            strict=False,
        )
        base = _deq_base(base_gguf, {"token_embd.weight", "output.weight"})
        head.embed = base["token_embd.weight"].reshape(-1, cfg.hidden_size).to(dtype)
        head.output_w = base["output.weight"].reshape(-1, cfg.hidden_size).to(dtype)
        return head.to(dtype)

    def freeze_for_finetune(self):
        self.requires_grad_(True)
        self.decoder.mlp.experts.requires_grad_(False)  # freeze the 256 routed experts
        # embed/output_w are buffers (no grad); nothing else to do.

    def enable_grad_ckpt(self):
        """Wire gradient checkpointing on the bare DecoderLayer. The model-level
        gradient_checkpointing_enable() (which sets _gradient_checkpointing_func)
        isn't available on a standalone GradientCheckpointingLayer, so set both."""
        import functools
        from torch.utils.checkpoint import checkpoint

        self.decoder.gradient_checkpointing = True
        # private attr the GradientCheckpointingLayer reads; normally set by the
        # model-level enable(). setattr keeps it dynamic (ty-clean).
        setattr(
            self.decoder,
            "_gradient_checkpointing_func",
            functools.partial(checkpoint, use_reentrant=False),
        )

    def enable_quant_aware(self, mtp_gguf: str = H.MTP_GGUF):
        """Quant-aware realignment: replace the frozen bf16 experts with the
        DEPLOYED Q4_K-dequant experts (the ~7% gap they actually run with), and
        route the expert GEMMs through Q8_K-quantized activations. The trainable
        conditioning path (attn/proj/norms/router) then realigns to the true
        deployment numerics. Call AFTER freeze_for_finetune. The trainer must set
        torch.backends.cuda.matmul.allow_tf32 = False so the fp32 fake-quant
        matmul isn't silently TF32 (which would realign to the wrong numerics)."""
        import types

        gate_up, down = H.build_q4k_experts(mtp_gguf)
        exp = self.decoder.mlp.experts
        dt = exp.gate_up_proj.dtype
        with torch.no_grad():
            exp.gate_up_proj.copy_(gate_up.to(dt))  # ty: ignore[call-non-callable]
            exp.down_proj.copy_(down.to(dt))  # ty: ignore[call-non-callable]
        exp.requires_grad_(False)  # stay frozen
        exp.forward = types.MethodType(_qa_experts_forward, exp)
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---- forward pieces ------------------------------------------------------
    def project(self, tokens: torch.Tensor, prev_hc: torch.Tensor) -> torch.Tensor:
        """tokens [B,S] long ; prev_hc [B,S,hc,D] -> input_hc [B,S,hc,D]."""
        e = self.e_proj(self.enorm(self.embed[tokens]))  # [B,S,D]
        h = self.h_proj(self.hnorm(prev_hc))  # [B,S,hc,D]
        return e.unsqueeze(2) + h

    def rope_mask(self, S: int, position_ids: torch.Tensor, device, dtype):
        ref = torch.zeros(1, S, self.cfg.hidden_size, dtype=dtype, device=device)
        pe = {
            "main": self.rotary(ref, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary(
                ref, position_ids=position_ids, layer_type="compress"
            ),
        }
        # STRICT causal additive mask over the unroll sequence. The HF
        # create_sliding_window_causal_mask helper (with a DynamicCache + non-zero
        # position_ids=i+k) is malformed for this synthetic batched-position
        # unroll: it over-masks legitimate past AND admits a forward channel that
        # training over-optimizes into a degenerate flat accept=1.0 (target leak).
        # Position i must never see slot i+1, whose projected input embeds t_{i+2}
        # (== the MTP target). triu(diag=1) guarantees that; the sliding band keeps
        # it faithful to the "sliding_attention" MTP layer when window < S.
        m = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
        m = torch.triu(m, diagonal=1)
        win = getattr(self.cfg, "sliding_window", None)
        if win and win < S:
            band = torch.tril(
                torch.full((S, S), float("-inf"), device=device, dtype=dtype),
                diagonal=-win,
            )
            m = m + band
        mask = m.view(1, 1, S, S)
        return pe, mask

    def decode(self, input_hc, position_ids, mask, pe):
        """input_hc [B,S,hc,D] -> out_hc [B,S,hc,D] (the DecoderLayer HC stream)."""
        return self.decoder(
            input_hc,
            input_ids=None,
            position_ids=position_ids,
            position_embeddings=pe,
            attention_mask=mask,
        )

    def to_logits(self, out_hc):
        """out_hc [B,S,hc,D] -> logits [B,S,vocab]."""
        collapsed = self.hc_head(out_hc)  # [B,S,D]
        return self.norm(collapsed) @ self.output_w.t()
