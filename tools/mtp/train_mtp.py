"""FastMTP fine-tune of the DeepSeek-V4-Flash MTP head (Lever A), on HF Trainer.

Warm-start the original head, FREEZE the 256 routed experts, train the
conditioning path to do recursive multi-step drafting. K-step recursive unroll
with exponential-decay weighted CE (beta=0.6).

Built on transformers.Trainer — the framework owns the loop, schedule, grad-accum,
clipping, bf16, wandb logging, resumable+best checkpointing, and eval
orchestration. We implement ONLY what's special to this objective:
  * unroll_steps   — the recursive K-step draft (prev_{k+1}=out_k) + index shifts
  * compute_loss   — weighted CE over the unroll (MtpTrainer override)
  * evaluation_loop— per-step-k top-1 accept proxy + per-class + leak guard
  * ThermalCallback— GB10 hard-off survival (cool down before a step over max_c)

THE TWO LOAD-BEARING INDEX SHIFTS (in unroll_steps; mirror FastMTP 2.1 + ds4):
  base position i (skip i=0=BOS); draft step k=1..K:
    input  = project(prev[i], tok=t_{i+k}) ;  target = t_{i+k+1}
    rotary position = i+k  ;  valid 1<=i<=N-2-k ;  prev_{k+1}[i] = out_k[i]
  k=1 <-> ds4 drafts[0] (~79-88%); k=2 <-> drafts[1] (~22-60%, the target).

Inputs: harvested .npz shards {tokens int32[N], hc float32[N,hc_dim]}.
Output: Trainer checkpoints (resume/best) + fp16 trainable-only export for
export_gguf. DOES NOT modify ds4. GB10 GPU.
"""

import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, "/home/trevor/Projects/ds4/tools/mtp")
import lk_loss as LK  # noqa: E402
import mtp_model as MM  # noqa: E402
import trainlib as TL  # noqa: E402

# Silence transformers' per-process standalone-module notices (AttentionInterface /
# ExpertsInterface "config._*_implementation set to None") — expected when we drive
# a DecoderLayer directly; they're warning-level, not errors.
from transformers.utils import logging as _hf_logging  # noqa: E402

_hf_logging.set_verbosity_error()

from transformers import (  # noqa: E402
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import EvalLoopOutput  # noqa: E402


# ---- objective: the recursive unroll (the genuinely special part) -------------
def alpha_weights(K, beta):
    w = [beta**k for k in range(K)]
    s = sum(w)
    return [x / s for x in w]


def unroll_steps(head, tokens, hc, K, device, dtype):
    """Yield (k, logits[L,vocab], targets[L], tgt_pos[L]) per draft step. Shared by
    train loss and accept eval; the recursion (prev_{k+1}[i]=out_k[i]) and the index
    shifts live here. tgt_pos = the absolute positions of the predicted tokens (so
    the LK loss can gather the harvested target dist p there). THE correctness block."""
    N = tokens.shape[0]
    prev = hc  # prev[i] = base HC h_i  (i=0 BOS, never used since i starts at 1)
    for k in range(1, K + 1):
        i_hi = N - 2 - k  # i in [1, i_hi]; need t_{i+k}, t_{i+k+1}
        if i_hi < 1:
            break
        idx = torch.arange(1, i_hi + 1, device=device)
        in_tok = tokens[idx + k]  # t_{i+k}
        tgt_pos = idx + k + 1  # positions of t_{i+k+1}
        tgt = tokens[tgt_pos]  # t_{i+k+1}
        prev_in = prev.index_select(0, idx)  # [L,hc,D]
        pos = (idx + k).unsqueeze(0)  # [1,L] rotary absolute position
        L = idx.shape[0]
        input_hc = head.project(in_tok.unsqueeze(0), prev_in.unsqueeze(0))
        pe, mask = head.rope_mask(L, pos, device, dtype)
        out = head.decode(input_hc, pos, mask, pe)  # [1,L,hc,D]
        logits = head.to_logits(out)[0]  # [L,vocab]
        yield k, logits, tgt, tgt_pos
        prev = prev.index_copy(0, idx, out[0])  # recursion for next step


def doc_loss(head, tokens, hc, K, weights, device, dtype, *, loss="ce", p=None,
             eta=3.0, margin_w=0.0, margin_m=2.0):
    """Weighted per-step draft loss. loss='ce' -> hard-label CE (FastMTP); 'lk_alpha'
    / 'lk_hybrid' -> LK acceptance-direct loss vs the harvested target dist p (a
    tuple (p_idx[N,topN] long, p_val[N,topN]) over absolute positions; gathered at
    each step's tgt_pos); 'lk_margin' -> lk_hybrid + margin_w·rank-1 hinge (sharpen
    top-1 toward the top-k ceiling). LK falls back to CE if p is None (accept eval)."""
    total, perk = None, {}
    for k, logits, tgt, tgt_pos in unroll_steps(head, tokens, hc, K, device, dtype):
        if p is not None and loss in ("lk_hybrid", "lk_margin"):
            lk = LK.lk_hybrid_loss(logits, p[0][tgt_pos], p[1][tgt_pos], eta)
        elif p is not None and loss == "lk_alpha":
            lk = LK.lk_alpha_loss(logits, p[0][tgt_pos], p[1][tgt_pos])
        else:
            lk = F.cross_entropy(logits.float(), tgt)
        if margin_w > 0.0 or loss == "lk_margin":
            mw = margin_w if margin_w > 0.0 else 0.5  # lk_margin default weight
            lk = lk + mw * LK.margin_loss(logits, tgt, margin_m)
        total = weights[k - 1] * lk if total is None else total + weights[k - 1] * lk
        perk[f"loss_k{k}"] = float(lk.detach())
    return total, perk


@torch.no_grad()
def accept_proxy(head, shards, classes, K, max_seq, device, dtype, cfg):
    """Held-out per-step-k top-1 agreement (proxy for MTP draft acceptance), +
    per-class breakdown. This is the REAL target metric (CE down != accept up)."""
    was_training = head.training
    head.eval()
    acc = {k: [0, 0] for k in range(1, K + 1)}
    per_cls = {}
    for sp in shards:
        toks, hc = load_shard(sp, max_seq, device, dtype, cfg)
        if toks.shape[0] < K + 3:
            continue
        cls = classes.get(sp, "all")
        pc = per_cls.setdefault(cls, {k: [0, 0] for k in range(1, K + 1)})
        for k, logits, tgt, _ in unroll_steps(head, toks, hc, K, device, dtype):
            corr = int((logits.argmax(-1) == tgt).sum())
            tot = int(tgt.shape[0])
            acc[k][0] += corr
            acc[k][1] += tot
            pc[k][0] += corr
            pc[k][1] += tot
    if was_training:
        head.train()
    out = {f"accept_k{k}": (acc[k][0] / acc[k][1] if acc[k][1] else 0.0) for k in acc}
    for cls, pc in per_cls.items():
        for k in pc:
            if pc[k][1]:
                out[f"accept/{cls}_k{k}"] = pc[k][0] / pc[k][1]
    # headline best-metric: mean accept over the CHAIN we're fixing (k>=2)
    chain = [out[f"accept_k{k}"] for k in range(2, K + 1) if f"accept_k{k}" in out]
    out["accept_chain"] = (
        sum(chain) / len(chain) if chain else out.get("accept_k1", 0.0)
    )
    return out


# ---- data ---------------------------------------------------------------------
def list_shards(shards_dir):
    """-> (shard_paths, class_by_path, npositions_by_path). Lengths come from the
    harvest manifest so we can drop too-short docs without opening every file."""
    man_path = os.path.join(shards_dir, "manifest.json")
    classes, lengths = {}, {}
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        for e in man.get("shards", []):
            p = os.path.join(shards_dir, e["shard"])
            classes[p] = e.get("class", "all")
            lengths[p] = e.get("n", 1 << 30)
    shards = sorted(glob.glob(os.path.join(shards_dir, "shard_*.npz")))
    return shards, classes, lengths


def load_shard(path, max_seq, device, dtype, cfg):
    d = np.load(path)
    toks = torch.from_numpy(d["tokens"].astype(np.int64))[:max_seq].to(device)
    N = toks.shape[0]
    hc = (
        torch.from_numpy(d["hc"][:N])
        .to(device, dtype)
        .reshape(N, cfg.hc_mult, cfg.hidden_size)
    )
    return toks, hc


class ShardDataset(Dataset[dict]):
    """One doc per item; load happens in dataloader workers. Returns raw arrays
    (truncated) — Trainer moves them to device, MtpTrainer.compute_loss reshapes."""

    def __init__(self, paths, max_seq):
        self.paths = paths
        self.max_seq = max_seq

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> dict:
        d = np.load(self.paths[index])
        out = {
            "tokens": d["tokens"].astype(np.int64)[: self.max_seq],
            "hc": d["hc"][: self.max_seq],  # [N, hc_dim] f32
        }
        # optional harvested target dist for the LK loss (top-N per position)
        if "p_idx" in d:
            out["p_idx"] = d["p_idx"].astype(np.int64)[: self.max_seq]
            out["p_val"] = d["p_val"][: self.max_seq]
        return out


def collate(features):
    # per_device_train_batch_size=1 — the unroll is per-doc; no padding/stacking.
    f = features[0]
    out = {"tokens": torch.as_tensor(f["tokens"]), "hc": torch.as_tensor(f["hc"])}
    if "p_idx" in f:
        out["p_idx"] = torch.as_tensor(f["p_idx"])
        out["p_val"] = torch.as_tensor(f["p_val"])
    return out


# ---- the trainer: only the special hooks --------------------------------------
class ThermalCallback(TrainerCallback):
    """GB10 long soak can hard-power-off the box; cool down before a hot step."""

    def __init__(self, guard):
        self.guard = guard

    def on_step_begin(self, args, state, control, **kw):
        self.guard.maybe_cooldown()


class RunInfoCallback(TrainerCallback):
    """Log the dataset fingerprint + per-category counts + per-category base
    confidence (mean p_val[:,0] = accept-difficulty/headroom signal) + git SHA to
    wandb config at train start, so runs are reproducible, cross-comparable, and the
    per-cat accept curves can be read against each category's difficulty."""

    def __init__(self, info):
        self.info = info

    def on_train_begin(self, args, state, control, **kw):
        try:
            import wandb

            if wandb.run is not None:
                wandb.config.update(self.info, allow_val_change=True)
        except Exception:  # noqa: BLE001  (telemetry must never break training)
            pass


def build_run_info(train_shards, eval_shards, classes, lengths, targs, margs):
    """Reproducibility + tuning telemetry for wandb config (all best-effort)."""
    info = {"shards_dir": margs.shards, "loss": margs.loss, "K": margs.K,
            "gamma": margs.gamma, "max_seq": margs.max_seq}
    try:
        import subprocess
        info["git_sha"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    eff_batch = targs.per_device_train_batch_size * targs.gradient_accumulation_steps
    info["eff_batch"] = eff_batch
    info["n_train"] = len(train_shards)
    info["n_eval"] = len(eval_shards)
    info["train_positions"] = int(sum(lengths.get(s, 0) for s in train_shards))
    info["est_opt_steps"] = int(len(train_shards) / max(1, eff_batch) * targs.num_train_epochs)

    def cat_counts(sh):
        c = {}
        for s in sh:
            c[classes.get(s, "all")] = c.get(classes.get(s, "all"), 0) + 1
        return c

    info["per_cat_train"] = cat_counts(train_shards)
    info["per_cat_eval"] = cat_counts(eval_shards)
    # base confidence per cat (mean top-1 prob over eval shards) = accept ceiling proxy
    try:
        conf = {}
        for s in eval_shards:
            d = np.load(s)
            if "p_val" in d:
                cls = classes.get(s, "all")
                conf.setdefault(cls, []).append(float(d["p_val"][:, 0].mean()))
        info["per_cat_base_conf"] = {k: round(sum(v) / len(v), 3) for k, v in conf.items()}
    except Exception:  # noqa: BLE001
        pass
    return info


class MtpTrainer(Trainer):
    def __init__(
        self, *a, K, alpha, eval_shards, classes, max_seq, mtp_cfg,
        loss="ce", lk_eta=3.0, margin_w=0.0, margin_m=2.0, **kw,
    ):
        super().__init__(*a, **kw)
        self.K = K
        self.alpha = alpha
        self._eval_shards = eval_shards
        self._classes = classes
        self.max_seq = max_seq
        self.mtp_cfg = mtp_cfg
        self.loss = loss
        self.lk_eta = lk_eta
        self.margin_w = margin_w
        self.margin_m = margin_m
        self._last_perk = {}
        self._warned_no_p = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        toks = inputs["tokens"].to(device)
        N = toks.shape[0]
        hc = (
            inputs["hc"]
            .to(device, dtype)
            .reshape(N, self.mtp_cfg.hc_mult, self.mtp_cfg.hidden_size)
        )
        p = None
        if "p_idx" in inputs:
            p = (inputs["p_idx"].to(device), inputs["p_val"].to(device, torch.float32))
        elif self.loss != "ce" and not self._warned_no_p:
            print(f"  WARN: --loss {self.loss} but shards have no p (top-N target "
                  f"dist) — falling back to CE. Harvest with the base-dist dump.")
            self._warned_no_p = True
        loss, perk = doc_loss(
            model, toks, hc, self.K, self.alpha, device, dtype,
            loss=self.loss, p=p, eta=self.lk_eta,
            margin_w=self.margin_w, margin_m=self.margin_m,
        )
        if loss is None or not torch.isfinite(loss):
            # skip (too short / non-finite): zero loss still wired to params so
            # backward is a clean no-op rather than a crash.
            loss = next(p for p in model.parameters() if p.requires_grad).sum() * 0.0
            perk = {}
        self._last_perk = perk
        return (loss, None) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._last_perk:
            logs = {**logs, **self._last_perk}
        return super().log(logs, *args, **kwargs)

    # Slim checkpoints: persist ONLY the 167M trainable params, not the 6.4B
    # frozen experts (a full save would be ~13GB/ckpt AND trip safetensors'
    # shared-memory check on the tied embed/output). Frozen weights come from
    # from_pt() at construction, so strict=False reload is correct here.
    _TRAINABLE = "trainable.bin"

    def _save(self, output_dir=None, state_dict=None):
        out = output_dir or self.args.output_dir
        assert out is not None
        os.makedirs(out, exist_ok=True)
        model = self.model
        assert model is not None
        sd = {
            n: p.detach().cpu()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        torch.save(sd, os.path.join(out, self._TRAINABLE))
        torch.save(self.args, os.path.join(out, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = model or self.model
        assert model is not None
        sd = torch.load(
            os.path.join(resume_from_checkpoint, self._TRAINABLE),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(sd, strict=False)  # frozen weights already loaded

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        model = self.model
        assert model is not None
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        m = accept_proxy(
            model,
            self._eval_shards,
            self._classes,
            self.K,
            self.max_seq,
            device,
            dtype,
            self.mtp_cfg,
        )
        ks = [m.get(f"accept_k{k}", 0.0) for k in range(1, self.K + 1)]
        # Leak guard: legitimate accept decays with depth. A flat near-1.0 profile
        # means the target is reachable from the input (mask/recursion leak).
        leak = min(ks) > 0.95 or (ks[0] > 0.95 and ks[-1] > 0.9)
        print(
            f"  [eval] step {self.state.global_step} "
            f"accept_chain={m['accept_chain']:.3f} "
            + " ".join(f"k{k}={v:.2f}" for k, v in enumerate(ks, 1))
            + ("   ** SUSPICIOUS: flat/high — possible leak **" if leak else ""),
            flush=True,
        )
        metrics = {f"{metric_key_prefix}_{k}": float(v) for k, v in m.items()}
        # operational telemetry to wandb. nvidia-smi mem is N/A on GB10, but torch's
        # allocator view works -> tells us headroom to scale batch/seq.
        for gk, gv in TL.gpu_stats().items():
            metrics[f"{metric_key_prefix}_{gk}"] = float(gv)
        if torch.cuda.is_available():
            metrics[f"{metric_key_prefix}_gpu_mem_gb"] = (
                torch.cuda.max_memory_allocated() / 1e9
            )
        return EvalLoopOutput(
            predictions=None,  # ty: ignore[invalid-argument-type]  (accept-only eval)
            label_ids=None,
            metrics=metrics,
            num_samples=len(self._eval_shards),
        )


def export_trainable(head, path, extra=None):
    """fp16 trainable-only dump for export_gguf (no optimizer/rng)."""
    sd = {
        n: p.detach().to(torch.float16).cpu()
        for n, p in head.named_parameters()
        if p.requires_grad
    }
    torch.save({"trainable": sd, **(extra or {})}, path)
    return path


@dataclass
class MtpTrainingArguments(TrainingArguments):
    """transformers.TrainingArguments with our defaults (all still CLI-overridable
    via HfArgumentParser). Only fields whose default we change from upstream — every
    generic training knob (and its validation) is owned by TrainingArguments."""

    # Defaults = the FastMTP recipe (LR/cosine/warmup/betas/clip) where it isn't
    # scale-coupled; eff-batch + seq + K are adjusted to our scale/deploy (see RECIPE_NOTES).
    output_dir: str = "tools/mtp/ckpt_v2"
    per_device_train_batch_size: int = 1  # unroll is per-doc; use grad-accum for batch
    gradient_accumulation_steps: int = 32  # eff-batch 32: scale-adjusted from FastMTP's 64
    num_train_epochs: float = 3.0          # FastMTP
    learning_rate: float = 5e-5            # FastMTP
    lr_scheduler_type: str = "cosine"      # FastMTP
    warmup_ratio: float = 0.05             # FastMTP
    adam_beta2: float = 0.95               # FastMTP (beta1 0.9 = upstream default)
    max_grad_norm: float = 1.0
    logging_steps: float = 10              # finer loss curves to wandb
    eval_strategy: str = "steps"
    eval_steps: float = 50                 # finer per-category accept curves
    save_strategy: str = "steps"
    save_steps: float = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accept_chain"
    greater_is_better: bool = True
    remove_unused_columns: bool = False  # keep our tokens/hc/p_idx/p_val columns
    dataloader_num_workers: int = 2
    report_to: str = "wandb"  # "none" to disable
    seed: int = 1234


@dataclass
class MtpArgs:
    """Args specific to the MTP head retrain — everything generic lives in
    MtpTrainingArguments."""

    shards: str = field(metadata={"help": "dir of harvested .npz shards"})
    K: int = 2  # draft-2 deploy focus (draft-3 dead on GB10)
    beta: float = 0.6  # CE per-step decay (only used when loss=ce)
    loss: str = field(
        default="lk_hybrid",
        metadata={"choices": ["ce", "lk_alpha", "lk_hybrid", "lk_margin"]},
    )
    lk_eta: float = 3.0  # LK adaptive-lambda rate
    margin_weight: float = 0.0  # >0 adds a rank-1 hinge to any loss; lk_margin=0.5 default
    margin_m: float = 2.0  # hinge margin in logit space
    gamma: float = 1.0  # LK per-step decay; 1.0 = equal k1/k2 weight (draft-2 neutral)
    max_seq: int = 320  # = harvest gen cap, so no doc is truncated
    max_docs: int = 0
    eval_per_class: int = 8  # held-out eval docs PER category (fast, per-cat coverage)
    max_temp: float = 84.0  # GB10 thermal guard
    export: str = "tools/mtp/mtp_v2.pt"
    quant_aware: bool = False  # realign vs deployed Q4_K experts + Q8_K activations
    early_stop: int = 4  # patience (evals w/o eval_accept_chain gain); 0 = off
    smoke: bool = False  # tiny end-to-end run: ~60 docs, fast eval, wandb + save +
    # export all exercised — validate the whole system before the multi-hour run


def main():
    parsed = HfArgumentParser(
        [MtpTrainingArguments, MtpArgs]  # ty: ignore[invalid-argument-type]
    ).parse_args_into_dataclasses()
    targs = cast(MtpTrainingArguments, parsed[0])
    margs = cast(MtpArgs, parsed[1])

    # Sync to wandb by default (override with WANDB_MODE=offline). We're authed.
    os.environ.setdefault("WANDB_MODE", "online")

    if margs.smoke:
        # Extremely small end-to-end run: exercises train step + per-cat eval + wandb
        # sync + checkpoint save/best-reload + export, in ~1-2 min, BEFORE the multi-
        # hour run. Same code paths, tiny scale. Only override what wasn't set explicitly.
        if not margs.max_docs:
            margs.max_docs = 60
        margs.eval_per_class = 2
        margs.early_stop = 0
        targs.num_train_epochs = 1.0
        targs.gradient_accumulation_steps = 2
        targs.eval_steps = 5
        targs.save_steps = 10
        targs.logging_steps = 1
        targs.save_total_limit = 1
        if targs.run_name is None:
            targs.run_name = "smoke"
        targs.output_dir = "tools/mtp/ckpt_smoke"
        margs.export = "/tmp/mtp_smoke_export.pt"
        print("=== SMOKE: tiny end-to-end validation (eval + wandb + save + export) ===")

    TL.set_seed(targs.seed)
    dtype = torch.bfloat16
    if margs.quant_aware:
        # The fake-quant expert matmul MUST be true fp32 — TF32/bf16 would inject a
        # second, wrong quantization and realign training to the wrong numerics.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # CE decays step-weights by beta; LK by gamma (1.0 = equal k1/k2, draft-2 neutral).
    decay = margs.gamma if margs.loss != "ce" else margs.beta
    alpha = alpha_weights(margs.K, decay)
    if targs.run_name is None:
        targs.run_name = f"{margs.loss}_K{margs.K}_lr{targs.learning_rate:.0e}"
    targs.disable_tqdm = not sys.stderr.isatty()
    print(
        f"loss={margs.loss}"
        f"{f' (eta={margs.lk_eta})' if margs.loss == 'lk_hybrid' else ''}"
        f" K={margs.K} decay={decay} alpha={[round(a, 3) for a in alpha]} "
        f"lr={targs.learning_rate} accum={targs.gradient_accumulation_steps}"
    )

    head = MM.DeepseekV4MtpHead.from_pt(dtype=dtype).to("cuda")
    head.freeze_for_finetune()
    if margs.quant_aware:
        head.enable_quant_aware()
        print("quant-aware: experts <- deployed Q4_K dequant, activations Q8_K (STE)")
    head.enable_grad_ckpt()
    n_train = sum(p.numel() for p in head.trainable_parameters())
    print(f"trainable params: {n_train / 1e6:.1f}M (experts frozen)")

    shards, classes, lengths = list_shards(margs.shards)
    if margs.max_docs:
        shards = shards[: margs.max_docs]
    shards = [s for s in shards if lengths.get(s, 1 << 30) >= margs.K + 3]
    # STRATIFIED eval split: hold out a FIXED eval_per_class docs from EACH category so
    # the during-training eval is FAST (bounded ~eval_per_class*n_cats docs) yet gives
    # per-category accept (a contiguous tail starves rare cats). Shard order is already
    # shuffled (corpus order), so the per-class tail is a random hold-out.
    by_cls = {}
    for s in shards:
        by_cls.setdefault(classes.get(s, "all"), []).append(s)
    eval_shards, train_shards = [], []
    if len(shards) > 4:
        for cls in sorted(by_cls):
            sl = by_cls[cls]
            k = min(margs.eval_per_class, max(0, len(sl) - 1)) if len(sl) >= 2 else 0
            eval_shards += sl[len(sl) - k :] if k else []
            train_shards += sl[: len(sl) - k] if k else sl
    else:
        train_shards = shards
    print(f"shards: {len(train_shards)} train / {len(eval_shards)} eval "
          f"({margs.eval_per_class}/class over {len(by_cls)} categories)")

    has_eval = bool(eval_shards)
    if not has_eval:  # no eval set -> disable eval + best-model tracking
        targs.eval_strategy = "no"
        targs.load_best_model_at_end = False

    trainer = MtpTrainer(
        model=head,
        args=targs,
        train_dataset=ShardDataset(train_shards, margs.max_seq),
        eval_dataset=ShardDataset(eval_shards, margs.max_seq) if has_eval else None,
        data_collator=collate,
        callbacks=(
            [ThermalCallback(TL.ThermalGuard(max_c=margs.max_temp))]
            + [RunInfoCallback(build_run_info(
                train_shards, eval_shards, classes, lengths, targs, margs))]
            + (
                [EarlyStoppingCallback(early_stopping_patience=margs.early_stop)]
                if margs.early_stop > 0 and has_eval
                else []
            )
        ),
        K=margs.K,
        alpha=alpha,
        eval_shards=eval_shards,
        classes=classes,
        max_seq=margs.max_seq,
        mtp_cfg=head.cfg,
        loss=margs.loss,
        lk_eta=margs.lk_eta,
        margin_w=margs.margin_weight,
        margin_m=margs.margin_m,
    )

    trainer.train(resume_from_checkpoint=targs.resume_from_checkpoint or None)

    export_trainable(head, margs.export, extra={"K": margs.K, "beta": margs.beta})
    final = (
        accept_proxy(
            head, eval_shards, classes, margs.K, margs.max_seq, "cuda", dtype, head.cfg
        )
        if has_eval
        else {}
    )
    print(f"done. final {final}\nexport -> {margs.export}, ckpts -> {targs.output_dir}")


if __name__ == "__main__":
    main()
