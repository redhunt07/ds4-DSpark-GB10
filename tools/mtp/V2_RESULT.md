# DeepSeek-V4-Flash MTP head retrain — v2 result

**Shipped:** `DeepSeek-V4-Flash-MTP-v2.gguf` (head weights `tools/mtp/mtp_v2.pt`).
**Outcome:** **+7 OOD draft-2 accept / +13% decode tok/s** on real prompts, deployment-
faithful (Q4_K-free), bilingual (EN+ZH). Frozen experts; only the head's attn/proj/
norms trained (167.6M params).

Lever A = retrain the V4-Flash MTP head to raise speculative-decode acceptance on
GB10 (decode ∝ accept; plain decode is at the membw wall). This is the result + the
full lever-exploration record (the negative results bound what's worth trying next).

---

## The fix that mattered: train/deploy distribution match

v1 hit **+18.5 k2 in-distribution but only +3.4 OOD**. Root cause: the harvest
tokenized **raw prompts** (`ds4_cli.c run_mtp_harvest`), so the base model
*document-continued* them into dataset/JSON artifacts (`{"answer":...}`,
`{"model":"gpt-4o-mini","messages":[...]}`) instead of answering — while deployment
(`ds4_agent`) wraps every turn in the chat template. The head trained on one
distribution and drafted on another.

**Fix:** gen-mode harvest now calls `ds4_encode_chat_prompt(.., DS4_THINK_NONE, ..)`
(`<｜User｜> prompt <｜Assistant｜> </think>`, matching `--nothink` deploy). Verified
16/16 smoke gens flipped from JSON garbage to clean assistant responses. This is what
made v2 work — found via the `analyze_harvest.py` ZH-coherence check before committing
a multi-hour harvest. **Always eyeball decoded generations before trusting harvest data.**

---

## Recipe (baked into `train_mtp.py` defaults)

FastMTP (arXiv 2509.18362) where not scale-coupled; adjusted where it is:

| knob | value | source |
| --- | --- | --- |
| loss | `lk_hybrid` (LK acceptance-direct, arXiv 2602.23881) | improvement over CE |
| K | 2 | draft-2 deploy (draft-3 dead on GB10) |
| LR / sched / warmup / betas | 5e-5 / cosine / 0.05 / (0.9, 0.95) | FastMTP |
| epochs | 3 | FastMTP |
| eff-batch | 32 | scale-adjusted (FastMTP's 64 starves steps at 5.9K docs) |
| max_seq | 320 | = harvest gen cap (no truncation) |
| frozen | base + embed + output + experts | FastMTP-style |

Self-distillation: ds4 greedily generates responses through its real Q4_K path
(chat-templated), dumping per generated position `(base HC 16384-d, token, base top-64
next-token dist p)`. The head trains to reproduce the base's hidden-state→distribution
map via LK loss → drafts what the base will actually pick.

## Dataset (open-aperture, tagged, bilingual)

`build_dataset.py`: **5,898 prompts / 1.365M positions, 20 categories** (5 measured +
11 aperture task-types), **15% ZH** (alpaca-gpt4-data-zh; firefly-1.1M rejected as
noisy). Streamed from OpenHermes-2.5. Harvested via the hardened pipeline (chunked,
resumable `.done` markers, inter-chunk thermal cooldown). Zero degenerate drops with
the chat template (every gen is a valid response).

---

## Results

**In-distribution (held-out, stratified 8/class):** k1 **0.851**, k2 **0.740**.
**Deployed (Q4_K) proxy:** untrained 0.602 → trained **0.765** k2 (+0.162); quant
penalty ~0 (Q4_K is free on the head — small logit perturbations rarely flip the argmax).
**OOD 20-category A/B (real prompts, draft-2, deterministic):** aggregate
**+7 accept / +13% tps**; on the original 5 cats **+4.8 vs v1's +3.4**.

Per-category accept tracks the base's own predictability (entropy):

| tier | categories (k2) |
| --- | --- |
| high 0.80–0.91 | extraction · code · math · open-chat · **zh-code** · structured · rewriting · translation |
| mid 0.65–0.79 | classification · factual · **zh-chat** · summarization · planning · **zh-qa** · chat-essay · analytical |
| floor <0.65 | roleplay · prose |

**ZH works** (no firefly regression): zh-code 0.855, zh-chat 0.718, zh-qa 0.681.

---

## Lever exploration — what's exhausted (and why v2 is the ceiling)

Every lever measured; all spent. The negative results are the deliverable — they say
what *not* to spend GPU on next.

| lever | result | why |
| --- | --- | --- |
| **data scale** | OOD **flat** 1.5K→5.9K (+3.75/+2.97/+4.58, within noise) | in-dist rises = memorization, not generalization |
| **loss** (CE vs LK vs margin) | **identical** top-1 & top-k | top-1→top-4 gap is the head's intrinsic approximation error; no loss fixes it |
| **α k2-focus** ([.3,.7]) | no gain | k1/k2 coupled — upweighting k2 starves its own input |
| **capacity** (unfreeze experts) | deprioritized | in-dist↑/OOD-flat → would overfit, not generalize |
| **tree / multi-candidate** | **NO-GO on GB10** | headroom real (top-4=90%) but verify cost grows ~21%/node > ~12% top-2 accept gain |

**Top-k ceiling** (`topk_ceiling.py`): the head's draft-2 top-1 is 75% but **top-4 is
90%** — it *has* the right token, just mis-ranks it. That headroom is purely
inference-time. But the **tree cost curve** (extended draft-3/4 cascade verify +
`verify_ms` timing) shows the MoE expert-union verify cost grows ~21%/node — capturing
the top-2 hedge (+12% tokens/step) costs +21%, net ~−8% tps. The headroom is real but
**uneconomical on membw+MoE hardware.** Further accept gains need different hardware
economics or a non-MoE drafter — a new project, not a tuning run.

---

## Reusable tooling built (`tools/mtp/`)

- `build_dataset.py` — open-aperture, tagged, multi-source, bilingual prompt builder
- `harvest.py` — chunked/resumable/thermal-safe self-distill harvest + per-shard class tags
- `train_mtp.py` — HF Trainer, LK/CE/margin losses, stratified per-cat eval, wandb telemetry, `--smoke`
- `analyze_harvest.py` — per-cat alignment/entropy + ZH-coherence health check (caught the template bug)
- `quant_gap.py` — bf16-vs-Q4_K accept 2×2 (quant penalty + deployed gain)
- `topk_ceiling.py` — `P(base argmax ∈ head top-k)` headroom probe
- ds4: `--mtp-harvest-gen`, `encode_chat_prompt` harvest, draft-3/4 cascade verify + `verify_ms`, `DS4_CUDA_MOE_NO_ATOMIC_DOWN` deterministic A/B
- A/B: `tools/perf/mtp/baseline_run.sh` (20-cat, deterministic) + 20 category prompts

## Reproduce

```
# 1. dataset (CPU/network)
uv run --project tools/mtp python tools/mtp/build_dataset.py --out tools/mtp/prompts_v2.txt --n 6000 --zh-frac 0.15
# 2. harvest (~1 day, GB10; resumable)
uv run --project tools/mtp python tools/mtp/harvest.py --corpus tools/mtp/prompts_v2.txt --out /tmp/mtp_shards_v2 --gen 320 --chunk-size 1000 --cooldown 120
# 3. smoke (~2 min) then train (~2.8h) — defaults carry the recipe
uv run --project tools/mtp python tools/mtp/train_mtp.py --shards /tmp/mtp_shards_v2 --smoke
uv run --project tools/mtp python tools/mtp/train_mtp.py --shards /tmp/mtp_shards_v2 --run_name v2
# 4. export + A/B
uv run --project tools/mtp python tools/mtp/export_gguf.py --orig <orig MTP gguf> --ckpt tools/mtp/mtp_v2.pt --out <v2 gguf>
CTX=8000 DRAFT=2 tools/perf/mtp/baseline_run.sh --label base && CTX=8000 DRAFT=2 tools/perf/mtp/baseline_run.sh --label v2 --mtp <v2 gguf>
```

## Caveats

- OOD A/B per-category is underpowered (1 prompt/cat → ±2-4% noise; only structured-
  list & analytical individually significant). The **20-cat aggregate (±0.67%)** is the
  reliable number. Multi-prompt/cat + mean±std (eugr llama-benchy style) would tighten
  per-cat if needed.
- Greedy deploy: gains are accept-rate → tps. Sampled-decode acceptance policies
  (typical/threshold) are a separate, untested axis.
