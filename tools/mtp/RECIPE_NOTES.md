# MTP retrain — recipe notes (scouted 2026-06-03)

Companion to PAPER_FINDINGS.md. Captures the FastMTP repo-level recipe + 2026 papers
+ ds4 accept-path recon, gathered while the v2 (ZH-augmented) harvest ran.

## FastMTP exact recipe (Tencent-BAC/FastMTP sft.sh + arXiv 2509.18362 Appendix A)

- **Data:** 389.4K self-distilled samples. **EN ~70% / ZH ~30%.** Domain by token:
  general 42% · math 18% · code 13% · Chinese 27%. Global MinHash dedup + quality filter.
- **Target generation:** sampled at **T=0.6, top-k 20, top-p 0.95, max-len 4096** (NOT
  greedy). Eval is greedy T=0. (We harvest GREEDY — see "why greedy is right for us".)
- **Hyperparams (code-level):** K=3, 3 epochs, **peak LR 5e-5** cosine, warmup 0.05,
  AdamW(0.9,0.95), bf16, seq 4096 (truncation `delete` not clip), **effective batch 64**
  (per-device 1 × accum 8 × 8 GPUs). Loss = β-weighted CE, β=0.6 → α=[.510,.306,.184] (K=3).
- **Frozen:** base model + lm_head; only `model.mtp_layers` trainable.

### Our deviations (and verdicts)
- LR 3e-5 vs **5e-5** → FIX to 5e-5 (we're 40% low). FREE.
- Effective batch 8 vs **64** → raise `--gradient_accumulation_steps` 8→32+ . FREE.
- K=2 vs K=3 → KEEP K=2: draft-3 is a dead end on GB10 (uninstrumented, tps pinned
  ~9.6, ≤ draft-2 best — see project_mtp-graph-capture / acceptance memory).
- Greedy harvest vs their T=0.6 → KEEP GREEDY. We do single-chain draft-2 at greedy
  deploy, so greedy harvest is ON-POLICY (visits exactly the states our drafter sees).
  Their sampling is for tree-verification coverage (VSD: accepted path = greedy path
  only 36% of time, for TREES). Not our regime.
- Seq 256 vs 4096 → data-regime gap; our ~320-tok docs train short-context recursion
  only. Longer gens match their regime but cost GPU-hours/doc. Lever, not a bug.
- Scale 5.9K vs 389K → the dominant open lever (unchanged story).

## 2026 papers (postdate PAPER_FINDINGS.md)

- **VSD (2602.05774):** path-level acceptance objective; "accepted path = greedy path
  only 36% of time" (TREE verification). Reinforces acceptance-direct loss; cautions
  against greedy harvest ONLY if we go to tree verification (we don't, for draft-2).
- **LCM / gated-LoRA (2507.11851):** Latent Consistency Matching loss + 2-layer MLP
  sampler head. Per-domain k=8 speedup: math 5.22 / code 5.35 / chat 2.52 / knowledge
  2.38 → confirms the structured>chat acceptance gap we already track. ~1M examples.
- **PIPO (2605.27255):** confidence head, BCE vs rejection-sampling accept labels, gate
  τ_c=0.95. Emits exactly 2 tokens/step (OUR draft-2 regime). The ONE accept-side lever
  that helps at GREEDY (others collapse to argmax at T=0). Orthogonal to LK — future add.
- **Draft-Verify-Improve (2510.05421):** on-policy verification-aware draft training;
  directional corroboration of acceptance-aware > marginal CE/KL.

## ds4 accept-path recon (for the relaxed-acceptance lever)

- Accept decision is HOST-side, EXACT-MATCH: `row_tops[commit] == drafts[commit]` at
  ds4.c:22522 (combined argmax path) and ds4.c:23127 (batch verify path).
- Verify-side full logits ARE available (combined path reads row_logits; batch path
  passes NULL but can capture). top-2/threshold accept is feasible at ~2-3 host sites,
  NO kernel changes. Existing knobs: DS4_MTP_MIN_MARGIN (verify-strategy gate, not
  accept), DS4_MTP_STRICT, DS4_MTP_NO_CASCADE, DS4_MTP_TIMING, DS4_MTP_CONF_LOG.
- VERDICT: relaxed/typical acceptance only bites at T>0 (at greedy it = accept argmax)
  AND is lossy (changes output). NOT a lever for greedy deploy. Deprioritized.
- The greedy-compatible accept lever is a PIPO-style confidence head (trained), not a
  pure policy tweak.

## Net deltas to apply on the next train (post v2 harvest)
1. `--learning_rate 5e-5` (was 3e-5).
2. `--gradient_accumulation_steps 32` (was 8) — toward effective batch 64.
3. Keep: K=2, greedy harvest, LK-hybrid loss, frozen experts.
4. Build first (CPU, no GPU): stratified-by-class eval split + per-category A/B prompts
   (we train 20 cats but only A/B 5; and the eval tail isn't stratified so rare cats
   like translation/zh-math get thin coverage).
