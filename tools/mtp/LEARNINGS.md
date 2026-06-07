# FastMTP head-retrain — learnings & next-attempt playbook

Consolidated 2026-06-02 across the build → fixed-data → quant-aware arc. This is
the rubric to clear before the *next* training attempt so we don't re-pay any
lesson below. See also memory `project_fastmtp-derisk`.

---

## 0. The goal, restated precisely

Raise **draft-2** speculative-decode acceptance of the V4-Flash MTP head on GB10
by retraining only the ~167M **conditioning path** (attn, e_proj/h_proj, norms,
2 HyperConnections, hc_head, MoE router gate) while the 256 routed **experts stay
frozen at Q4_K** (kernel-locked). The metric that pays is **k2** — the draft-2
step-1 conditional (22–60% by class); k1 (draft-0) is already healthy (73–88%).

Non-negotiable framing learned the hard way: **the deliverable is real ds4
draft-2 accept-by-class, not a training proxy.**

---

## 1. What is BUILT and validated (reuse as-is)

❨`✓`❩ **Harvest** — prefill base-HC dump (`DS4_MTP_HC_DUMP`, `ds4 --mtp-harvest`) →
`.npz {tokens int32[N], hc f32[N,16384]}`. `/tmp/mtp_shards_pol` = 4826 shards,
1.16M positions. Source-agnostic: the trainer doesn't care how shards were made.
❨`✓`❩ **mtp_model** — trainable `DeepseekV4MtpHead`, `from_pt` warm-start, freeze
experts, grad-ckpt (needs BOTH the flag AND `_gradient_checkpointing_func`).
❨`✓`❩ **train_mtp** — `MtpTrainer(transformers.Trainer)`. Override only:
`compute_loss` (K-step recursive unroll, weighted CE), `evaluation_loop` (per-k
accept proxy + per-class + leak guard), `log` (per-k loss), `_save/_load`
(slim 335MB `trainable.bin`, strict=False reload). `ThermalCallback`.
❨`✓`❩ **export_gguf** — trained `.pt` → standalone `mtp.0.*` gguf (28 patched + 4
verbatim).
❨`✓`❩ **Quant-aware** (`--quant-aware`) — `build_q4k_experts` (dequant deployed
Q4_K), `_q8k_ste` (Q8_K STE), `_qa_linear` (fp32/TF32-off), `enable_quant_aware`.
❨`✓`❩ **Probes (regression tools)** — `probe_hc` (HC causality), `probe_mask`
(mask leak), `probe_q4k` (the 7% quant gap + orientation).
❨`✓`❩ **A/B harness** — `baseline_run.sh`, parametrized `--mtp`/`--draft`/`--ctx`
(env `MTP`/`DRAFT`/`CTX`).

---

## 2. Bugs found — root causes and the general lesson each taught

| # | bug | root cause | general lesson |
|---|-----|-----------|----------------|
| 1 | flat accept=1.0 collapse | `create_sliding_window_causal_mask` (DynamicCache + `position_ids=i+k`) malformed for the synthetic batched-position unroll → forward leak training over-optimizes | **a leak is LEARNED over ~50 steps; short smokes (18 steps) look healthy while climbing. Flat-across-depth (k1=k2=k3=k4) is the leak signature — real accept DECAYS with depth.** Now flagged by the `SUSPICIOUS` metric. |
| 2 | export_gguf quant crash | passed `raw_shape=gguf_shape` (element) where `add_tensor` derives element shape from the PACKED bytes | match the library's contract, don't assume |
| 3 | export_gguf unloadable | `orient()` transpose backwards — the writer stores numpy shape REVERSED (written S reads back S[::-1], bytes unchanged), so it double-reversed the 12 non-square tensors | **validate the FULL write→load→generate path, not just shapes.** ds4 caught it (`hc_head_fn dim[0]=4`); shape-equality checks passed. |
| 4 | export "validated" but never run | EXPORT_MAP was meta-init validated, never a real write → 2 bugs on first execution | **meta/shape validation ≠ real execution. Run the artifact end-to-end before trusting it.** |

---

## 3. Measurement methodology — the meta-lessons (most expensive category)

These cost the most time and are the easiest to repeat. **Clear every one before
declaring an attempt positive or negative.**

1. **Proxy ≠ real accept.** Training on corpus-token agreement optimizes the wrong
   target: acceptance scores the head against the *model's own sampled
   distribution*, not corpus text. This is THE central failure of fixed-data.
2. **Eval must run through deployment numerics.** The accept proxy is only
   meaningful if it flows through the *quantized* experts (fixed in quant-aware).
   A proxy on bf16 experts measures a model that doesn't deploy.
3. **Train depth must match A/B depth.** We trained K=4 but A/B'd at draft-2,
   which only fires k1+k2 — so the k3/k4 gains were structurally invisible.
4. **Verify-bound regime hides accept.** At long ctx (100k) decode is
   verify-bound (~99% of the spec-iter) → **tps is insensitive to accept**. Use
   short ctx OR measure accept-rate directly, never tps-at-long-ctx.
5. **draft-3+ is unmeasurable AND uncompetitive on GB10.** `DS4_MTP_TIMING` emits
   only on the draft≤2 path (no accept lines at draft≥3); tps pinned ~9.6 at both
   8k and 100k ctx (intrinsic overhead, not verify-bound) and ≤ draft-2's best.
   **Draft-2 is the only path that is both instrumented and accept-sensitive.**
6. **Sampling noise needs matched denominators.** Per-class accept/tps is
   non-comparable when token counts diverge (analytical 82 vs 471 tok). **Use
   greedy (temp=0) for a clean A/B**, or fix the token count.
7. **You need an untrained-baseline reference for the proxy.** First eval is 100
   steps in; without the step-0 (original-head-through-quant) number you can't
   read the *lift*, only the level.

---

## 4. Experimental results so far

| attempt | data | depth | numerics | proxy | real ds4 draft-2 A/B |
|---------|------|-------|----------|-------|----------------------|
| fixed-data K=4 | corpus (ShareGPT+UltraChat) | K=4 | bf16 experts | chain 0.509→0.528 (plateau), k4 +0.025 | **FLAT/neg** (chat-essay −4.5, structured −0.6, code 0) → NEGATIVE |
| quant-aware K=2 | corpus | K=2 β=1.0 | Q4_K+Q8_K STE | chain 0.564→0.574 (k2 climbing, flattening) | *pending* |

Fixed-data verdict: triply confounded (wrong distribution, K=4 diluted k2 to 28%,
class-imbalanced) → not a clean test of the lever, but a clean negative on
*corpus + K=4* specifically.

---

## 5. Why headroom might exist (theory) — and the honest caveat

- **DeepSeek never optimized MTP for this task.** It's a pretraining auxiliary
  loss (depth-1, densifies gradient for the main model). Recursive inference
  drafting was never an optimization target → k2 collapse is *untrained*, not
  *failed* capability. We run an optimization they never ran.
- **Generalist→specialist.** Their head averages 32T tokens; we specialize to our
  deployment slice (GB10, our content mix, our quant).
- **Lever is conditioning (167M), not knowledge (frozen experts).** The deficit's
  class-dependence (structured 22.5% vs code 73%) is a conditioning/distribution
  fingerprint — uniform quant or a knowledge limit would hit all classes equally.
- **Quant-aware is the strongest "why us": they don't ship Q4_K**, so they never
  realigned for it. The gap is real and measured: **gate_up 7.1% / down 7.2%**
  rel-err (per-expert 5.3–5.9% spread → router can route around bad experts).
- **HONEST CAVEAT:** none of this is proven. Every positive theory must survive
  the draft-2 A/B. Pre-commit kill criteria (§8).

---

## 6. Quant-aware / QAT recipe (from the literature scout)

It's **QLoRA-style "fixed-external-quantizer error-feedback", not classic QAT**
(quantized weights are frozen; we train *other* fp params to absorb their error).

- **STE on activations ONLY** — experts frozen ⇒ weights are a dequant constant,
  no weight STE. `x + (q(x)−x).detach()`, scale `d=amax/127` **detached** (kernel
  amax is non-diff). Because `amax` sets the max element exactly at the range edge,
  the gradient range-clip never fires (simpler than PACT/LSQ static-scale methods).
- **No LSQ/PACT scale-learning** — our scales are kernel-fixed; learning them
  un-matches deployment. Keep only LSQ's `g=1/√(N·Qmax)` as an LR sanity check.
- **Router UNconstrained** — let it learn to down-weight high-quant-error experts
  (emergent, free). Do NOT add EAQuant routing-consistency/KL (fights adaptation).
- **argmax routing ⊥ quant STE** — compose cleanly, no Gumbel.
- **fp32, never TF32/bf16** in the fake-quant matmul. TF32 ≈ 1e-3, bf16 ≈ 4e-3 —
  both comparable to the Q4_K/Q8_K error we're capturing and would corrupt the
  target. fp32/TF32-off residual ≈ 1e-6 (cross-block reduction order only;
  within-256-block int8 dots are fp32-exact below 2²⁴). Bit-exact would need a
  block-ordered manual sum — not worth it (below bf16 grad noise floor).
- **LR/stability:** lower LR than vanilla FT + warmup; optional fp-warmup→quant
  schedule; watch router oscillation (EMA/lower router LR if it flip-flops). No BN
  gotcha (RMSNorm). RMSNorm gains + projections are the natural sink for the
  systematic bias-shift (bias-correction lit).

---

## 7. Data learnings — the corpus is the wrong KIND, not too small

- **Size is plenty:** 1.16M positions ≈ 3× FastMTP; the proxy *plateaus* by step
  ~400 ⇒ information-saturated, not size-limited. More of the same won't help.
- **Wrong distribution:** corpus text ≠ model's own sampled outputs (what accept
  scores). Fix = **self-distillation** (ds4 generates, harvest its own tokens +
  base-HC). Decode-path HC dump ≈ 20 LOC mirror of the prefill hook (scouted).
- **Class imbalance:** `build_corpus.py` pulled generic chat → underweights
  structured-list/code (our hard targets). Fix = **class-balanced seed prompts,
  structured-list/chat-essay heavy**.

---

## 7b. Hyperparameters — an UNMEASURED dimension (be honest)

We changed *big* things one at a time (data source, K-depth, quant) but **never
swept hyperparameters**, so we don't actually know HP sensitivity. What the
evidence says vs what's untested:

- **Evidence leans data-ceiling, not HP-ceiling.** Both runs *plateaued by step
  ~400/543* with **healthy optimization** — grad_norm stable ~2–3, loss decreasing
  smoothly, no instability. That is not the signature of wrong-LR / undertraining
  (which keeps climbing at the end); it's "extracted what this data+objective
  offers." Both plateaus land ~0.52–0.58 chain across different K/β/lr ⇒ suggests
  a data ceiling, not an optimization one. **You usually can't tune your way out
  of wrong data.**
- **But it's not proven, and these levers are untested:**
  - `β` / k2-weighting — pushed 0.6→1.0; could go k2-heavier (α≈[0.3,0.7]). Cheap.
  - **Router LR** — the key quant-aware adaptation knob (route around bad experts)
    currently runs at the uniform LR. A higher *router-only* LR is the most
    theory-motivated untested HP. Cheap.
  - epochs / LR schedule / warmup — minor; the plateau predicts diminishing.
  - **Unfreeze experts under QAT** — the biggest untested structural "HP". Memory
    said "unfreeze only if step-1 stalls" — k2 HAS stalled. Under quant-aware STE
    the experts could realign too (6.4B params, expensive, breaks the
    conditioning-only thesis — a separate experiment, not a sweep knob).
- **The disciplined move:** a **cheap proxy-screened HP sweep** on the current
  quant-aware + existing-shards setup BEFORE paying for self-distill. The proxy now
  runs through the quantized experts, so it's a fair *relative* screen even though
  it's not the absolute deliverable. Sweep e.g. β∈{1.0,1.5}, lr∈{3e-5,1e-4},
  router-lr-mult∈{1,5}, epochs∈{1,2}; rank by proxy; **A/B only the winner**. This
  disambiguates **data-ceiling vs HP-ceiling** for a fraction of a self-distill
  harvest. If the proxy is HP-insensitive (all plateau ~same) → confirmed data
  ceiling → commit to self-distill. If a config breaks out → cheaper win.

## 8. Decision tree & pre-committed kill criteria

- **Quant-aware K=2 draft-2 A/B moves structured-list/chat-essay accept** →
  realignment is real; stack self-distill + class-balance for more.
- **Flat with full quant error in the loop** → the gap **isn't** quant. Stop the
  quant line. Either (a) self-distill (distribution fix) or (b) shelve the lever.
- Don't rerun the same confounded experiment hoping for a different number.

---

## 8b. 2026 literature (searched 2026-06-02) — reorients the plan

**Headline: FastMTP IS self-distillation; we ran the wrong data.** FastMTP
generates training responses FROM THE MODEL ITSELF (self-gen, frozen offline),
not corpus. Our "fixed-data" run used corpus = the bug. FastMTP's real recipe:
step-2 accept 11%→56%, step-3 2%→36%. The whole 2026 literature converges: corpus
joint ≠ model's joint, and acceptance scores the model's joint. Our negative
result is PREDICTED by every paper — a data bug, not a method dead-end.

New actionable levers (all training-time, no kernel change, frozen-expert OK):
- **Self-distillation is NECESSARY** (not the "optional final run" our memory said).
  On-policy GREEDY self-gen (argmax rollouts, base frozen) is the established
  recipe — no RL sampling needed. `MTP-D` [2603.23911](https://arxiv.org/abs/2603.23911)
  (Mar 2026) = our EXACT config (frozen base, head-only self-distill, +7.5% head
  accept; "looped extension" = depth-1 head reapplied recursively, +220% over
  1-head). Read its full PDF for the per-step weighting.
- **LK Losses** [2602.23881](https://arxiv.org/abs/2602.23881) (Feb 2026, Nebius) —
  acceptance-DIRECT loss (max α=1−TV(p,q)) instead of CE/KL. Proves KL & accept
  diverge UNDER CAPACITY CONSTRAINTS = our frozen-167M regime; gains largest for
  small-draft→large-MoE-target (DeepSeek-V3 +5.6%), tested on DeepSeek-MTP heads.
  Drop-in, zero overhead. **Better objective than β-weighted recursive CE.**
- **Attention Drift / "EAGLE 3.1"** [2605.09992](https://arxiv.org/abs/2605.09992)
  (May 2026) — KNOWN k2 fix: unnormalized inter-step residual grows hidden
  magnitude with depth → drafter acts like stacked layers, collapses past training
  horizon. Fix = post-norm + per-hidden-state RMSNorm on the conditioning path
  (depth-4 train generalizes to depth-20+). CAVEAT: MTP heads already use post-norm
  and STILL drift → necessary-but-NOT-sufficient for MTP; pair with self-distill.
- **Quant-aware is MINOR, de-prioritize as a standalone lever.** ML-SpecQD
  [2503.13565](https://arxiv.org/abs/2503.13565): a self-aligned quantized drafter
  keeps 71–91% accept with NO finetune ("intrinsic alignment" — drafter shares the
  target's quantized weights). So DON'T run quant-aware as a headline; instead
  **generate the self-distill targets THROUGH the deployed Q4_K/Q8_K path** — folds
  quant into the data for free (our `enable_quant_aware` becomes the harvest fwd).
- **EAGLE-3 multi-layer feature fusion** [2503.01840](https://arxiv.org/abs/2503.01840)
  — transferable arch idea (fuse early/mid/late frozen-block features for better
  deep-step conditioning), but a real architecture change → propose, don't ship.
- Infra option: `SpecForge` [2603.18567](https://arxiv.org/abs/2603.18567) supports
  EAGLE-3 + MTP training (online/offline) if we ever want a maintained trainer.

## 9. THE NEXT ATTEMPT — the rubric to clear

Combine every learning into one clean experiment. Priority reordered by the 2026
literature (§8b): self-distillation is the #1 lever, quant folds into the data.
An attempt is **valid** only if it clears all six:

1. **Right data (THE lever)** — **self-distilled**: ds4 GREEDILY generates responses
   to a class-balanced (structured-list/chat-essay heavy) prompt set, harvested
   **through the deployed Q4_K/Q8_K path** so quant folds in for free. *Needs the
   decode-path HC dump (~20 LOC mirror of the prefill hook).* This is the FastMTP
   recipe we skipped — corpus was the bug.
2. **Right objective** — **LK acceptance-direct loss** (max α=1−TV) instead of
   β-weighted CE; wins most in our small-draft→large-MoE regime. Drop-in.
3. **Right focus** — K=2, k2-weighted. Draft-2 only.
4. **Right architecture tweak** — **Attention-Drift fix**: post-norm + per-stream
   RMSNorm on the conditioning path (k2 fix; pair with self-distill, not alone).
5. **Right measurement** — draft-2 A/B in real ds4; **accept-rate** (not tps),
   **greedy/temp=0** for matched denominators, **vs the original head**, and
   record the **untrained proxy baseline** for lift context.
6. **Pre-committed kill criterion** stated before the run.

Sequence (cheapest disambiguation first): finish the current quant-aware K=2 run +
its draft-2 A/B (clean quant-vs-no-quant datapoint; expect SMALL per ML-SpecQD) →
then the self-distill + LK + norm-fix attempt (the real shot). An optional cheap
HP sweep (§7b) can run on existing shards in parallel to rule out an HP ceiling.

Lower-priority cleanups that would de-risk measurement:
- Fix `DS4_MTP_TIMING` to emit accept lines on the draft>2 path (only if we ever
  revisit deeper drafts — currently draft-2 is the path).
- Greedy mode in `baseline_run.sh` for deterministic A/B.
- Compute the original-head-through-quant proxy as the standing reference.

Environment gotchas: `gsed` NOT installed (use sed/grep). Wandb writes to repo
root (gitignored). draft-3 code-gen A/B can drop stray files at repo root (e.g.
`anagram_groups.py`) — rm before commit. GPU sometimes busy with vLLM-Omni.
