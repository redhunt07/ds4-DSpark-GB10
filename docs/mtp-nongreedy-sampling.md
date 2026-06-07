# MTP + non-greedy sampling — feasibility research

Working notes on what it takes to make MTP speculative decode work under
**temperature sampling** (today it is greedy-only), and whether it pays off on
GB10. Includes the acceptance-rate measurement that de-risks the build.

Companion to `gb10-decode-perf.md` (which covers the greedy MTP decode path and
its bottleneck). Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8` +
`DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32` support GGUF. Hardware: GB10 / sm_121a.

## TL;DR

- **MTP spec is hardwired to greedy.** Drafts are MTP argmax, verification is
  target-argmax equality, and every caller gates the spec path behind
  `temperature <= 0.0f`. At any `temp > 0` you fall back to one-token-at-a-time
  decode and lose the entire MTP speedup.
- **Correct fix = speculative sampling** (Leviathan/Chen rejection): sample
  drafts from `q`, accept with prob `min(1, p/q)`, resample the residual
  `norm(max(0, p−q))` on reject. Distribution-preserving — output is exactly
  target sampling, just faster.
- **The gains question reduces to one ratio.** Sampling reuses the *identical*
  combined forward, so iter cost is unchanged; speedup scales only with how much
  the acceptance rate `α` drops going from greedy-match to sampling-accept.
- **Measured `α`: it barely drops.** `α(T=0.7) = 0.780` vs greedy match `0.793`
  — a ~1.5% gap. MTP's draft distribution tracks the target closely
  (`TV(p,q) ≈ 0.22`). The "CE-trained head is overconfident → α collapses" worry
  is **refuted by data**.
- **Verdict:** sampling preserves ~99% of the greedy MTP win. On GB10 that win
  is modest (1.06–1.17×, front-loaded at short context), but it is **positive,
  never net-negative**. Worth building if you want sampling to stop discarding
  MTP.

## Why it's greedy-only today

Three layers, all assuming argmax:

| Layer | Location | Greedy assumption |
|---|---|---|
| Draft | `ds4.c` `metal_graph_eval_mtp_draft*` | drafts are MTP argmax (`mtp_top`) |
| Verify | `ds4.c` spec combined/canonical | `row_tops[i] == drafts[i]` (target argmax equality) |
| Caller gate | `ds4_cli.c`, `ds4_server.c` | spec taken only when `temperature <= 0.0f` |

The public entry point is literally `ds4_session_eval_speculative_argmax` and
carries no sampler args.

## The math

### Speculative sampling acceptance

For a draft token drawn from `q`, exact speculative sampling accepts with
probability

```
α = E_{x~q}[min(1, p(x)/q(x))] = Σ_x min(p(x), q(x)) = 1 − TV(p_T, q_T)
```

Greedy verify instead uses `P(argmax p == argmax q)`. Relationship:

- `T→0`: spec → greedy (delta vs delta), `α → greedy match rate`.
- `T>0`: `α = 1−TV` is sensitive to the *whole* distribution; greedy only cares
  about the top token. If `q` were overconfident (sharper than `p`),
  `Σ min ≈ p_A` at the shared mode → `α` well below the greedy rate. **This was
  the feared failure mode.**

### Cost model — the gain collapses to an α ratio

Spec-sampling reuses the same combined N=K+1 verify forward as greedy MTP, so
the per-iter cost `C_iter` is identical. Only tokens/iter changes:

```
spec_sampling_speedup = greedy_MTP_speedup × (TPI_samp / TPI_greedy)
TPI = 1 + α₀ + α₀·α₁           (K=2 combined path)
```

So the entire question is: **how much does `α` fall under sampling?**

## Measurement

Rather than build the rejection sampler to find out, we measured `α` directly
with an instrumentation probe (`DS4_MTP_TV`, env-gated, ~70 LOC in `ds4.c`).

### Method

At each committed position the plain decode path already produces, side by side:

- `s->logits` = target `p` for the next position,
- `s->mtp_logits` = one-ahead MTP draft `q` for the same position (forced on by
  the probe).

The probe accumulates `Σ min(softmax(p/T), softmax(q/T))` at `T ∈ {0.3, 0.7,
1.0}` plus the greedy argmax-match, printing a running mean every 16 tokens.

Reproduce:

```
DS4_MTP_TV=1 ./ds4-server -m ds4flash.gguf \
  --mtp /path/to/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --warm-weights --ctx 524288 --port 8000
# then drive any temp>0 completion; grep the server log for "mtp-alpha"
```

The probe fires in the plain-eval path (`temp > 0`), measuring the one-ahead
draft `α₀`. Cost is ~3×129k-vocab softmax passes per token — strip
`DS4_MTP_TV` for production.

### Result (n=256, technical prose, top_p=0.95)

| Metric | Value | vs greedy |
|---|---:|---:|
| greedy match `P(argmax p == argmax q)` | 0.793 | — |
| α @ T=0.3 | 0.784 | 0.989× |
| α @ T=0.7 | 0.780 | 0.984× |
| α @ T=1.0 | 0.773 | 0.975× |

`α` is within ~1.5% of greedy at T=0.7 and nearly temperature-flat. `TV(p,q) ≈
0.22` — MTP `q` ≈ target `p` across the full distribution, not just the mode.
The overconfidence failure mode does not occur for this head.

### What it implies for speedup

`α` drops <2% ⇒ `TPI` drops <1% (e.g. knight-scale `2.24 → 2.22`). Folding into
the cost model against the measured greedy envelope (`speed-bench/gb10*.csv`):

| Context | greedy MTP (measured) | spec-sampling (projected) |
|---|---:|---:|
| 2k | 16.77 / 14.36 = **1.17×** | ~1.16× |
| 64k | 12.89 / 12.14 = **1.06×** | ~1.05× |

Positive at both ends, net-negative at neither.

## Caveats

- **Conservative number.** Probe measures raw temperature-scaled `1−TV` with no
  top-p/top-k truncation. Real sampling truncates `p` and `q` identically, which
  lops shared tails and pushes `α` **up** — true acceptance ≥ 0.78.
- **α₀ only.** This is the one-ahead draft. The chained second draft `α₁` needs a
  combined-path probe; calibration is a property of the head, so it should scale
  by the same ~0.98 factor.
- **Absolute size varies, ratio holds.** Predictable prose sits high
  (`α₀≈0.79`). Creative/chat text lowers *both* greedy and sampling `α` in
  absolute terms (so the absolute speedup decays toward 1.0× faster with
  context), but the sampling-vs-greedy **ratio stays ~0.98**. So "sampling ≈
  greedy" holds across workloads even though the size of the win does not.

## Implementation scope (if greenlit)

Fork the combined N=K+1 path (`ds4_session_eval_speculative_argmax_combined`)
into a sampling variant. The target per-row logits are already captured in
`s->spec_row_logits_buf`; the gaps are on the draft and acceptance sides.

- **Files:** `ds4.c` (~+250/−30 — new `_sample` fn, sampler refactor @
  `sample_top_p_min_p`, per-row draft-logits sink @ `metal_graph_eval_mtp_draft_n_from_hc`),
  `ds4.h` (+3, additive prototype), `ds4_cli.c` (~2 call sites), `ds4_server.c`
  (1 call site).
- **Named units:** 1 new spec fn; 1 sampler refactor (expose the chosen token's
  filtered prob + the renormalized vector for the residual draw); 1 draft
  primitive change (per-row `q` sink instead of last-row only); 3 call-site
  re-routes (drop the `temp<=0` gate, route by `temp>0 → _sample`).
- **Gaps to close:** per-row draft logits `q` (primitive sinks only the last
  row today); drafts must be *sampled* from `q` not argmaxed (host RNG per draft
  row → one device sync per row, negligible vs the forward); residual `max(0,
  p−q)` must use the *same* (temp, top_p, top_k, min_p)-filtered supports as the
  caller's sampler; disable the `mtp_margin` confidence-skip on the sample path
  initially.
- **Verification:** new test — seeded sampled-spec stream matches plain sampling
  in *distribution* (χ² over N draws), existing greedy spec regression
  unaffected, GB10 tokens/iter perf run at temp 0.7.
- **Risk:** public API yes (additive) · data migration no · cross-module no
  (engine-local + thin caller edits) · reversible yes · external blocker no.

## Open questions

- Combined-path `α₁` (chained second draft) under sampling — measure with a
  combined-path probe before committing to K=2 sampling.
- Truncated-`α` (top-p/top-k applied) — expected ≥ raw; quantify to tighten the
  speedup projection.
- Does `α` hold on creative/chat workloads at the same ~0.98 ratio? Bracket the
  low end with a second probe run.
