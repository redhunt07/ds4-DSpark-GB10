# SCOUT — MTP accept rate (P3)

Target: the MTP combined-forward speculative-decode path + accept logic, in
`ds4.c` (commit/accept ~`:18687`, cascade `:18590-18753`) and the n_tok=2/3/4
share-warp verify kernel in `ds4_cuda.cu:6067`.

## Gamut signal

accept ~54.5% · ~2.09 tokens/iter (drafted=2, committed ~1.09). Decode t/s
scales ~linearly with tokens/iter → an orthogonal speedup lever.

## Findings

- **Accept = pure greedy exact-match.** Verifier takes per-row argmax over the
  full vocab; a draft is accepted iff it equals the target argmax:
  `while (commit < draft_n && row_tops[commit] == drafts[commit]) commit++;`
  (`:18687`). No sampling/threshold relaxation exists. The whole spec path is
  greedy-only (gated on `temperature ≤ 0`, `ds4_cli.c:530`).
- **Cascade N=3** (`ds4_session_eval_speculative_argmax_combined`, `:18590`):
  `[first_token, drafts[0], drafts[1]]` verified in one batched forward.
  `drafts[0]` from `combined_prev_hc` (last committed HC); `drafts[1]` chained off
  drafts[0] through the MTP block's own HC (`metal_graph_eval_mtp_draft_n_from_hc`,
  `:13437`). Partial-accept rolls back to `prefix1`/`prefix2` compressor snapshots
  (`:18704-18716`). `draft_cap` **hardcoded to 2** (`:18611`).
- **MTP head = one transformer block** (`mtp.0.*`), predicts 1 token/call; depth
  is autoregressive chaining, **capped by code, not architecture**.
- **The ~54% ceiling is model-bound**: drafts[0] ≈ 0.79 accept, drafts[1] ≈ 0.58
  conditional (`docs/gb10-decode-perf.md:156`). Product ≈ the observed ~1.09
  committed/iter. Not code-fixable without retraining the MTP GGUF.
- The combined path is **not** bit-identical to plain greedy (batched-MoE
  verifier drift, commit `45ba761`) — it's default-on only in **non-strict**
  mode; `--quality`/`DS4_MTP_STRICT` uses the bit-exact canonical verifier.

## Knobs that exist

- `--mtp-draft N` — clamped [1,16] (`:17847`). **CLI/bench default 2, server
  default 1** (`ds4_server.c:11417`). Combined-forward requires `== 2` exactly
  (`:18799`).
- `--mtp-margin F` (default 3.0) — draft-confidence gate; only *reduces*
  tokens/iter; canonical path, non-strict only.
- Env: `DS4_MTP_NO_CASCADE` (force N=2), `DS4_MTP_SPEC_DISABLE`, `DS4_MTP_STRICT`,
  `DS4_MTP_TIMING` (the `drafted=/committed=` telemetry).

## Levers (ranked by tradeoff)

#### Free — tunable now
- **Fix the server `--mtp-draft` default (1 → 2).** Server decode currently runs
  with combined-forward **disabled** (precondition is `==2`); CLI/bench had it on
  (our captures used the CLI, so the gamut numbers reflect draft=2). One
  initializer, output-identical, **zero risk** — *if* the GB10 target is the
  server. Confirm which frontend production uses.

#### Code — bit-safe
- **Adaptive cascade depth N=2↔3** (designed in `gb10-decode-perf.md:143-180`):
  EWMA of p1, flip `draft_cap` at `:18611` (break-even p1*≈0.31; current ≈0.58).
  ~30–40 LOC, ds4.c only, flag-gated. Value = *robustness* on low-accept prose,
  not headline throughput. Bit-safe (both depths emit the verifier's greedy
  stream).
- **N=4 (draft_cap=3)**: partially wired (kernels compiled `<3>/<4>`,
  `spec_logits` 16 rows) but needs a `prefix3` snapshot slot + opening the
  hardcoded `draft_cap=2`/`==2` gate; ~80 LOC. Marginal payoff (drafts[2] accept
  compounds low). Do adaptive first.

#### Code — correctness-risky (strict-mode gates this)
- **Acceptance relaxation** (typical-acceptance / threshold vs exact match):
  doesn't exist; would **break byte-equivalence to plain decode** — exactly the
  invariant strict mode protects. Highest risk; non-strict-only + logprob-vector
  quality eval if ever attempted.

#### Out of scope — model
- **Draft-head quality** (the ~54% ceiling). Needs retraining/replacing the
  `--mtp` GGUF.

## Recommendation

1. Confirm the production frontend; if server, flip `--mtp-draft` default to 2
   (free). 2. Adaptive cascade depth (bit-safe robustness). 3. Leave acceptance
   relaxation alone unless a quality-budgeted experiment is explicitly wanted.
