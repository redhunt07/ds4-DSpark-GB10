# PR #1 — CUDA Correctness Gate: A Literate Primer

*Branch:* `prep/correctness-gate` (`8950efb`) on `TrevorS/ds4`, one commit on `upstream/main` (`59d9bc7`).
*Shape:* `+~555 / −1`, 9 files, **purely additive** — no inference path is modified.

This is written so you can own the PR: defend every line, answer antirez's questions,
and regenerate or extend it without me. It walks the real committed code in reading
order. It assumes you (skips C/CUDA/quant basics) and dwells only on the design
rationale and the non-obvious.

---

## 0. The thesis (why this PR exists at all)

antirez merged your GB10 work (#13), then **reverted** it; he **declined** the MTP work
(#14). Both for the same reason: *correctness he could not independently verify.* His
words after the revert: he'd be cautious merging CUDA changes "not tested for
correctness in a very deep way." Critically — our old **token-diff** gate *masked* the
RMS-0.2 divergence he caught by hand. Token-diff only compares final sampled token
streams; it integrates over everything and only trips when drift is large enough to
flip a greedy argmax. A 0.2-RMS hidden-state error that doesn't (yet) flip the top
token sails straight through.

So PR #1 is **not** a perf change. It is the **instrument**: three CUDA correctness
gates that would have caught exactly what burned him, handed over *before* we
re-submit any kernel work. Every later PR (#2 MTP, #3 GB10 HBM) then ships *with its
proof attached* (`make cuda-ppl`, `--mtp-correctness`). That's the strategic point you're
taking responsibility for: we're rebuilding trust by giving him the ruler, not another
thing to measure.

---

## 1. Topology — what's additive, what it leans on

The entire PR is new symbols. Nothing upstream is edited except *insertion points*:

| File | What's added |
|---|---|
| `ds4.h` | 1 decl: `ds4_cuda_tensor_equivalence_selftest` |
| `ds4.c` | 1 function (the selftest, ~137 lines) |
| `tests/ds4_test.c` | 6 statics (3 gates + 3 helpers) + 3 registry rows |
| `tests/test-vectors/` | 4 files: 2 corpora, 2 baselines |
| `Makefile` | `.PHONY` + 4 phony targets |
| `CONTRIBUTING.md` | doc block under *Correctness Regression Tests* |

The reason it *can* be additive: every primitive the gates call already exists upstream
— `ds4_session_create/sync/token_logprob/eval`, `metal_graph_alloc`,
`metal_graph_encode_decode_layer`, `ds4_gpu_embed_token_hc_tensor`,
`ds4_gpu_tensor_write/read`, `embed_token_f16`, `hc_from_plain_embedding`,
`layer_forward_self_one`, `rms_abs_diff`, `max_abs_diff`, and the
`g.materialize_ffn_out` graph field. We verified each is present in `upstream/main`.
The selftest is, in effect, a *productized* copy of the already-in-tree diagnostic
`metal_graph_first_token_full_test` (`ds4.c:15913`) — same alloc, same
`materialize_ffn_out`, same per-layer teacher-forced loop — turned into an asserting
gate. That sibling is your single best "this is idiomatic, not novel" defense.

---

## 2. Three gates, and why exactly three

They form a **layered argument**, weakest-but-most-precise → strongest-but-coarsest:

1. **`--cuda-tensor-equivalence`** — *per-layer, localizing.* Recomputes the forward
   one layer at a time on both CUDA and CPU-f32 and RMS-diffs each layer's output.
   Answers: *"if CUDA drifts, **which layer** first?"* This is the one that catches the
   RMS-0.2 class antirez found — and points at the kernel responsible.

2. **`--cuda-ppl`** — *one integral scalar, self-regression.* Teacher-forces a fixed
   corpus and compares avg-NLL to a committed self-baseline. Answers: *"did **my own**
   kernels change vs the last blessed build?"* It integrates every layer's drift into
   one number — the standard llama.cpp-style regression scalar.

3. **`--cpu-cuda-ppl`** — *cross-check vs ground truth.* Same scalar, but compared to a
   committed **CPU-f32** reference instead of a CUDA self-baseline. Answers: *"is CUDA
   **right at all**?"* — the claim a self-baseline structurally cannot make (a wrong
   CUDA build is perfectly self-consistent).

The progression matters: (1) tells you *where*, (2) tells you *whether it moved*, (3)
tells you *whether it was ever correct*. Token-diff sat above all three and saw none of
it.

---

## 3. `ds4_cuda_tensor_equivalence_selftest` — the core, walked

The declaration (`ds4.h`) is the contract. It returns the **number of failing layers**
(0 = pass) and fills optional diagnostics:

```c
int ds4_cuda_tensor_equivalence_selftest(ds4_session *s,
                                         double rms_tol,
                                         double max_abs_tol,
                                         double *out_worst_rms,
                                         double *out_worst_max_abs,
                                         int *out_first_fail_layer,
                                         int *out_nonfinite);
```

### 3.1 Out-params zeroed first, then the guards

```c
    if (out_worst_rms) *out_worst_rms = 0.0;
    /* ...zero all out-params... */
#ifdef DS4_NO_GPU
    /* ...requires a GPU build... */ return 1;
#else
    if (!s || !s->engine || s->engine->backend != DS4_BACKEND_CUDA) { /* ... */ return 1; }
    if (!s->checkpoint_valid || s->checkpoint.len < 1) { /* needs a synced prefix */ return 1; }
```

Out-params are initialized **before** any early return so a caller never reads garbage.
Three guard gates: GPU build, CUDA backend, and a synced prefix (we need at least one
real token to forward). Each returns `1` (= "one failure") so a misconfigured call
reads as a failed gate, not a silent pass.

### 3.2 The determinism knob — and why it's saved/restored

```c
    char prev_atomic[32];
    const char *prev_atomic_env = getenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN");
    const bool had_prev_atomic = prev_atomic_env != NULL;
    if (had_prev_atomic) { strncpy(prev_atomic, prev_atomic_env, sizeof(prev_atomic)-1);
                           prev_atomic[sizeof(prev_atomic)-1] = '\0'; }
    setenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN", "1", 1);
```

The MoE down-projection, at `n_tokens >= 128`, accumulates via float `atomicAdd`
(`use_atomic_down`). `atomicAdd` ordering is scheduling-dependent → ulp-scale variation
**run to run**, which can occasionally flip a greedy argmax. For a gate that must be
*bit-reproducible*, that's poison. `DS4_CUDA_MOE_NO_ATOMIC_DOWN=1` forces the ordered
(deterministic) reduction. (This is the same knob that made `DS4_BENCH_TOKEN_DUMP`
greedy gates deterministic — see your memory note on it.) It is **CUDA-only**; on Metal
the symbol doesn't exist and the env is a harmless no-op.

Why copy the string before `setenv`? `getenv` returns a pointer *into* `environ`;
`setenv` can reallocate `environ` and invalidate it. So we snapshot, then set. And we
restore at the end (§3.7) because this is a **public** function — a library hook that
silently mutates process-global state and leaks it into the rest of an `--all` run is a
real wart. This was a review-pass fix.

### 3.3 Buffers and the hidden-channel layout

```c
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    float *plain    = xmalloc(DS4_N_EMBD * sizeof(float)); /* plain embedding row     */
    float *cpu_cur  = xmalloc(hc_dim     * sizeof(float)); /* CPU layer input          */
    float *cpu_next = xmalloc(hc_dim     * sizeof(float)); /* CPU layer output         */
    float *gpu_hc   = xmalloc(hc_dim     * sizeof(float)); /* GPU layer output (readback)*/
```

`DS4_N_HC`/`DS4_N_EMBD`/`DS4_N_LAYER` are **runtime** shape values (`g_ds4_shape`), not
compile-time — for Flash this is the 43-layer model you saw in the run. `hc_dim` is the
per-token hidden-channel working-state width the graph materializes; the gate compares
it **element-wise** (`hc_dim` floats) between CUDA and CPU at each layer boundary. All
four buffers are `xmalloc` (which aborts on OOM, so no NULL checks needed) and freed
unconditionally at the end.

### 3.4 Graph alloc + the `materialize_ffn_out` flag

```c
    ds4_gpu_graph g;
    bool ok = metal_graph_alloc(&g, weights, &weights->layer[0]);
    if (ok) g.materialize_ffn_out = true;
```

`metal_graph_alloc` → `metal_graph_alloc_raw_cap`, whose **first statement is
`memset(g, 0, sizeof(*g))`.** That single fact makes the whole error path safe: even if
alloc fails, `g` is all-zero, so the `metal_graph_free(&g)` at the end frees NULLs
rather than walking uninitialized pointers. `materialize_ffn_out = true` tells the graph
to keep each layer's post-FFN hidden state resident so we can read it back — without it
the intermediate is consumed in place.

### 3.5 Seed the forward: embed the first token

```c
    embed_token_f16(model, weights, token, plain);                 /* CPU side */
    hc_from_plain_embedding(cpu_cur, plain, DS4_N_EMBD, DS4_N_HC); /* → hidden-channel form */
    ds4_gpu_begin_commands();
    ds4_gpu_embed_token_hc_tensor(g.cur_hc, ...token...);          /* GPU side */
    ds4_gpu_end_commands();
```

Both sides start from the *same* token (`s->checkpoint.v[0]`, the first synced token).
CPU produces `cpu_cur` in hidden-channel form; GPU produces `g.cur_hc`. From here they
march layer by layer.

### 3.6 The per-layer teacher-forced loop — the key idea

```c
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        /* Teacher-force the GPU layer input from the CPU reference. */
        ds4_gpu_tensor_write(g.cur_hc, 0, cpu_cur, hc_dim * sizeof(float));
        ds4_gpu_begin_commands();
        metal_graph_encode_decode_layer(&g, model, &weights->layer[il], il, 0,
                                         g.layer_raw_cache[il], g.raw_cap, 0, 1, token);
        ds4_gpu_tensor *tmp = g.cur_hc; g.cur_hc = g.after_ffn_hc; g.after_ffn_hc = tmp;
        ds4_gpu_end_commands();

        layer_forward_self_one(cpu_next, model, &weights->layer[il], cpu_cur, il, 0, token);
        ds4_gpu_tensor_read(g.cur_hc, 0, gpu_hc, hc_dim * sizeof(float));
```

**This is the whole design.** At the top of each layer we *overwrite* the GPU's input
with the CPU reference (`ds4_gpu_tensor_write(g.cur_hc, ... cpu_cur ...)`). That's
"teacher forcing": each layer is measured **in isolation**, fed the known-good input, so
a layer's reported error is *its own* — not error inherited from upstream layers. Without
this, a tiny L0 drift would compound through 43 layers and you'd only ever see "the end
is off," never *which* layer started it. The teacher-force is what makes `first_fail_layer`
meaningful.

The pointer swap (`cur_hc ↔ after_ffn_hc`) is just reading the materialized output into
`cur_hc` so the readback and the next iteration's write target line up. `cpu_next` is the
CPU's independent recompute of the same layer.

### 3.7 The metric — why **relative** RMS, not absolute

```c
        double refsq = 0.0;
        for (uint64_t i = 0; i < hc_dim; i++) {
            if (!isfinite(gpu_hc[i])) nonfinite++;
            refsq += (double)cpu_next[i] * (double)cpu_next[i];
        }
        const double ref_rms = sqrt(refsq / (double)hc_dim);
        const double denom   = ref_rms > 1e-9 ? ref_rms : 1.0;          /* div-by-0 guard */
        const double rel_rms = rms_abs_diff(cpu_next, gpu_hc, hc_dim) / denom;
        const double rel_mx  = max_abs_diff(cpu_next, gpu_hc, hc_dim)  / denom;
        const bool bad = (rel_rms > rms_tol) || (rel_mx > max_abs_tol);
```

The residual stream **grows in magnitude with depth** (you can see it in the run:
`ref_rms` climbs from tens at L26 to ~1300 by L41). An *absolute* RMS tolerance would
therefore be far too loose at the deep layers (masking real drift) and far too tight at
the shallow ones (false-flagging float noise). Normalizing the diff by the reference's
own RMS gives a **scale-invariant** relative drift that is roughly uniform across depth
when it's pure float noise — so one tolerance works for all 43 layers. `max_abs` is kept
as a coarse "did something blow up" guard, also normalized. The `nonfinite` counter
catches NaN/Inf in the GPU output directly (a separate failure axis).

### 3.8 Bookkeeping, the verbose gate, double-buffer swap

```c
        if (bad) { fails++; if (first_fail < 0) first_fail = (int)il; }
        if (bad || te_verbose)
            fprintf(stderr, "ds4: cuda-tensor-equivalence L%u rel_rms=%g ...", ...);
        /* end if(ok) */
        float *ctmp = cpu_cur; cpu_cur = cpu_next; cpu_next = ctmp;   /* this layer's output → next layer's input */
    }
```

`fails` counts every exceeding layer; `first_fail` records the first (the localization
payload). The per-layer line prints **only** when that layer is bad *or*
`DS4_TEST_TE_VERBOSE=1` — that's the quiet-by-default behavior from a review pass; a
clean run emits just the summary line, full per-layer detail on demand. The final swap
makes this layer's CPU output the next layer's CPU input (the CPU march; the GPU march is
teacher-forced back to it at the next top-of-loop).

### 3.9 Teardown — symmetric and leak-free

```c
    if (!ok) { /* failed mid-forward */ fails++; }
    if (nonfinite > 0) fails++;
    metal_graph_free(&g);
    free(gpu_hc); free(cpu_next); free(cpu_cur); free(plain);
    if (had_prev_atomic) setenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN", prev_atomic, 1);
    else                 unsetenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN");
    /* write out-params; return fails; */
```

Any mid-forward failure becomes a `fails++` (so the gate trips). Then free everything
(safe even on alloc failure, §3.4), restore the env to exactly the caller's prior state,
fill the diagnostics, return `fails`. There is exactly **one** exit after the `setenv`,
which is why a single restore point is sufficient.

---

## 4. The perplexity harness (`tests/ds4_test.c`)

### 4.1 `test_ppl_score` — backend-agnostic avg-NLL

```c
    /* sync a 32-token prefix as context, then teacher-force the rest: */
    for (int j = 0; ok && j < scored; j++) {
        const int i = prefix_len + j;
        ds4_session_token_logprob(session, tokens.v[i], &score);  /* logprob of the TRUE next token */
        nll -= (double)score.logprob;                              /* accumulate −log p */
        if (j+1 < scored) ds4_session_eval(session, tokens.v[i], ...); /* advance with the true token */
    }
    *out_avg_nll = nll / (double)scored;
```

Standard teacher-forced perplexity: sync 32 tokens of context, then for each subsequent
true token ask the model its log-probability, accumulate `−logprob` (natural-log NLL),
and step the session forward **with the true token** (teacher forcing, not the model's
own sample). `avg_nll = Σ(−log p)/N`; perplexity is `exp(avg_nll)`. Lower = the model is
less surprised by the real text. It's backend-agnostic — the caller hands it a CUDA or
CPU engine, which is exactly what lets gate #3 run both.

### 4.2 `test_cuda_perplexity` — self-regression with a stale-baseline tripwire

```c
    setenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN", "1", 1);            /* bit-reproducible scalar */
    test_ppl_score(engine, "tests/test-vectors/ppl-corpus.txt", 32, &avg_nll, &scored);
    /* compare to committed tests/test-vectors/ppl-baseline.txt: */
    TEST_ASSERT(scored == base_scored);                       /* corpus/tokenizer moved → regen, don't fail-as-regression */
    TEST_ASSERT(fabs(avg_nll - base_avg) < base_tol);         /* the actual regression check */
```

Two asserts, deliberately separate. `scored == base_scored` is a **tripwire**: if the
corpus or tokenizer changed, the token count moves and the baseline is *stale*, not a
model regression — so it fails loudly with a distinct cause (telling you to regenerate)
rather than masquerading as a numeric drift. Then the real check: avg-NLL within
`tol=0.03`. Regenerate the baseline after an *intentional* numeric change with
`DS4_TEST_PPL_WRITE_BASELINE=1` (`make cuda-ppl-baseline`).

### 4.3 `test_cpu_cuda_ppl` — capture vs check, and why the CPU ref is truth

```c
    if (want_write) { /* CAPTURE (slow, opt-in): score on a DS4_BACKEND_CPU engine, commit the constant */ }
    if (!have_base) { /* SKIP: no committed CPU reference → never auto-run the slow capture */ return; }
    /* CHECK (fast): score the same corpus on CUDA, compare to the committed CPU constant */
    TEST_ASSERT(fabs(cuda_avg - base_avg) < base_tol);
```

The CPU-f32 forward is the **ground-truth** scalar — it's build-independent (it doesn't
change between upstream and our kernels), so it's captured *once* on the Grace cores
(slow) and committed as a constant. Routine runs only score the fast CUDA path against
that constant. If no reference is committed, the gate **skips** — it never silently
triggers the slow capture. This is the gate that proves *CUDA == reference*, not merely
*CUDA == itself*.

---

## 5. The numbers you're standing behind

| Gate | Result | Tolerance | What it means |
|---|---|---|---|
| tensor-equivalence | worst_rms `0.0052`, worst_max `0.039` | `0.05` / `0.5` | ~10× margin over the measured GB10 float-noise floor; ~40× under the RMS-0.2 regression class |
| cuda-ppl | ppl `3.856`, Δ `0.000000` | `0.03` | self-baseline regenerated **on this branch** → bit-exact vs itself |
| cpu-cuda-ppl | CUDA `5.816` vs CPU-f32 `5.824`, Δ `0.0015` | `0.03` | the genuine CUDA-vs-CPU numeric gap, well inside tol |

Three things to internalize so you can defend them:

- **Why tensor-equivalence isn't 0.** CUDA runs the production decode path (`quality=0`,
  i.e. TF32 + WMMA); the CPU reference accumulates in plain f32/f64. TF32 has a 10-bit
  mantissa on the multiply inputs. ~0.5% relative RMS vs an f32 reference is *expected
  and healthy* — it's the hardware's documented precision, not a bug. A real kernel bug
  shows up as the 0.2-class (≈40×) jump, or as a single layer spiking while its
  neighbors stay at noise (which is why per-layer localization matters).

- **Why cuda-ppl Δ is exactly 0.000000.** The committed baseline was *regenerated on this
  branch* (a review-pass fix — it had been carried over from the `land` tree, which has
  different kernels, giving a confusing 0.0079 drift). With `NO_ATOMIC_DOWN` forcing
  determinism, the score is bit-reproducible, so a fresh build matches its own committed
  baseline exactly. If you ever see Δ≠0 here, *your kernels changed.*

- **Why cpu-cuda-ppl Δ is nonzero (and the sign is meaningless).** `0.0015` avg-NLL is
  the real TF32/atomic-vs-f32 gap integrated over 398 tokens. CUDA scoring *lower* than
  CPU is not "CUDA is better" — it's sub-ulp noise that happened to land that way. Only
  the **magnitude** matters, and it's 20× under tol. (The two corpora differ — 1185 vs
  398 tokens, ppl 3.856 vs 5.816 — simply because they're different text; the ref corpus
  is shorter and denser.)

---

## 6. Backend behavior matrix (so you can answer "does this break my Mac?")

| Build | Registry rows | Functions | Selftest |
|---|---|---|---|
| **CUDA (Linux)** | included (`#ifndef DS4_NO_GPU`) | run | runs, asserts |
| **Metal (macOS)** | included | run, **print "skipped"** via `#ifdef __APPLE__` early-return | never *called*; **compiles** |
| **CPU (`DS4_NO_GPU`)** | excluded | compiled-but-unused (`-Wno-unused-function`) | `#ifdef DS4_NO_GPU` → returns 1 |

The one subtle claim is "**the selftest compiles on Metal even though it's never called
there.**" Proof by construction: the in-tree sibling `metal_graph_first_token_full_test`
(`ds4.c:15913`) is **not** behind any `DS4_NO_GPU`/Apple guard and uses the *identical*
symbol set (`metal_graph_alloc`, `materialize_ffn_out`, `ds4_gpu_embed_token_hc_tensor`,
`metal_graph_encode_decode_layer`, `ds4_gpu_tensor_write/read`). If that compiles in the
Metal TU — and it does, it ships — ours does too. (Still worth one `make` on a Mac before
upstreaming, but it's low risk.)

---

## 7. What changed across the three review passes (the provenance)

You watched these happen; here they are in one place because a reviewer may ask "why is
the baseline what it is":

1. **Baseline cross-tree staleness** → regenerated `ppl-baseline.txt` on this branch.
   Δ went `0.0079 → 0.000000`. (The old value came from the `land` tree's kernels.)
2. **Quiet-by-default** → per-layer dump gated behind `DS4_TEST_TE_VERBOSE`; passing runs
   print only the summary.
3. **Env hygiene** → the public selftest now saves/restores `DS4_CUDA_MOE_NO_ATOMIC_DOWN`
   instead of leaking it into the rest of an `--all` run.
4. **Docs** → the three gates documented under *Correctness Regression Tests* in
   `CONTRIBUTING.md` (where upstream already documents `ds4_test` flags + `cuda-regression`).

---

## 8. Defend-it Q&A (anticipated antirez questions)

- **"Does this touch inference?"** No. `git diff --stat upstream/main..HEAD` is additive;
  the only non-test edits are a new function and a header decl. No existing function body
  changes.
- **"Why not just keep token-diff?"** Token-diff *masked* the RMS-0.2 you caught — it only
  trips when drift flips a greedy argmax. Per-layer tensor RMS catches sub-argmax drift
  *and localizes it*; the CPU cross-check anchors CUDA to f32 truth.
- **"Will it pass on my Mac / CI?"** On Metal it registers and **skips** at runtime
  (`#ifdef __APPLE__`), so `make test` stays green; it compiles via the sibling pattern.
- **"Why isn't cpu-cuda-ppl exactly 0?"** It's the TF32-vs-f32 gap (`0.0015`), 20× under
  tol; the sign is noise.
- **"Maintenance cost?"** On an intentional numeric change, regenerate baselines with
  `make cuda-ppl-baseline` / `make cpu-ppl-baseline`. tensor-equivalence needs no
  baseline file (it recomputes the CPU reference live).
- **"Corpus licensing?"** Original synthetic prose+code (a surveyor narrative with an
  embedded Welford-variance Python snippet), written for this PR — no third-party text.

---

## 9. Run it / where it lives

```sh
# from the branch checkout, with the model available:
export DS4_TEST_MODEL=/path/to/ds4flash.gguf
make cuda-ppl                          # self-regression (Δ should be ~0 on a clean build)
make cpu-cuda-ppl                      # CUDA-vs-CPU-f32 cross-check
./ds4_test --cuda-tensor-equivalence   # per-layer RMS gate (quiet summary)
DS4_TEST_TE_VERBOSE=1 ./ds4_test --cuda-tensor-equivalence   # full per-layer dump

# regenerate baselines after an intentional numeric change:
make cuda-ppl-baseline                 # rewrites tests/test-vectors/ppl-baseline.txt
make cpu-ppl-baseline                  # rewrites tests/test-vectors/ppl-baseline-cpu.txt (slow, Grace cores)
```

Tolerance knobs: `DS4_TEST_TE_RMS_TOL`, `DS4_TEST_TE_MAX_TOL` (tensor-equivalence);
corpus/baseline overrides: `DS4_TEST_PPL_CORPUS`, `DS4_TEST_PPL_BASELINE`,
`DS4_TEST_PPL_REF_CORPUS`, `DS4_TEST_PPL_CPU_BASELINE`.

Branch `prep/correctness-gate` (`8950efb`), pushed to `origin` only — **no upstream PR
is open**; opening one is your call.
