# Fork environment flags — provenance & naming convention

This GB10/Spark fork adds ~90 `getenv` flags on top of upstream `antirez/ds4`.
Both upstream and the fork use the **`DS4_`** namespace (it's antirez's project),
so this doc records which flags are *ours*, the user-facing ones, and the naming
convention for new fork flags.

## Convention for NEW fork flags

New fork-added flags use the **`DS4_SPARK_`** prefix, e.g. `DS4_SPARK_<AREA>_<NAME>`.
This makes provenance obvious and avoids colliding with a future upstream `DS4_*`
flag of the same name.

**Existing fork flags are deliberately left on the bare `DS4_` prefix.** Renaming
them would (a) churn ~90 call sites, and (b) — for flags in files shared with
upstream (`ds4.c`) — push our `getenv` lines further from upstream's, *increasing*
rebase-conflict surface. The collision risk for existing names is low and is
caught at rebase time anyway. So: prefix forward, don't rewrite history.

## User-facing fork flags (the ones worth knowing)

| flag | effect |
| --- | --- |
| `DS4_CUDA_FAST_VERIFY` | **Experimental MTP speed knob.** Route speculative verification through fast batched kernels. It can change near-tied greedy choices; use only after a model/context-specific regression gate. |
| `DS4_GRAPH_DECODE` | CUDA-graph plain-greedy decode (+~5%, bit-identical). |
| `DS4_NO_F16_PREWARM` | skip the Q8→f16 dense-weight prewarm (faster CLI launch, slower first prefill). |
| `DS4_MTP_NO_CASCADE` | force the N=2 (K=1) MTP window instead of cascaded N=3. |
| `DS4_MTP_TV` | host-side probe: log MTP speculative-sampling acceptance (`1 − TV`). |
| `DS4_LOG_MEM` | log KV/buffer memory at session open. |
| `DS4_CUDA_MOE_TILE4` | use the `[4]`-wide `moe_down` decode kernel (measured slower; debug). |

### `DS4_CUDA_FAST_VERIFY` — MTP verify speed knob

Prepend `DS4_CUDA_FAST_VERIFY=1` to any **MTP** run (bench / agent / server). It
only affects the speculative *verify* forward, so it is a no-op without `--mtp`.
It swaps the small-n verify kernels: `ordered_max_tokens` 8 → 0 (drops the
deterministic N=1 ordered-F16 MoE) and forces the indexed-heads8 batched
attention, routing the verify GEMMs to fast batched cuBLAS.

Measured ladder (GB10, IQ2XXS model, MTP-v2 head, ctx=4096, `--gen-tokens 128
--temp 0 --power 85`, managed-KV path):

| config | decode t/s | vs plain |
| --- | ---: | ---: |
| MTP off (plain decode) | 9.73 | — |
| MTP on, default (deterministic verify) | 18.70 | 1.92× |
| MTP on + `DS4_CUDA_FAST_VERIFY=1` | 21.52 | 2.21× |

Rules of thumb:
- **Do not assume greedy identity**: fast floating-point reductions can change a
  near-tied token. The deterministic path is the production correctness
  reference; fast mode is an opt-in performance experiment.
- It is **not bit-exact**: sampled fidelity can differ, and greedy can differ on
  close logits as well.
- **Never set it when running the correctness gates** (`ds4_test --mtp-correctness`
  / `--mtp-selfconsistency`). Those exist to validate the *deterministic* path and
  assert `worst_rms=0`; the flag is process-global, so it flips the test's own
  reference too and the gate fails (NaN/top1-mismatch — a harness artifact, not a
  decode bug). The shipped gates are run with the flag **off**.

Speculative decode (both greedy argmax and temp>0 sampling) is **on by default**
when an MTP model is loaded; `DS4_MTP_SPEC_DISABLE` (upstream's flag) turns all of
it off.

Upstream flags the fork's recommended GB10 setup leans on (antirez's, not ours):
`DS4_METAL_PREFILL_CHUNK`, `DS4_MTP_SPEC_DISABLE`, `DS4_MTP_STRICT`,
`DS4_MTP_TIMING`.

## Fork flag families (provenance map)

- **`DS4_CUDA_*`** (~80) — GB10/CUDA backend kernel, cache, preload, and MoE-tile
  tuning knobs. All live in `ds4_cuda.cu`, a fork-only file (upstream has no CUDA
  backend), so they carry zero rebase risk and are already namespaced by `CUDA`.
  Almost all are internal debug/tuning switches; grep `ds4_cuda.cu` for the set.
- **`DS4_GRAPH_*`** (6) — CUDA-graph decode capture: `DECODE`, `MTP_VERIFY`,
  `STABLE_EMIT`, `NO_UPDATE`, `DUMP_LOGITS`, `CAPTURE_STATS`.
- **`DS4_MTP_*`** fork additions (3) — `SAMPLE`, `TV`, `NO_CASCADE`. The rest of
  the `DS4_MTP_*` family (`SPEC_DISABLE`, `STRICT`, `TIMING`, `PROBE`,
  `FULL_LOGITS`, `CONF_LOG`, `MIN_MARGIN`, `BATCH_VERIFY`, `SPEC_LOG`,
  `EXACT_REPLAY`, `FORCE_SNAPSHOT`) is **upstream's** MTP scaffolding.
- **misc** — `DS4_LOG_MEM`, `DS4_NO_F16_PREWARM`.

To regenerate the fork-vs-upstream split:

```sh
grep -rhoE 'getenv\("DS4_[A-Z0-9_]+"\)' *.c *.cu | grep -oE 'DS4_[A-Z0-9_]+' | sort -u > /tmp/ours
for f in ds4.c ds4_cli.c ds4_server.c ds4_agent.c ds4_eval.c ds4_bench.c ds4_metal.m; do git show upstream/main:$f; done \
  | grep -hoE 'getenv\("DS4_[A-Z0-9_]+"\)' | grep -oE 'DS4_[A-Z0-9_]+' | sort -u > /tmp/upstream
comm -23 /tmp/ours /tmp/upstream   # fork-only
```
