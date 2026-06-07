# tools/perf/agent — profiling ds4-agent under nsys

The agent has different perf characteristics than ds4-bench: chat-template
prefill, DSML tool framing, MTP spec decode, multi-turn sessions. This
directory profiles a one-shot non-interactive agent invocation end-to-end
and aggregates the results into a single report dir.

## Quick start

```
tools/perf/agent/profile_run.sh \
    --prompt "Write a 200-word essay about lighthouses." \
    --label lighthouse-mtp
```

Output: `tools/perf/runs/agent-profile-lighthouse-mtp/`

```
report.md          headline: metrics + accept + top kernels
agent.stdout       model output
agent.stderr       full agent stderr including init lines
agent.metrics      one-line +DWARFSTAR_METRICS
agent.mtp.txt      ds4: mtp timing lines (input to parse_timing.py)
plain.nsys-rep     nsys trace (-t cuda), open in nsys-ui
plain.sqlite       exported db for gamut.py / gpu_metrics.py
top-kernels.csv    cuda_gpu_kern_sum
mtp-summary.md     accept rate + committed distribution per spec step
```

## With GPU hardware metrics

`--gamut` runs a second pass with `--gpu-metrics-set=gb20b` so SM-issue,
tensor-pipe utilisation, and roofline %peakBW are computed via the existing
`tools/perf/gamut.py`. The second pass perturbs timing and roughly doubles
total wall clock.

```
tools/perf/agent/profile_run.sh \
    --prompt-file tools/perf/mtp/prompts/code-generation.txt \
    --tokens 600 --label codegen-gb20b --gamut
```

## Caveats — what the wrapper does and doesn't capture

| concern | behaviour |
| ------- | --------- |
| **phase detection** | uses `perflib.phases()` heuristic (TOKEN_MARKER = `embed_token_hc`). Clean for one-shot prompts without tool calls. Breaks if the agent calls tools mid-session — tool-result prefill gets attributed to "decode" by the heuristic. |
| **MTP token counting** | `embed_token_hc` fires once per *spec step* under MTP, not per emitted token. `decode_tps` from agent metrics is the source of truth; kernel-derived token rates will be 1–3× off. |
| **multi-turn sessions** | wrapper only runs `--non-interactive -p TEXT` (one turn). For multi-turn, run the agent yourself under nsys and use this dir's sub-tools manually. |
| **tool calls** | DSML tool invocations show up as kernel gaps + a re-prefill burst. Phase detection sees them as "decode → idle → decode" which is wrong. Tier 2 (NVTX annotations in `ds4_agent.c`) is the proper fix. |
| **cooldown** | enabled by default via `thermal_guard.sh`; gates each nsys pass on board temp < 55°C. `--no-cooldown` skips. |

## Suggested investigations

1. **Compare same prompt with vs without MTP** to see which kernels MTP shifts time toward (small-batch matmul, `embed_tokens_hc`, etc.):
   ```
   profile_run.sh --prompt "..." --label essay-mtp
   profile_run.sh --prompt "..." --label essay-nomtp --no-mtp
   ```

2. **Profile each of the five canonical prompt classes** to see how kernel mix shifts with content:
   ```
   for cls in prose-continuation chat-essay code-generation analytical-qa structured-list; do
       profile_run.sh --prompt-file tools/perf/mtp/prompts/$cls.txt --label $cls
   done
   ```

3. **High-resolution gb20b** on a class showing low MTP accept (e.g. chat-essay) to see if the accept-misses correlate with specific kernel stall reasons:
   ```
   profile_run.sh --prompt-file tools/perf/mtp/prompts/chat-essay.txt --gamut --label chat-essay-deep
   ```
