# Fork Release Notes

This fork is the GB10 / DGX Spark performance branch of `antirez/ds4`.
It is meant to be published as a clean GitHub fork, so the repository only
contains source, docs, and scripts. Local model downloads, virtualenvs, and
temporary test binaries are not tracked.

## What Changed Versus Upstream

- Added the CUDA backend and GB10-specific memory residency path.
- Added DSpark runtime support for the official DeepSeek V4 Flash DSpark
  carrier layout.
- Added the quantization and conversion flow for the DSpark checkpoint.
- Added speculative decode support and the deterministic / fast verify split.
- Added GB10-focused performance tooling, telemetry, and release checks.
- Added server and agent integration for the fork-specific inference paths.

## Measured Gains On GB10

These are the numbers currently documented in the fork and used as the public
reference point for the release:

| Path | Before | After | Notes |
| --- | ---: | ---: | --- |
| Plain decode | ~12.7 tok/s | ~16.9 tok/s | Fast verify path at 32k ctx |
| MTP / DSpark-style decode | ~15 tok/s | ~18.8-21.5 tok/s | Depends on verify mode and prompt class |
| Agent coding workload | ~11 tok/s | ~15-18 tok/s | Chat-formatted, tool-aware sessions |
| Long-context greedy decode | ~17.6 tok/s | ~19.7 tok/s | Sustained at large ctx |

The practical release message is simple: the fork keeps output quality stable
while turning GB10 into a much faster local inference box for code and agent
workloads.

## What Is Not In The Repo

- Hugging Face model downloads are not committed.
- `.venv` / `.venv-hf` environments are not committed.
- Generated `ds4_*` binaries are not committed.
- Temporary test artifacts are not committed.

Use `download_model.sh` to fetch the model inputs and `quantize_dspark.sh` to
rebuild the release GGUFs from the source checkpoint.

## Public Release Checklist

- Run the release QA checklist in `QA_BEFORE_RELEASES.md`.
- Capture the benchmark numbers from `docs/gb10-decode-perf.md`.
- Keep the repo clean with `git status --short` before tagging or pushing.
- Include the model name, quantization, backend, context, and flags in any
  release note or benchmark post.
