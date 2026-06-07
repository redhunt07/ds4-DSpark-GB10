"""gamut.mtp — parse DS4_MTP_TIMING telemetry into accept-rate + per-step timing.

Lines look like:
  ds4: mtp timing micro    drafted=2 committed=2 draft=5.7 ms snapshot=0.0 ms verify=105.0 ms total=110.7 ms
  ds4: mtp timing combined drafted=2 committed=1 total=143.7 ms
The `combined` steps are steady-state speculative decode (the throughput-relevant
cost); `micro` is the warmup probe. We surface accept-rate AND the per-step ms
breakdown (combined total drives tok/s; verify is the dominant sub-cost) so the
verify-forward residual is visible alongside throughput.
"""

from __future__ import annotations

import re

_DC = re.compile(r"drafted=(\d+)\s+committed=(\d+)")
_KIND = re.compile(r"mtp timing (\w+)")


def _field(key: str, line: str) -> float | None:
    m = re.search(rf"{key}=([\d.]+)\s*ms", line)
    return float(m.group(1)) if m else None


def parse_timing(path: str) -> dict:
    drafted = committed = iters = 0
    comb_total: list[float] = []
    comb_acc = [0, 0]                     # [committed, drafted] for combined steps
    verify_ms: list[float] = []
    draft_ms: list[float] = []
    micro_total: list[float] = []
    try:
        with open(path) as f:
            for line in f:
                m = _DC.search(line)
                if not m:
                    continue
                drafted += int(m.group(1)); committed += int(m.group(2)); iters += 1
                km = _KIND.search(line)
                kind = km.group(1) if km else ""
                tot = _field("total", line)
                if kind == "combined":
                    if tot is not None:
                        comb_total.append(tot)
                    comb_acc[0] += int(m.group(2)); comb_acc[1] += int(m.group(1))
                elif kind == "micro":
                    if tot is not None:
                        micro_total.append(tot)
                    v = _field("verify", line); d = _field("draft", line)
                    if v is not None:
                        verify_ms.append(v)
                    if d is not None:
                        draft_ms.append(d)
    except OSError:
        return {}
    if iters == 0:
        return {}

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    out: dict = {
        "accept_pct": 100.0 * committed / drafted if drafted else float("nan"),
        "tokens_per_iter": 1.0 + committed / iters,
        "iters": iters,
        "drafted": drafted,
        "committed": committed,
    }
    if comb_total:
        out["combined_total_ms"] = mean(comb_total)
        out["combined_steps"] = len(comb_total)
        if comb_acc[1]:
            out["combined_accept_pct"] = 100.0 * comb_acc[0] / comb_acc[1]
            out["combined_tokens_per_iter"] = 1.0 + comb_acc[0] / len(comb_total)
    if verify_ms:
        out["verify_ms"] = mean(verify_ms)
    if draft_ms:
        out["draft_ms"] = mean(draft_ms)
    if micro_total:
        out["micro_total_ms"] = mean(micro_total)
    return out
