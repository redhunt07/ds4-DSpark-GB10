"""gamut.monitor — GPU/thermal sampler thread (replaces the bash nvidia-smi loop).

Polls nvidia-smi at 1 Hz on a background thread, tagging each sample with the
current stage (cell#iter), writes gpu.csv, and computes a global + per-cell
summary (peak temp/power/clock, throttle-mask histogram). Decodes the throttle
bitmask so thermal/power-brake events are named, not just counted.
"""

from __future__ import annotations

import csv
import subprocess
import threading
from collections import defaultdict
from pathlib import Path

# clocks_throttle_reasons.active bits we care about (nvidia-smi bitmask).
THROTTLE_BITS = {
    0x0000000000000001: "gpu_idle",
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000020: "sw_thermal",
    0x0000000000000040: "hw_thermal",
    0x0000000000000080: "hw_power_brake",
}
# Bits that actually mean trouble (not benign idle/app-clock).
TROUBLE_MASK = 0x0000000000000004 | 0x8 | 0x20 | 0x40 | 0x80

_QUERY = ("timestamp,temperature.gpu,power.draw,clocks.current.sm,"
          "clocks_throttle_reasons.active,utilization.gpu")


def decode_throttle(mask_hex: str) -> list[str]:
    try:
        m = int(mask_hex, 16)
    except (ValueError, TypeError):
        return []
    return [name for bit, name in THROTTLE_BITS.items() if m & bit]


class GpuMonitor:
    """Background 1 Hz GPU sampler with stage tagging. Use as a context manager
    or start()/stop(). Thread-safe stage tagging via set_stage()."""

    def __init__(self, out_dir: str, interval: float = 1.0):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.csv_path = self.dir / "gpu.csv"
        self._stage = "idle"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []

    def set_stage(self, stage: str) -> None:
        self._stage = stage

    def _sample_once(self) -> dict | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
        except Exception:
            return None
        line = out.stdout.strip().splitlines()
        if not line:
            return None
        f = [c.strip() for c in line[0].split(",")]
        if len(f) < 6:
            return None
        def num(x):
            try:
                return float(x)
            except ValueError:
                return float("nan")
        return {"ts": f[0], "temp": num(f[1]), "power": num(f[2]), "sm": num(f[3]),
                "throttle": f[4], "util": num(f[5]), "stage": self._stage}

    def _loop(self) -> None:
        with open(self.csv_path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["ts", "temp_c", "power_w", "sm_mhz", "throttle", "util_pct", "stage"])
            while not self._stop.is_set():
                s = self._sample_once()
                if s:
                    self.samples.append(s)
                    wr.writerow([s["ts"], s["temp"], s["power"], s["sm"],
                                 s["throttle"], s["util"], s["stage"]])
                    fh.flush()
                self._stop.wait(self.interval)

    def start(self) -> "GpuMonitor":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- summary --------------------------------------------------------
    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0}
        busy = [s for s in self.samples if not _isnan(s["util"]) and s["util"] > 50]
        def agg(rows):
            if not rows:
                return {}
            sm = [r["sm"] for r in rows if not _isnan(r["sm"])]
            pw = [r["power"] for r in rows if not _isnan(r["power"])]
            tp = [r["temp"] for r in rows if not _isnan(r["temp"])]
            thr = defaultdict(int)
            for r in rows:
                for name in decode_throttle(r["throttle"]):
                    thr[name] += 1
            return {"n": len(rows),
                    "sm_mean": _mean(sm), "sm_peak": max(sm) if sm else None,
                    "power_mean": _mean(pw), "power_peak": max(pw) if pw else None,
                    "temp_mean": _mean(tp), "temp_peak": max(tp) if tp else None,
                    "throttled": dict(thr)}
        per_cell = {}
        cells = defaultdict(list)
        for s in self.samples:
            if s["stage"] != "idle":
                cells[s["stage"]].append(s)
        for cell, rows in cells.items():
            per_cell[cell] = agg([r for r in rows if not _isnan(r["util"]) and r["util"] > 50] or rows)
        return {"samples": len(self.samples), "global": agg(self.samples),
                "busy": agg(busy), "per_cell": per_cell}


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _isnan(x):
    return isinstance(x, float) and x != x
