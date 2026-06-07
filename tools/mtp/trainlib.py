"""Training harness helpers for the FastMTP fine-tune: seeding, GPU/thermal
monitoring (GB10 hard-off survival), and a non-finite guard. The loop, schedule,
logging, and checkpointing are owned by transformers.Trainer (see train_mtp.py);
this module is just the GB10-specific bits the framework doesn't provide.
"""

import random
import subprocess
import time

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_stats() -> dict:
    """nvidia-smi temp/util/power (memory is 'Not Supported' on GB10)."""
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            .stdout.strip()
            .split(",")
        )
        return {
            "gpu_temp": float(out[0]),
            "gpu_util": float(out[1]),
            "gpu_power": float(out[2]),
        }
    except Exception:
        return {}


class ThermalGuard:
    """GB10 long-run soak can hard-power-off the box. Before a step, if temp is
    over max_c, block (logging) until it cools to cool_c. Returns paused seconds."""

    def __init__(self, max_c=84.0, cool_c=70.0, poll_s=5.0):
        self.max_c, self.cool_c, self.poll_s = max_c, cool_c, poll_s

    def maybe_cooldown(self) -> float:
        t = gpu_stats().get("gpu_temp")
        if t is None or t < self.max_c:
            return 0.0
        t0 = time.time()
        print(
            f"\n[thermal] {t:.0f}C >= {self.max_c:.0f}C — cooling to {self.cool_c:.0f}C ...",
            flush=True,
        )
        while True:
            time.sleep(self.poll_s)
            t = gpu_stats().get("gpu_temp", 0.0)
            if t <= self.cool_c:
                break
        dt = time.time() - t0
        print(f"[thermal] cooled to {t:.0f}C in {dt:.0f}s", flush=True)
        return dt

