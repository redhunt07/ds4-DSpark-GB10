"""gamut.hw — GB10 hardware constants + model shapes + metric IDs.

GB10 numbers are calibrated by tools/perf/membw.cu (the LPDDR5X read ceiling
measures ~236 GB/s, 87% of the 273 GB/s theoretical), not guessed. Override
Model per model if the shapes change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HW:
    name: str = "GB10 sm_121a"
    hbm_read_gbps: float = 236.0          # MEASURED sustained read ceiling (membw.cu)
    hbm_theoretical_gbps: float = 273.0   # LPDDR5X spec peak
    f32_tflops: float = 31.0
    f16_tc_tflops: float = 125.0
    i8_dp4a_tops: float = 250.0
    n_sms: int = 48
    # occupancy model (sm_121a): 64K 32-bit regs/SM, ~227KB smem/SM, 64 warps/SM
    regs_per_sm: int = 65536
    smem_per_sm: int = 232448
    max_warps: int = 64


@dataclass
class Model:
    """DeepSeek-V4-Flash shapes (decode). Override per model if needed."""
    n_embd: int = 4096
    n_ff_exp: int = 2048
    n_expert_used: int = 6
    bits_q8: float = 8.5      # Q8_0 incl. scales (8 + 16/32)
    bits_q2k: float = 2.6     # Q2_K incl. scales/mins
    bits_iq2: float = 2.6     # IQ2_XXS


# gb20b GPU_METRICS metricId -> short key. Confirmed against
# TARGET_INFO_GPU_METRICS on nsys 2025.3.
METRIC = {
    "gpc_clock_mhz": 0,
    "gr_active": 6,
    "sms_active": 7,
    "sm_issue": 8,
    "tensor_active": 9,
    "compute_warps": 16,
}

# Metrics surfaced in the verdict, in display order.
VERDICT_METRICS = ["sms_active", "sm_issue", "tensor_active", "compute_warps"]

# A sample counts as "busy" (real compute, not an inter-token launch gap) when
# SMs Active exceeds this.
BUSY_SMS_ACTIVE = 40.0

# Launches exactly once per decode token -> a clean per-token boundary marker
# (the engine has no NVTX to lean on).
TOKEN_MARKER = "embed_token_hc"
