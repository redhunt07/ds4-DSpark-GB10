"""Pythonic high-level wrapper for DS4 directional-steering workflows.

Sits on top of the pure ctypes binding in ``ds4.py`` and adds numpy-shaped
ergonomics for the two end-to-end flows we care about today:

* **Building a steering direction** from paired (target, control) prompts.
  See :func:`build_direction` and :func:`save_direction` / :func:`load_direction`.
* **Capturing per-layer activations** from a single prompt into an
  :class:`numpy.ndarray` of shape (n_layers, n_embd).  See
  :func:`capture_activations`.

Everything here is composable: the lower-level ``ds4`` module remains
stdlib-only and usable on its own; importing this module pulls in numpy.

Shape defaults (``N_LAYER_FLASH=43``, ``N_EMBD_FLASH=4096``) match
DeepSeek-V4-Flash; pass explicit values for other shapes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import ds4


__all__ = [
    "N_LAYER_FLASH",
    "N_EMBD_FLASH",
    "SPECIALS",
    "render_ds4_prompt",
    "capture_activations",
    "build_direction",
    "save_direction",
    "load_direction",
]


N_LAYER_FLASH = 43
N_EMBD_FLASH = 4096

SPECIALS = {
    "bos": "<｜begin▁of▁sentence｜>",
    "user": "<｜User｜>",
    "assistant": "<｜Assistant｜>",
    "think": "<think>",
    "nothink": "</think>",
}


def render_ds4_prompt(system: str | None, user: str, think: bool) -> str:
    """Render the minimal DS4 chat prefix used for activation capture.

    Matches the structure the file-based dump hooks anchor to (capture
    position 0 of this prompt). Use a system slot when you want it
    consistent across pairs; leave None to skip.
    """
    pieces = [SPECIALS["bos"]]
    if system:
        pieces.append(system)
    pieces += [
        SPECIALS["user"],
        user,
        SPECIALS["assistant"],
        SPECIALS["think"] if think else SPECIALS["nothink"],
    ]
    return "".join(pieces)


def _activations_to_ndarray(rows: Iterable, n_layers: int, n_embd: int) -> np.ndarray:
    """Stack per-layer array.array("f") rows into an (n_layers, n_embd) f32."""
    mat = np.stack([np.frombuffer(r, dtype=np.float32) for r in rows], axis=0)
    if mat.shape != (n_layers, n_embd):
        raise RuntimeError(
            f"activation shape mismatch: {mat.shape} != ({n_layers},{n_embd})"
        )
    return mat


def capture_activations(
    eng: "ds4.Engine",
    prompt_text: str,
    *,
    component: str = "ffn_out",
    n_layers: int = N_LAYER_FLASH,
    n_embd: int = N_EMBD_FLASH,
    pos: int = 0,
    ctx_size: int = 512,
    work_dir: Path | None = None,
) -> np.ndarray:
    """Run one forward pass over ``prompt_text`` and return per-layer activations.

    ``prompt_text`` is tokenized as a pre-rendered chat string (so include
    BOS / role / think markers yourself, e.g. via :func:`render_ds4_prompt`).
    Returns an (n_layers, n_embd) float32 ndarray.
    """
    tokens = eng.tokenize_rendered_chat(prompt_text)
    with eng.session(ctx_size=ctx_size) as s:
        rows = s.collect_layer_activations(
            tokens,
            component=component,
            n_layers=n_layers,
            n_embd=n_embd,
            pos=pos,
            work_dir=work_dir,
        )
    return _activations_to_ndarray(rows, n_layers, n_embd)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize; zero rows pass through unchanged."""
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return mat / safe


def build_direction(
    eng: "ds4.Engine",
    good_prompts: Sequence[str],
    bad_prompts: Sequence[str],
    *,
    component: str = "ffn_out",
    system: str | None = "You are a helpful assistant.",
    think: bool = False,
    ctx: int = 512,
    n_layers: int = N_LAYER_FLASH,
    n_embd: int = N_EMBD_FLASH,
    pair_normalize: bool = False,
    orthogonalize: bool = True,
    progress: bool = True,
) -> np.ndarray:
    """Extract a (n_layers, n_embd) f32 direction from paired prompts.

    ``good_prompts`` are the desired/target direction; ``bad_prompts`` are
    the contrast/control. Pairs are zipped 1:1, truncated to the shorter
    list.

    ``pair_normalize=True`` averages L2-normalized per-pair differences
    instead of taking the difference of means. ``orthogonalize=True``
    removes the component parallel to the control mean.

    Returns rows already L2-normalized so direct dot-product use at
    runtime gives meaningful scales.
    """
    n = min(len(good_prompts), len(bad_prompts))
    if n == 0:
        raise ValueError("need at least one prompt pair")
    good_prompts = good_prompts[:n]
    bad_prompts = bad_prompts[:n]

    good_sum = np.zeros((n_layers, n_embd), dtype=np.float64)
    bad_sum = np.zeros((n_layers, n_embd), dtype=np.float64)
    pair_sum = np.zeros((n_layers, n_embd), dtype=np.float64)

    with tempfile.TemporaryDirectory(prefix="ds4-dir-steer-") as td:
        root = Path(td)
        for i, (good, bad) in enumerate(zip(good_prompts, bad_prompts), 1):
            if progress:
                print(f"pair {i}/{n}", flush=True)
            good_rows = capture_activations(
                eng,
                render_ds4_prompt(system, good, think),
                component=component,
                n_layers=n_layers,
                n_embd=n_embd,
                ctx_size=ctx,
                work_dir=root / f"good-{i}",
            )
            bad_rows = capture_activations(
                eng,
                render_ds4_prompt(system, bad, think),
                component=component,
                n_layers=n_layers,
                n_embd=n_embd,
                ctx_size=ctx,
                work_dir=root / f"bad-{i}",
            )
            good_sum += good_rows
            bad_sum += bad_rows
            if pair_normalize:
                pair_sum += _normalize_rows(good_rows - bad_rows)

    good_mean = good_sum / n
    bad_mean = bad_sum / n
    if pair_normalize:
        directions = _normalize_rows(pair_sum / n)
    else:
        directions = _normalize_rows(good_mean - bad_mean)
    if orthogonalize:
        base = _normalize_rows(bad_mean)
        proj = np.sum(directions * base, axis=-1, keepdims=True)
        directions = _normalize_rows(directions - proj * base)
    return directions.astype(np.float32, copy=False)


def save_direction(
    directions: np.ndarray,
    json_path: Path | str,
    *,
    meta: dict | None = None,
) -> tuple[Path, Path]:
    """Write the flat f32 vectors + JSON sidecar. Returns (json_path, f32_path).

    The JSON carries shape and provenance metadata; the .f32 (sibling with
    same stem) is the binary the runtime loads via ``--dir-steering-file``.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ds4-directional-steering-v1",
        "shape": list(directions.shape),
        **(meta or {}),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    f32_path = json_path.with_suffix(".f32")
    directions.astype(np.float32, copy=False).tofile(f32_path)
    return json_path, f32_path


def load_direction(json_path: Path | str) -> tuple[np.ndarray, dict]:
    """Load a (directions, meta) pair from a previously saved direction set."""
    json_path = Path(json_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    shape = tuple(meta["shape"])
    f32_path = json_path.with_suffix(".f32")
    directions = np.fromfile(f32_path, dtype=np.float32).reshape(shape)
    return directions, meta
