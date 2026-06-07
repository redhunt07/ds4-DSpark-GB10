# Python embedding

Three layers:

- **`ds4.py`** — pure ctypes binding to the full `ds4.h` public API. Stdlib-only; loads `libds4.so` and exposes `Engine` / `Session` / `ChatBuilder` directly over the C calls. Use this when you want the lowest-overhead in-process control.
- **`ds4_steering.py`** — Pythonic SDK on top, numpy-shaped. Adds `capture_activations`, `build_direction`, `save_direction`, `load_direction`, and the chat-prompt helpers used by the direction-steering workflow.
- **Scripts** (`dir-steering/tools/build_direction.py`, `run_sweep.py`) — thin CLIs over the SDK.

No server, no HTTP, shared address space with direct access to logits, sampling, steering, and KV state. Use this when you want in-process control. For plain chat, the OpenAI/Anthropic routes on `ds4-server` are the easier call; this path pays off for logit access, custom decode loops, steering experiments, and KV snapshot reuse.

The binding lives in `python/`; the engine sources and `Makefile` are in the repo root. Numpy (used by `ds4_steering.py` and the scripts) is declared in the repo-root `pyproject.toml` — run with `uv run python ...` or activate `.venv` after `uv sync`.

## Build

From the repo root:

```
make libds4.so
```

Produces a CUDA-linked shared lib from `ds4.c` + `ds4_cuda.cu` (compiled `-fPIC`),
written to the repo root alongside the other binaries. `ds4.py` locates it
automatically (checks `python/`, the repo root, then the cwd); override with
`DS4_LIB=/path/to/libds4.so`.

## Run

`ds4.py` is a single module — put `python/` on the path, e.g.:

```
PYTHONPATH=python python3 -c "from ds4 import Engine; print(Engine().complete('hi'))"
```

or run the built-in demo directly:

```
python3 python/ds4.py
```

## Constraint: one engine per box

The model is ~81 GB; GB10 has 128 GB unified. You cannot open an engine while
`ds4-server` is running — both the 81 GB weights and the single-instance lock
(`/tmp/ds4.lock`, override `DS4_LOCK_FILE`) conflict. Stop the server first.

## Quick start

```python
from ds4 import Engine

eng = Engine("ds4flash.gguf")                  # backend=CUDA by default
print(eng.complete("Explain MoE routing in one sentence.", max_tokens=80))

for piece in eng.generate("Write a haiku about GB10.", temperature=0.7):
    print(piece, end="", flush=True)           # streaming
```

`generate`/`complete` knobs: `system`, `max_tokens`, `temperature` (≤0 = greedy),
`top_p`, `min_p`, `think` (`THINK_NONE|HIGH|MAX`), `seed`, `ctx_size`.

## Manual session control

The decode loop, opened up — sync a prompt, then sample/eval yourself with full
logit visibility:

```python
from ds4 import Engine, THINK_HIGH

eng = Engine("ds4flash.gguf")
with eng.session(ctx_size=8192) as s:
    s.sync(eng.encode_chat("Name three primes.", think=THINK_HIGH))
    for _ in range(64):
        print(s.top_logprobs(5))               # [(id, logit, logprob), ...]
        tok = s.sample(temperature=0.0)        # or s.argmax()
        if tok == eng.eos:
            break
        s.eval(tok)
        print(eng.token_text(tok), end="", flush=True)
    print("\npos:", s.pos, "of", s.ctx)
```

Raw vocab logits and a specific token's logprob:

```python
logits = s.logits()                  # array('f'), len == eng.vocab_size
print(logits[eng.eos])
print(s.token_logprob(eng.assistant_token))   # (id, logit, logprob) or None
```

## Building chat prompts piece by piece

`encode_chat` is the one-shot path; `ChatBuilder` exposes the primitives for
multi-turn prompts:

```python
cb = eng.chat_builder()
cb.message("user", "Hi").message("assistant", "Hello!").message("user", "2+2?")
cb.assistant_prefix()                 # or .max_effort_prefix()
ids = cb.tokens()
s.sync(ids)
```

## MTP speculative decode

Greedy decode with the draft model accepting multiple tokens per step (mirrors
the CLI's fast path):

```python
tok = s.sample(temperature=0.0)
if eng.has_mtp and tok != eng.eos:
    accepted = s.eval_speculative_argmax(tok, max_tokens=64, eos_token=eng.eos)
    print(eng.detokenize(accepted))
```

## Directional steering

Low level — load a profile and set attn/ffn scales at runtime (scales are read fresh each forward; 0/0 is bit-identical to no steering):

```python
s.steering_select("refusal", "/tmp/steer/refusal.bin", attn=0.0, ffn=2.0)
print(s.get_steering())               # (attn, ffn, loaded)
s.set_steering_scale(0.0, 0.0)        # disable without unloading
```

Or scope a scale change to a block via the context manager (auto-restores on exit) — ergonomic for sweeps:

```python
with s.steering(attn=0.0, ffn=2.0):
    for _ in range(64):
        s.eval(s.sample(temperature=0.0))
# scale is back to whatever it was before the `with`
```

High level — `ds4_steering.py` covers the whole "build a direction from paired prompts" workflow without leaving Python:

```python
import ds4, ds4_steering as steering

eng = ds4.Engine("ds4flash.gguf")
good = ["...desired-style prompt 1...", "...prompt 2..."]
bad  = ["...contrast-style prompt 1...", "...prompt 2..."]

directions = steering.build_direction(
    eng, good, bad,
    component="ffn_out",       # or "attn_out"
    system="You are a helpful assistant.",
    think=False,
    ctx=512,
    pair_normalize=False,
    orthogonalize=True,
)
# (n_layers, n_embd) float32 ndarray — single model load for the whole sweep.

steering.save_direction(directions, "out/direction.json",
                        meta={"component": "ffn_out", "label": "refusal"})

# load later:
directions, meta = steering.load_direction("out/direction.json")
```

Or just capture one prompt's per-layer activations (used internally by `build_direction`, exposed for ad-hoc probing):

```python
prompt = steering.render_ds4_prompt("You are helpful.", "What is 2+2?", think=False)
acts = steering.capture_activations(eng, prompt, component="ffn_out")
# acts.shape == (43, 4096), float32
```

Under the hood `capture_activations` calls `Session.collect_layer_activations(...)` (the low-level API) which wraps the file-based `DS4_METAL_GRAPH_DUMP_*` hooks in `ds4.c`: env vars are set around a `sync()` call, per-layer .bin files are read back, and the env is restored on exit.

## KV persistence

Two routes — in-memory bytes (snapshot) or a file (payload):

```python
blob = s.save_snapshot()              # bytes
# ... later, on a fresh session at the same ctx_size:
s2 = eng.session(ctx_size=8192)
s2.load_snapshot(blob)
print(s2.pos)                         # restored timeline

s.save_payload("/tmp/sess.kv")        # file
s2.load_payload("/tmp/sess.kv")
```

## Engine-free helpers (no model load)

These run without opening the engine — safe to call while a server holds the GPU:

```python
import ds4
ds4.backend_name(ds4.BACKEND_CUDA)            # 'cuda'
ds4.think_max_min_context()                   # 393216
ds4.context_memory_estimate(ds4.BACKEND_CUDA, 524288)
# {'total_bytes':..., 'raw_bytes':..., 'prefill_cap':..., ...}
```

## API map

| Python | C (`ds4.h`) |
| --- | --- |
| `Engine(...)` / `.close()` | `ds4_engine_open` / `ds4_engine_close` |
| `.tokenize` `.encode_chat` `.detokenize` | `ds4_tokenize_text` `ds4_encode_chat_prompt` `ds4_token_text` |
| `ChatBuilder` | `ds4_chat_begin` / `ds4_chat_append_*` |
| `.generate_argmax` | `ds4_engine_generate_argmax` |
| `Session(...)` / `.sync` `.eval` | `ds4_session_create` / `ds4_session_sync` / `ds4_session_eval` |
| `.sample` `.argmax` `.argmax_excluding` | `ds4_session_sample` / `ds4_session_argmax*` |
| `.top_logprobs` `.token_logprob` `.logits` | `ds4_session_top_logprobs` / `_token_logprob` / `_copy_logits` |
| `.eval_speculative_argmax` | `ds4_session_eval_speculative_argmax` |
| `.steering_select` `.set_steering_scale` `.get_steering` | `ds4_session_steering_select` / `_set_steering_scale` / `_get_steering` |
| `.save_snapshot` `.load_snapshot` | `ds4_session_save_snapshot` / `_load_snapshot` |
| `.save_payload` `.load_payload` | `ds4_session_save_payload` / `_load_payload` |
| `.common_prefix` `.rewrite_from_common` | `ds4_session_common_prefix` / `_rewrite_from_common` |
| `context_memory_estimate` `think_*` `backend_name` | `ds4_context_memory_estimate` / `ds4_think_*` / `ds4_backend_name` |

Every public `ds4.h` symbol (78 total) is bound; this table is the common subset.
See the `ds4.py` module docstring for the one-paragraph version.

## Development

Format, lint, and type-check with the Astral toolchain (via `uvx`, no install):

```
uvx ruff format python/
uvx ruff check python/
uvx ty check python/
```

All three are clean on `ds4.py`.
