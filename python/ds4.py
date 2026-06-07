"""In-process Python binding for the ds4 inference engine.

Loads libds4.so (build with `make libds4.so`) and exposes the *entire* public
C API from ds4.h via ctypes: engine + session lifecycle, tokenization and chat
prompt builders, sampling / argmax / logprobs / raw logits, MTP speculative
decode, runtime directional steering, KV snapshot + payload persistence,
context-memory estimation, imatrix collection, and the diagnostics helpers.

Quick start:
    from ds4 import Engine
    eng = Engine("ds4flash.gguf")
    print(eng.complete("Explain MoE routing in one sentence.", max_tokens=80))

Low-level (manual session control):
    s = eng.session(ctx_size=8192)
    s.sync(eng.encode_chat("Hello"))
    tok = s.sample(temperature=0.7)
    s.eval(tok)
    print(s.top_logprobs(5))

Only the loaded variant fits GB10's 128 GB unified memory once; you cannot run
this alongside a live ds4-server (single-instance lock + 81 GB weights).
"""

import array
import ctypes as C
import os
import tempfile
from pathlib import Path

# ── enums (mirror ds4.h) ────────────────────────────────────────────────────
BACKEND_METAL, BACKEND_CUDA, BACKEND_CPU = 0, 1, 2
THINK_NONE, THINK_HIGH, THINK_MAX = 0, 1, 2
(
    LOG_DEFAULT,
    LOG_PREFILL,
    LOG_GENERATION,
    LOG_KVCACHE,
    LOG_TOOL,
    LOG_WARNING,
    LOG_TIMING,
    LOG_OK,
    LOG_ERROR,
) = range(9)
# ds4_session_rewrite_result
REWRITE_ERROR, REWRITE_OK, REWRITE_REBUILD_NEEDED = -1, 0, 1

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.05

_P = C.c_void_p  # opaque ds4_engine* / ds4_session*


# ── structs (layouts must match ds4.h exactly) ──────────────────────────────
class _EngineOptions(C.Structure):
    _fields_ = [
        ("model_path", C.c_char_p),
        ("mtp_path", C.c_char_p),
        ("backend", C.c_int),
        ("n_threads", C.c_int),
        ("mtp_draft_tokens", C.c_int),
        ("mtp_margin", C.c_float),
        ("directional_steering_file", C.c_char_p),
        ("directional_steering_attn", C.c_float),
        ("directional_steering_ffn", C.c_float),
        ("power_percent", C.c_int),
        ("warm_weights", C.c_bool),
        ("quality", C.c_bool),
        ("inspect_only", C.c_bool),
    ]


class _Tokens(C.Structure):
    _fields_ = [("v", C.POINTER(C.c_int)), ("len", C.c_int), ("cap", C.c_int)]


class _TokenScore(C.Structure):
    _fields_ = [("id", C.c_int), ("logit", C.c_float), ("logprob", C.c_float)]


class _ContextMemory(C.Structure):
    _fields_ = [
        ("total_bytes", C.c_uint64),
        ("raw_bytes", C.c_uint64),
        ("compressed_bytes", C.c_uint64),
        ("scratch_bytes", C.c_uint64),
        ("prefill_cap", C.c_uint32),
        ("raw_cap", C.c_uint32),
        ("comp_cap", C.c_uint32),
    ]


class _Snapshot(C.Structure):
    _fields_ = [("ptr", C.c_void_p), ("len", C.c_uint64), ("cap", C.c_uint64)]


# callback function types
_EMIT_FN = C.CFUNCTYPE(None, C.c_void_p, C.c_int)
_DONE_FN = C.CFUNCTYPE(None, C.c_void_p)
_PROGRESS_FN = C.CFUNCTYPE(None, C.c_void_p, C.c_char_p, C.c_int, C.c_int)


# ── libc helpers (free strings, FILE* for the payload/dump APIs) ─────────────
_libc = C.CDLL(None)
_libc.free.argtypes = [C.c_void_p]
_libc.free.restype = None
_libc.fopen.argtypes = [C.c_char_p, C.c_char_p]
_libc.fopen.restype = C.c_void_p
_libc.fclose.argtypes = [C.c_void_p]
_libc.fclose.restype = C.c_int
_libc.fflush.argtypes = [C.c_void_p]
_libc.fflush.restype = C.c_int


# ── library loading + symbol binding ─────────────────────────────────────────
_LIB = None


def _find_lib(path=None):
    path = path or os.environ.get("DS4_LIB")
    if path:
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(here, "libds4.so"),  # built into python/
        os.path.join(os.path.dirname(here), "libds4.so"),  # repo root (parent)
        os.path.join(os.getcwd(), "libds4.so"),
    ):  # cwd
        if os.path.exists(cand):
            return cand
    return "libds4.so"  # last resort: let the dynamic loader search


def _bind(lib):
    def f(name, res, args):
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args

    cp, sz, fl, i, b = C.c_char_p, C.c_size_t, C.c_float, C.c_int, C.c_bool
    PT, PS, PSN = C.POINTER(_Tokens), C.POINTER(_TokenScore), C.POINTER(_Snapshot)

    # engine lifecycle / metadata
    f("ds4_engine_open", i, [C.POINTER(_P), C.POINTER(_EngineOptions)])
    f("ds4_engine_close", None, [_P])
    f("ds4_engine_summary", None, [_P])
    f("ds4_engine_vocab_size", i, [_P])
    f("ds4_engine_power", i, [_P])
    f("ds4_engine_set_power", i, [_P, i])
    f("ds4_engine_model_name", cp, [_P])
    f("ds4_engine_model_id", i, [_P])
    f("ds4_engine_routed_quant_bits", i, [_P])
    f("ds4_engine_has_mtp", b, [_P])
    f("ds4_engine_mtp_draft_tokens", i, [_P])
    f("ds4_backend_name", cp, [i])

    # think modes
    f("ds4_think_mode_enabled", b, [i])
    f("ds4_think_mode_name", cp, [i])
    f("ds4_think_max_prefix", cp, [])
    f("ds4_think_max_min_context", C.c_uint32, [])
    f("ds4_think_mode_for_context", i, [i, i])
    f("ds4_context_memory_estimate", _ContextMemory, [i, i])

    # logging
    f("ds4_log_is_tty", b, [C.c_void_p])
    lib.ds4_log.restype = None  # varargs: leave argtypes unset

    # engine ops
    f(
        "ds4_engine_generate_argmax",
        i,
        [_P, PT, i, i, _EMIT_FN, _DONE_FN, C.c_void_p, _PROGRESS_FN, C.c_void_p],
    )
    f("ds4_engine_collect_imatrix", i, [_P, cp, cp, i, i, i])
    f("ds4_engine_dump_tokens", None, [_P, PT])
    f("ds4_dump_text_tokenization", i, [cp, cp, C.c_void_p])
    f("ds4_engine_head_test", i, [_P, PT])
    f("ds4_engine_first_token_test", i, [_P, PT])
    f("ds4_engine_metal_graph_test", i, [_P, PT])
    f("ds4_engine_metal_graph_full_test", i, [_P, PT])
    f("ds4_engine_metal_graph_prompt_test", i, [_P, PT, i])

    # token vectors
    f("ds4_tokens_push", None, [PT, i])
    f("ds4_tokens_free", None, [PT])
    f("ds4_tokens_copy", None, [PT, PT])
    f("ds4_tokens_starts_with", b, [PT, PT])

    # tokenization / chat builders
    f("ds4_tokenize_text", None, [_P, cp, PT])
    f("ds4_tokenize_rendered_chat", None, [_P, cp, PT])
    f("ds4_chat_begin", None, [_P, PT])
    f("ds4_encode_chat_prompt", None, [_P, cp, cp, i, PT])
    f("ds4_chat_append_max_effort_prefix", None, [_P, PT])
    f("ds4_chat_append_message", None, [_P, PT, cp, cp])
    f("ds4_chat_append_assistant_prefix", None, [_P, PT, i])

    # token <-> text / special ids
    f("ds4_token_text", C.c_void_p, [_P, i, C.POINTER(sz)])
    f("ds4_token_eos", i, [_P])
    f("ds4_token_user", i, [_P])
    f("ds4_token_assistant", i, [_P])

    # session lifecycle / power
    f("ds4_session_create", i, [C.POINTER(_P), _P, i])
    f("ds4_session_free", None, [_P])
    f("ds4_session_power", i, [_P])
    f("ds4_session_set_power", i, [_P, i])

    # steering
    f("ds4_session_set_steering_scale", i, [_P, fl, fl])
    f(
        "ds4_session_get_steering",
        None,
        [_P, C.POINTER(fl), C.POINTER(fl), C.POINTER(b)],
    )
    f("ds4_session_steering_is_cached", b, [_P, cp])
    f("ds4_session_steering_select", i, [_P, cp, cp, fl, fl, cp, sz])
    f("ds4_session_reload_steering", i, [_P, cp, fl, fl, cp, sz])
    f("ds4_session_set_progress", None, [_P, _PROGRESS_FN, C.c_void_p])
    f("ds4_session_set_display_progress", None, [_P, _PROGRESS_FN, C.c_void_p])

    # sync / rewrite
    f("ds4_session_sync", i, [_P, PT, cp, sz])
    f("ds4_session_rewrite_requires_rebuild", b, [i, i, i])
    f("ds4_session_rewrite_from_common", i, [_P, PT, i, cp, sz])
    f("ds4_session_common_prefix", i, [_P, PT])

    # sampling / logits
    f("ds4_session_argmax", i, [_P])
    f("ds4_session_argmax_excluding", i, [_P, i])
    f("ds4_session_sample", i, [_P, fl, i, fl, fl, C.POINTER(C.c_uint64)])
    f("ds4_session_top_logprobs", i, [_P, PS, i])
    f("ds4_session_token_logprob", i, [_P, i, PS])
    f("ds4_session_copy_logits", i, [_P, C.POINTER(fl), i])

    # eval / position
    f("ds4_session_eval", i, [_P, i, cp, sz])
    f("ds4_session_eval_speculative_argmax", i, [_P, i, i, i, C.POINTER(i), i, cp, sz])
    f("ds4_session_invalidate", None, [_P])
    f("ds4_session_rewind", None, [_P, i])
    f("ds4_session_pos", i, [_P])
    f("ds4_session_ctx", i, [_P])
    f("ds4_session_tokens", PT, [_P])

    # KV persistence
    f("ds4_session_payload_bytes", C.c_uint64, [_P])
    f("ds4_session_save_payload", i, [_P, C.c_void_p, cp, sz])
    f("ds4_session_load_payload", i, [_P, C.c_void_p, C.c_uint64, cp, sz])
    f("ds4_session_save_snapshot", i, [_P, PSN, cp, sz])
    f("ds4_session_load_snapshot", i, [_P, PSN, cp, sz])
    f("ds4_session_snapshot_free", None, [PSN])
    return lib


def lib():
    """Lazily load + bind libds4.so once per process."""
    global _LIB
    if _LIB is None:
        _LIB = _bind(C.CDLL(_find_lib()))
    return _LIB


# ── token-vector helpers ─────────────────────────────────────────────────────
def _in_tokens(ids):
    """Build a read-only ds4_tokens backed by a Python array (returns struct, keepalive)."""
    n = len(ids)
    arr = (C.c_int * n)(*ids)
    t = _Tokens(v=C.cast(arr, C.POINTER(C.c_int)), len=n, cap=n)
    return t, arr


def _read_tokens(t):
    return [t.v[i] for i in range(t.len)] if (t.v and t.len > 0) else []


def _errbuf():
    return C.create_string_buffer(256)


# ── module-level helpers that don't need an engine ──────────────────────────
def backend_name(backend):
    return lib().ds4_backend_name(backend).decode()


def think_mode_enabled(mode):
    return bool(lib().ds4_think_mode_enabled(mode))


def think_mode_name(mode):
    return lib().ds4_think_mode_name(mode).decode()


def think_max_prefix():
    return lib().ds4_think_max_prefix().decode()


def think_max_min_context():
    return int(lib().ds4_think_max_min_context())


def think_mode_for_context(mode, ctx_size):
    return lib().ds4_think_mode_for_context(mode, ctx_size)


def context_memory_estimate(backend, ctx_size):
    """Returns a dict of the ds4_context_memory fields for (backend, ctx_size)."""
    cm = lib().ds4_context_memory_estimate(backend, ctx_size)
    return {field[0]: getattr(cm, field[0]) for field in _ContextMemory._fields_}


def dump_text_tokenization(model_path, text, out_path="/dev/stdout"):
    fp = _libc.fopen(out_path.encode(), b"w")
    if not fp:
        raise OSError(f"cannot open {out_path}")
    try:
        return lib().ds4_dump_text_tokenization(model_path.encode(), text.encode(), fp)
    finally:
        _libc.fflush(fp)
        _libc.fclose(fp)


# ── chat prompt builder ──────────────────────────────────────────────────────
class ChatBuilder:
    """Incrementally assemble a chat token sequence via the ds4_chat_* primitives.

    cb = eng.chat_builder()
    cb.message("user", "Hi").assistant_prefix(THINK_HIGH)
    ids = cb.tokens()
    """

    def __init__(self, engine):
        self._e = engine
        self._t = _Tokens()
        lib().ds4_chat_begin(engine.h, C.byref(self._t))

    def message(self, role, content):
        lib().ds4_chat_append_message(
            self._e.h, C.byref(self._t), role.encode(), content.encode()
        )
        return self

    def assistant_prefix(self, think=THINK_NONE):
        lib().ds4_chat_append_assistant_prefix(self._e.h, C.byref(self._t), think)
        return self

    def max_effort_prefix(self):
        lib().ds4_chat_append_max_effort_prefix(self._e.h, C.byref(self._t))
        return self

    def push(self, token):
        lib().ds4_tokens_push(C.byref(self._t), token)
        return self

    def tokens(self):
        return _read_tokens(self._t)

    def __del__(self):
        try:
            lib().ds4_tokens_free(C.byref(self._t))
        except Exception:
            pass


# ── engine ───────────────────────────────────────────────────────────────────
class Engine:
    def __init__(
        self,
        model_path="ds4flash.gguf",
        *,
        backend=BACKEND_CUDA,
        mtp_path=None,
        n_threads=0,
        mtp_draft_tokens=2,
        mtp_margin=3.0,
        steering_file=None,
        steering_attn=0.0,
        steering_ffn=0.0,
        power_percent=0,
        warm_weights=False,
        quality=False,
        inspect_only=False,
    ):
        self.model_path = model_path
        self.backend = backend
        opt = _EngineOptions(
            model_path=model_path.encode(),
            mtp_path=mtp_path.encode() if mtp_path else None,
            backend=backend,
            n_threads=n_threads,
            mtp_draft_tokens=mtp_draft_tokens,
            mtp_margin=mtp_margin,
            directional_steering_file=steering_file.encode() if steering_file else None,
            directional_steering_attn=steering_attn,
            directional_steering_ffn=steering_ffn,
            power_percent=power_percent,
            warm_weights=warm_weights,
            quality=quality,
            inspect_only=inspect_only,
        )
        h = C.c_void_p()
        if lib().ds4_engine_open(C.byref(h), C.byref(opt)) != 0:
            raise RuntimeError(f"ds4_engine_open failed for {model_path!r}")
        self.h = h
        self._vocab = lib().ds4_engine_vocab_size(h)

    # lifecycle
    def close(self):
        if getattr(self, "h", None):
            lib().ds4_engine_close(self.h)
            self.h = None

    def __del__(self):
        self.close()

    def summary(self):
        lib().ds4_engine_summary(self.h)

    # metadata
    @property
    def model_name(self):
        return lib().ds4_engine_model_name(self.h).decode()

    @property
    def model_id(self):
        return lib().ds4_engine_model_id(self.h)

    @property
    def vocab_size(self):
        return self._vocab

    @property
    def backend_name(self):
        return backend_name(self.backend)

    @property
    def routed_quant_bits(self):
        return lib().ds4_engine_routed_quant_bits(self.h)

    @property
    def has_mtp(self):
        return bool(lib().ds4_engine_has_mtp(self.h))

    @property
    def mtp_draft_tokens(self):
        return lib().ds4_engine_mtp_draft_tokens(self.h)

    @property
    def power(self):
        return lib().ds4_engine_power(self.h)

    @power.setter
    def power(self, percent):
        lib().ds4_engine_set_power(self.h, percent)

    # special token ids
    @property
    def eos(self):
        return lib().ds4_token_eos(self.h)

    @property
    def user_token(self):
        return lib().ds4_token_user(self.h)

    @property
    def assistant_token(self):
        return lib().ds4_token_assistant(self.h)

    # tokenization
    def tokenize(self, text):
        t = _Tokens()
        lib().ds4_tokenize_text(self.h, text.encode(), C.byref(t))
        out = _read_tokens(t)
        lib().ds4_tokens_free(C.byref(t))
        return out

    def tokenize_rendered_chat(self, text):
        t = _Tokens()
        lib().ds4_tokenize_rendered_chat(self.h, text.encode(), C.byref(t))
        out = _read_tokens(t)
        lib().ds4_tokens_free(C.byref(t))
        return out

    def encode_chat(self, prompt, system=None, think=THINK_NONE):
        t = _Tokens()
        lib().ds4_encode_chat_prompt(
            self.h,
            system.encode() if system else None,
            prompt.encode(),
            think,
            C.byref(t),
        )
        out = _read_tokens(t)
        lib().ds4_tokens_free(C.byref(t))
        return out

    def chat_builder(self):
        return ChatBuilder(self)

    def token_text(self, token):
        n = C.c_size_t(0)
        ptr = lib().ds4_token_text(self.h, token, C.byref(n))
        if not ptr:
            return ""
        try:
            return (
                C.string_at(ptr, n.value).decode("utf-8", "replace") if n.value else ""
            )
        finally:
            _libc.free(ptr)

    def detokenize(self, ids):
        return "".join(self.token_text(t) for t in ids)

    def dump_tokens(self, ids):
        t, _keep = _in_tokens(ids)
        lib().ds4_engine_dump_tokens(self.h, C.byref(t))

    def dump_text_tokenization(self, text, out_path="/dev/stdout"):
        return dump_text_tokenization(self.model_path, text, out_path)

    # diagnostics (head/first-token always available; metal_* are Metal-build only)
    def head_test(self, ids):
        t, _ = _in_tokens(ids)
        return lib().ds4_engine_head_test(self.h, C.byref(t))

    def first_token_test(self, ids):
        t, _ = _in_tokens(ids)
        return lib().ds4_engine_first_token_test(self.h, C.byref(t))

    def metal_graph_test(self, ids):
        t, _ = _in_tokens(ids)
        return lib().ds4_engine_metal_graph_test(self.h, C.byref(t))

    def metal_graph_full_test(self, ids):
        t, _ = _in_tokens(ids)
        return lib().ds4_engine_metal_graph_full_test(self.h, C.byref(t))

    def metal_graph_prompt_test(self, ids, ctx_size):
        t, _ = _in_tokens(ids)
        return lib().ds4_engine_metal_graph_prompt_test(self.h, C.byref(t), ctx_size)

    def collect_imatrix(
        self, dataset_path, output_path, ctx_size, max_prompts=0, max_tokens=0
    ):
        return lib().ds4_engine_collect_imatrix(
            self.h,
            dataset_path.encode(),
            output_path.encode(),
            ctx_size,
            max_prompts,
            max_tokens,
        )

    # engine built-in argmax generation (uses MTP if available); returns token ids
    def generate_argmax(
        self, prompt_ids, n_predict, ctx_size, *, on_token=None, on_progress=None
    ):
        out = []

        @_EMIT_FN
        def emit(_ud, tok):
            out.append(tok)
            if on_token:
                on_token(tok)

        @_DONE_FN
        def done(_ud):
            pass

        @_PROGRESS_FN
        def progress(_ud, event, cur, total):
            if on_progress:
                on_progress(event.decode() if event else "", cur, total)

        t, _keep = _in_tokens(prompt_ids)
        rc = lib().ds4_engine_generate_argmax(
            self.h, C.byref(t), n_predict, ctx_size, emit, done, None, progress, None
        )
        if rc != 0:
            raise RuntimeError(f"generate_argmax failed (rc={rc})")
        return out

    # sessions
    def session(self, ctx_size=8192):
        return Session(self, ctx_size)

    # high-level convenience: streaming sampled generation
    def generate(
        self,
        prompt,
        *,
        system=None,
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        min_p=0.05,
        think=THINK_NONE,
        seed=0,
        ctx_size=8192,
    ):
        """Yield decoded text pieces. temperature<=0 => greedy."""
        s = self.session(ctx_size)
        try:
            s.sync(self.encode_chat(prompt, system=system, think=think))
            budget = max(0, min(max_tokens, s.ctx - s.pos - 1))
            rng = seed or 0
            for _ in range(budget):
                tok = s.sample(
                    temperature=temperature, top_p=top_p, min_p=min_p, seed=rng
                )
                rng = s._rng_state  # carry rng forward
                if tok == self.eos:
                    break
                s.eval(tok)
                yield self.token_text(tok)
        finally:
            s.close()

    def complete(self, prompt, **kw):
        return "".join(self.generate(prompt, **kw))


# ── session ──────────────────────────────────────────────────────────────────
class Session:
    def __init__(self, engine, ctx_size=8192):
        self.engine = engine
        h = C.c_void_p()
        if lib().ds4_session_create(C.byref(h), engine.h, ctx_size) != 0:
            raise RuntimeError("ds4_session_create failed (needs a session backend)")
        self.h = h
        self._rng_state = 0
        self._cb_refs = []  # keep CFUNCTYPE callbacks alive

    def close(self):
        if getattr(self, "h", None):
            lib().ds4_session_free(self.h)
            self.h = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # position / context
    @property
    def pos(self):
        return lib().ds4_session_pos(self.h)

    @property
    def ctx(self):
        return lib().ds4_session_ctx(self.h)

    def tokens(self):
        tp = lib().ds4_session_tokens(self.h)
        return _read_tokens(tp.contents) if tp else []

    def invalidate(self):
        lib().ds4_session_invalidate(self.h)

    def rewind(self, pos):
        lib().ds4_session_rewind(self.h, pos)

    # power
    @property
    def power(self):
        return lib().ds4_session_power(self.h)

    @power.setter
    def power(self, percent):
        lib().ds4_session_set_power(self.h, percent)

    # steering
    def set_steering_scale(self, attn, ffn):
        return lib().ds4_session_set_steering_scale(self.h, attn, ffn)

    def steering(self, attn, ffn):
        """Context manager: temporarily set the steering scale, restoring
        the prior (attn, ffn) on exit. Ergonomic for sweeps.

        Requires the direction tensor to already be loaded on this session
        — pass non-zero ``steering_attn``/``steering_ffn`` at engine open OR
        call :meth:`reload_steering` first. Setting a scale here without
        loaded dirs has no effect on the forward pass (the C side gates
        load on a non-zero scale at session create)."""
        sess = self

        class _Scoped:
            def __enter__(self):
                prior_attn, prior_ffn, _ = sess.get_steering()
                self._prior = (prior_attn, prior_ffn)
                sess.set_steering_scale(attn, ffn)
                return sess

            def __exit__(self, *_exc):
                sess.set_steering_scale(*self._prior)
                return False

        return _Scoped()

    def get_steering(self):
        attn, ffn, loaded = C.c_float(), C.c_float(), C.c_bool()
        lib().ds4_session_get_steering(
            self.h, C.byref(attn), C.byref(ffn), C.byref(loaded)
        )
        return attn.value, ffn.value, bool(loaded.value)

    def steering_is_cached(self, name):
        return bool(lib().ds4_session_steering_is_cached(self.h, name.encode()))

    def steering_select(self, name, path, attn, ffn):
        err = _errbuf()
        rc = lib().ds4_session_steering_select(
            self.h,
            name.encode() if name else None,
            path.encode() if path else None,
            attn,
            ffn,
            err,
            len(err),
        )
        if rc != 0:
            raise RuntimeError(f"steering_select failed: {err.value.decode()}")
        return rc

    def reload_steering(self, path, attn, ffn):
        err = _errbuf()
        rc = lib().ds4_session_reload_steering(
            self.h, path.encode() if path else None, attn, ffn, err, len(err)
        )
        if rc != 0:
            raise RuntimeError(f"reload_steering failed: {err.value.decode()}")
        return rc

    # activation capture (file-based dump hooks in ds4.c)
    def collect_layer_activations(
        self,
        prompt_tokens,
        *,
        component,
        n_layers,
        n_embd,
        pos=0,
        work_dir=None,
    ):
        """Sync prompt_tokens with per-layer activation dumps enabled, then
        read back the captured rows. Returns list[array.array("f")] of length
        n_layers, each entry n_embd floats long.

        Wraps the file-based DS4_METAL_GRAPH_DUMP_{PREFIX,NAME,POS} hooks
        in ds4.c: env vars are set around the underlying sync() call and
        the prior values are restored on exit. component must match a
        dumpable tensor name (e.g. "ffn_out", "attn_out").

        Disturbs the session's KV cache (sync writes through it). If
        work_dir is None, a private tempdir is created and cleaned up.
        """
        own_work = work_dir is None
        work = Path(tempfile.mkdtemp(prefix="ds4-act-") if own_work else work_dir)
        if not own_work:
            work.mkdir(parents=True, exist_ok=True)
        prefix = work / "dump"
        keys = (
            "DS4_METAL_GRAPH_DUMP_PREFIX",
            "DS4_METAL_GRAPH_DUMP_NAME",
            "DS4_METAL_GRAPH_DUMP_POS",
        )
        prev = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["DS4_METAL_GRAPH_DUMP_PREFIX"] = str(prefix)
            os.environ["DS4_METAL_GRAPH_DUMP_NAME"] = component
            os.environ["DS4_METAL_GRAPH_DUMP_POS"] = str(pos)
            self.sync(prompt_tokens)
            rows = []
            for layer in range(n_layers):
                path = work / f"dump_{component}-{layer}_pos{pos}.bin"
                if not path.exists():
                    raise RuntimeError(f"dump file missing: {path}")
                data = array.array("f")
                with path.open("rb") as f:
                    data.fromfile(f, path.stat().st_size // 4)
                if len(data) < n_embd or len(data) % n_embd != 0:
                    raise RuntimeError(f"bad dump shape {path}: {len(data)} floats")
                rows.append(data[-n_embd:])
            return rows
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            if own_work:
                for p in work.glob("*"):
                    try:
                        p.unlink()
                    except OSError:
                        pass
                try:
                    work.rmdir()
                except OSError:
                    pass

    def set_progress(self, callback, display=False):
        """callback(event:str, current:int, total:int). Pass None to clear."""
        setter = (
            lib().ds4_session_set_display_progress
            if display
            else lib().ds4_session_set_progress
        )
        if callback is None:
            setter(self.h, C.cast(0, _PROGRESS_FN), None)  # NULL fn ptr clears it
            return

        @_PROGRESS_FN
        def cb(_ud, event, cur, total):
            callback(event.decode() if event else "", cur, total)

        self._cb_refs.append(cb)
        setter(self.h, cb, None)

    # sync / rewrite
    def sync(self, ids):
        t, _keep = _in_tokens(ids)
        err = _errbuf()
        if lib().ds4_session_sync(self.h, C.byref(t), err, len(err)) != 0:
            raise RuntimeError(f"sync failed: {err.value.decode()}")

    def common_prefix(self, ids):
        t, _keep = _in_tokens(ids)
        return lib().ds4_session_common_prefix(self.h, C.byref(t))

    def rewrite_from_common(self, ids, common):
        t, _keep = _in_tokens(ids)
        err = _errbuf()
        rc = lib().ds4_session_rewrite_from_common(
            self.h, C.byref(t), common, err, len(err)
        )
        if rc == REWRITE_ERROR:
            raise RuntimeError(f"rewrite_from_common failed: {err.value.decode()}")
        return rc

    @staticmethod
    def rewrite_requires_rebuild(live_len, canonical_len, common):
        return bool(
            lib().ds4_session_rewrite_requires_rebuild(live_len, canonical_len, common)
        )

    # sampling / logits
    def argmax(self):
        return lib().ds4_session_argmax(self.h)

    def argmax_excluding(self, excluded_id):
        return lib().ds4_session_argmax_excluding(self.h, excluded_id)

    def sample(
        self,
        temperature=DEFAULT_TEMPERATURE,
        top_k=0,
        top_p=DEFAULT_TOP_P,
        min_p=DEFAULT_MIN_P,
        seed=0,
    ):
        rng = C.c_uint64(seed or self._rng_state or 0x9E3779B97F4A7C15)
        tok = lib().ds4_session_sample(
            self.h, temperature, top_k, top_p, min_p, C.byref(rng)
        )
        self._rng_state = rng.value
        return tok

    def top_logprobs(self, k):
        arr = (_TokenScore * k)()
        n = lib().ds4_session_top_logprobs(self.h, arr, k)
        return [
            (arr[i].id, arr[i].logit, arr[i].logprob)
            for i in range(n)
            if arr[i].id >= 0
        ]

    def token_logprob(self, token):
        sc = _TokenScore()
        if lib().ds4_session_token_logprob(self.h, token, C.byref(sc)) != 1:
            return None
        return (sc.id, sc.logit, sc.logprob)

    def logits(self):
        """Full vocab logit vector as an array('f')."""
        v = self.engine.vocab_size
        buf = (C.c_float * v)()
        n = lib().ds4_session_copy_logits(self.h, buf, v)
        return array.array("f", buf[:n])

    # eval
    def eval(self, token):
        err = _errbuf()
        if lib().ds4_session_eval(self.h, token, err, len(err)) != 0:
            raise RuntimeError(f"eval failed: {err.value.decode()}")

    def eval_speculative_argmax(self, first_token, max_tokens, eos_token, cap=16):
        arr = (C.c_int * cap)()
        err = _errbuf()
        n = lib().ds4_session_eval_speculative_argmax(
            self.h, first_token, max_tokens, eos_token, arr, cap, err, len(err)
        )
        if n < 0:
            raise RuntimeError(f"speculative decode failed: {err.value.decode()}")
        return list(arr[:n])

    # KV persistence
    def payload_bytes(self):
        return int(lib().ds4_session_payload_bytes(self.h))

    def save_payload(self, path):
        fp = _libc.fopen(path.encode(), b"wb")
        if not fp:
            raise OSError(f"cannot open {path} for write")
        err = _errbuf()
        try:
            if lib().ds4_session_save_payload(self.h, fp, err, len(err)) != 0:
                raise RuntimeError(f"save_payload failed: {err.value.decode()}")
        finally:
            _libc.fclose(fp)

    def load_payload(self, path):
        n = os.path.getsize(path)
        fp = _libc.fopen(path.encode(), b"rb")
        if not fp:
            raise OSError(f"cannot open {path} for read")
        err = _errbuf()
        try:
            if lib().ds4_session_load_payload(self.h, fp, n, err, len(err)) != 0:
                raise RuntimeError(f"load_payload failed: {err.value.decode()}")
        finally:
            _libc.fclose(fp)

    def save_snapshot(self):
        """Serialize live KV state to bytes."""
        snap = _Snapshot()
        err = _errbuf()
        if lib().ds4_session_save_snapshot(self.h, C.byref(snap), err, len(err)) != 0:
            raise RuntimeError(f"save_snapshot failed: {err.value.decode()}")
        try:
            return C.string_at(snap.ptr, snap.len) if snap.ptr and snap.len else b""
        finally:
            lib().ds4_session_snapshot_free(C.byref(snap))

    def load_snapshot(self, data):
        buf = C.create_string_buffer(bytes(data), len(data))
        snap = _Snapshot(ptr=C.cast(buf, C.c_void_p), len=len(data), cap=len(data))
        err = _errbuf()
        if lib().ds4_session_load_snapshot(self.h, C.byref(snap), err, len(err)) != 0:
            raise RuntimeError(f"load_snapshot failed: {err.value.decode()}")


# back-compat alias
Ds4 = Engine


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "ds4flash.gguf"
    eng = Engine(model)
    print(
        f"[loaded {eng.model_name} id={eng.model_id} vocab={eng.vocab_size} "
        f"mtp={eng.has_mtp} q{eng.routed_quant_bits}]",
        file=sys.stderr,
    )
    for piece in eng.generate(
        "Explain MoE expert routing in one sentence.", max_tokens=80
    ):
        print(piece, end="", flush=True)
    print()
