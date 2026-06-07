"""Export a fine-tuned MTP head -> standalone mtp.0.* gguf that ds4 loads.

Only the ~25 trained (conditioning) tensors change; the 256 routed experts are
FROZEN, so we take the ORIGINAL standalone MTP gguf, overwrite the trained tensors
(re-quantized to their original gguf type: Q8_0 dense / F32 norms), and copy
everything else (incl. the Q4_K experts) byte-for-byte. Tensor orientation
(transformers [out,in] vs the gguf layout) is auto-detected per tensor by
comparing shapes against the original — no guessing.

Trained ckpt = train_mtp.py's export ({"trainable": {param_name: tensor}}).
Usage: export_gguf.py --orig ORIG.gguf --ckpt mtp_finetuned.pt --out NEW.gguf
"""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/trevor/Projects/llama.cpp-tjs-fork/gguf-py")
from gguf import (  # noqa: E402
    GGMLQuantizationType,
    GGUFReader,
    GGUFValueType,
    GGUFWriter,
    quants,
)

# transformers param name (head.named_parameters)  ->  gguf tensor name
EXPORT_MAP = {
    "decoder.self_attn.q_a_proj.weight": "mtp.0.attn_q_a.weight",
    "decoder.self_attn.q_a_norm.weight": "mtp.0.attn_q_a_norm.weight",
    "decoder.self_attn.q_b_proj.weight": "mtp.0.attn_q_b.weight",
    "decoder.self_attn.kv_proj.weight": "mtp.0.attn_kv.weight",
    "decoder.self_attn.kv_norm.weight": "mtp.0.attn_kv_a_norm.weight",
    "decoder.self_attn.o_a_proj.weight": "mtp.0.attn_output_a.weight",
    "decoder.self_attn.o_b_proj.weight": "mtp.0.attn_output_b.weight",
    "decoder.self_attn.sinks": "mtp.0.attn_sinks.weight",
    "decoder.mlp.gate.weight": "mtp.0.ffn_gate_inp.weight",
    # NB: mtp.0.exp_probs_b.bias (router noaux_tc bias) is a non-gradient buffer,
    # not trained -> copied verbatim, deliberately NOT in this map.
    "decoder.mlp.shared_experts.gate_proj.weight": "mtp.0.ffn_gate_shexp.weight",
    "decoder.mlp.shared_experts.up_proj.weight": "mtp.0.ffn_up_shexp.weight",
    "decoder.mlp.shared_experts.down_proj.weight": "mtp.0.ffn_down_shexp.weight",
    "decoder.input_layernorm.weight": "mtp.0.attn_norm.weight",
    "decoder.post_attention_layernorm.weight": "mtp.0.ffn_norm.weight",
    "decoder.attn_hc.fn": "mtp.0.hc_attn_fn.weight",
    "decoder.attn_hc.base": "mtp.0.hc_attn_base.weight",
    "decoder.attn_hc.scale": "mtp.0.hc_attn_scale.weight",
    "decoder.ffn_hc.fn": "mtp.0.hc_ffn_fn.weight",
    "decoder.ffn_hc.base": "mtp.0.hc_ffn_base.weight",
    "decoder.ffn_hc.scale": "mtp.0.hc_ffn_scale.weight",
    "enorm.weight": "mtp.0.enorm.weight",
    "hnorm.weight": "mtp.0.hnorm.weight",
    "norm.weight": "mtp.0.norm.weight",
    "e_proj.weight": "mtp.0.e_proj.weight",
    "h_proj.weight": "mtp.0.h_proj.weight",
    "hc_head.hc_fn": "mtp.0.hc_head_fn.weight",
    "hc_head.hc_base": "mtp.0.hc_head_base.weight",
    "hc_head.hc_scale": "mtp.0.hc_head_scale.weight",
}

_SCALAR_ADDERS = {
    GGUFValueType.UINT8: "add_uint8",
    GGUFValueType.INT8: "add_int8",
    GGUFValueType.UINT16: "add_uint16",
    GGUFValueType.INT16: "add_int16",
    GGUFValueType.UINT32: "add_uint32",
    GGUFValueType.INT32: "add_int32",
    GGUFValueType.FLOAT32: "add_float32",
    GGUFValueType.UINT64: "add_uint64",
    GGUFValueType.INT64: "add_int64",
    GGUFValueType.FLOAT64: "add_float64",
    GGUFValueType.BOOL: "add_bool",
    GGUFValueType.STRING: "add_string",
}


def copy_field(writer, name, field):
    t = field.types[0]
    if t == GGUFValueType.ARRAY:
        writer.add_array(name, list(field.contents()))
    else:
        getattr(writer, _SCALAR_ADDERS[t])(name, field.contents())


def orient(val: np.ndarray, gguf_shape: tuple) -> np.ndarray:
    """Return the numpy array to WRITE so GGUFReader reads it back as gguf_shape.

    GGUF/ggml stores ne[] reversed from numpy and the reader reverses it back, so
    a written numpy array of shape S reads back as S[::-1] with identical bytes
    (verified). So the write-shape we want is gguf_shape[::-1]. The trained tensors
    are already in the base producer's orientation: 1D/square match gguf_shape
    (== its own reverse), 2D non-square match gguf_shape[::-1] — both write AS-IS.
    Only a 2D tensor stored in the opposite (== gguf_shape) layout needs transpose.
    The old code targeted gguf_shape directly and transposed the 12 non-square
    tensors backwards, double-reversing and corrupting the layout (caught by ds4
    rejecting hc_head_fn with dim[0]=4)."""
    want = tuple(gguf_shape[::-1])
    vs = tuple(val.shape)
    if vs == want:
        return val
    if val.ndim == 2 and vs == tuple(gguf_shape):
        return np.ascontiguousarray(val.T)
    raise ValueError(f"shape mismatch: trained {vs} vs gguf {gguf_shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, help="original standalone MTP gguf")
    ap.add_argument("--ckpt", required=True, help="train_mtp export (.pt)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    trained = torch.load(args.ckpt, map_location="cpu", weights_only=False)["trainable"]
    gguf_to_param = {v: k for k, v in EXPORT_MAP.items()}
    reader = GGUFReader(args.orig)
    arch = reader.fields["general.architecture"].contents()
    writer = GGUFWriter(args.out, arch=arch)

    skip = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "general.architecture",
    }
    for fname, field in reader.fields.items():
        if fname not in skip:
            copy_field(writer, fname, field)

    n_patched, n_verbatim, n_missing = 0, 0, 0
    for t in reader.tensors:
        param = gguf_to_param.get(t.name)
        gguf_shape = tuple(int(x) for x in t.shape)
        qtype = GGMLQuantizationType(t.tensor_type)
        if param is not None and param in trained:
            val = orient(trained[param].float().numpy(), gguf_shape)
            if qtype == GGMLQuantizationType.F32:
                writer.add_tensor(t.name, val.astype(np.float32))
            else:
                # quants.quantize returns the PACKED byte array (e.g. Q8_0 row of
                # 1024 elems -> 1088 bytes); add_tensor derives the element shape
                # from those bytes. Passing raw_shape=gguf_shape (element shape)
                # would be mis-read as a byte shape (1024 % 34 != 0). Omit it.
                q = quants.quantize(val, qtype)
                writer.add_tensor(t.name, q, raw_dtype=qtype)
            n_patched += 1
        else:
            if param is not None:
                n_missing += (
                    1  # mapped but not in ckpt (shouldn't happen for trained set)
                )
            writer.add_tensor(t.name, t.data, raw_dtype=qtype)
            n_verbatim += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(
        f"export -> {args.out}\n  patched (trained): {n_patched}  verbatim: {n_verbatim}"
        + (f"  WARN mapped-but-missing: {n_missing}" if n_missing else "")
    )
    if n_patched != len(EXPORT_MAP):
        print(
            f"  WARN: patched {n_patched} of {len(EXPORT_MAP)} expected trained tensors"
        )


if __name__ == "__main__":
    sys.exit(main())
