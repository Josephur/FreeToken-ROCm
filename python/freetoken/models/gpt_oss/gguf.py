"""GGUF adapter for gpt-oss (MoE): loads llama.cpp gpt-oss GGUF files by
repacking the GGML_MXFP4 expert tensors into the exact HF ``mxfp4_triton``
parameter layout ``GptOssMxfp4TritonMoELayer`` allocates, so the whole existing
MXFP4 fused-MoE machinery runs unchanged. Non-expert tensors dequantize to bf16
under their HF names (same policy as the dense-family adapter).

Layout facts this file reconciles (llama.cpp vs HF/triton):

- ggml ``block_mxfp4`` = 1 E8M0 scale byte + 16 nibble bytes where byte ``j``
  holds element ``j`` (lo nibble) and element ``j+16`` (hi). The HF/triton
  unpack is ``stack((b & 0xF, b >> 4), -1)`` -- byte ``k`` holds elements
  ``2k``/``2k+1`` -- so nibbles must be re-paired per 32-element block.
- llama.cpp stores gate and up experts as separate ``ffn_gate_exps`` /
  ``ffn_up_exps`` tensors; the triton kernels want one ``2I`` axis with gate at
  even and up at odd indices (see ``moe/fused_mxfp4.py::gpt_oss_swiglu``).
- E8M0 scale bytes (bias 127) are identical on both sides -- copied verbatim.

Supported: MXFP4-expert GGUFs (the canonical llama.cpp gpt-oss format), TP=1,
``--moe-backend fused``. Expert tensors in other ggml quants would need the
generalized gguf-quant expert bank (not implemented).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.gguf.dequant import GGML_MXFP4, GGML_NAME, dequant_any
from freetoken.models.gguf.dense import _rope_scaling
from freetoken.models.gguf.reader import GgufTensor, iter_gguf_tensors

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim


def parse_gguf_config(shim: "GgufConfigShim") -> "ModelConfig":
    """Build the gpt-oss ModelConfig from GGUF KV metadata via a HF-config
    lookalike fed to the family's own ``parse_config`` (dense-adapter pattern)."""
    m = shim.metadata
    prefix = shim.model_type  # "gpt-oss"

    def g(key: str, default=None):
        return m.get(f"{prefix}.{key}", default)

    num_layers = int(g("block_count"))
    # llama.cpp hardcodes the gpt-oss SWA pattern (every other layer, starting at
    # layer 0) instead of writing attention.sliding_window_pattern -- mirror it.
    layer_types = [
        "sliding_attention" if i % 2 == 0 else "full_attention"
        for i in range(num_layers)
    ]
    key_length = g("attention.key_length")

    hf_like = SimpleNamespace(
        num_hidden_layers=num_layers,
        num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")),
        head_dim=int(key_length) if key_length else None,
        hidden_size=int(g("embedding_length")),
        # gpt-oss's parse_config uses intermediate_size as the expert width
        intermediate_size=int(
            g("expert_feed_forward_length", g("feed_forward_length"))
        ),
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon", 1e-5)),
        max_position_embeddings=int(g("context_length", 131072)),
        rope_theta=float(g("rope.freq_base", 150_000.0)),
        rope_scaling=_rope_scaling(m, prefix),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        num_local_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        sliding_window=int(g("attention.sliding_window", 128)),
        layer_types=layer_types,
        attention_bias=True,
        quantization_config={"quant_method": "mxfp4"},
        swiglu_limit=float(g("swiglu_limit", 7.0)),
        hidden_act_alpha=1.702,
        model_type="gpt_oss",
        architectures=["GptOssForCausalLM"],
        torch_dtype="bfloat16",
    )
    return parse_config(hf_like)


# gguf layer-tensor suffix -> HF module-relative name for the bf16 (non-expert) path.
_SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_output.bias": "self_attn.o_proj.bias",
    "attn_sinks.weight": "self_attn.sinks",
    "ffn_gate_inp.weight": "mlp.router.weight",
    "ffn_gate_inp.bias": "mlp.router.bias",
}

# q/k/v fuse into qkv_proj in this concat order (matches the HF _MERGE_RULES).
_QKV_SLOTS = ("attn_q", "attn_k", "attn_v")

_EXPERT_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_gate_exps.bias",
    "ffn_up_exps.bias",
    "ffn_down_exps.bias",
)


def _to_bf16(t: GgufTensor) -> torch.Tensor:
    return dequant_any(t).to(torch.bfloat16)


def _mxfp4_to_hf(t: GgufTensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ggml MXFP4 ``[E, N, K]`` -> HF (blocks ``[E, N, K//32, 16]`` uint8,
    scales ``[E, N, K//32]`` uint8), with nibbles re-paired to the HF order."""
    assert t.ggml_type == GGML_MXFP4, (
        f"{t.name}: expert tensor is {GGML_NAME.get(t.ggml_type, t.ggml_type)}, "
        "but the gpt-oss GGUF adapter supports MXFP4 experts only"
    )
    e, n, k = t.shape
    raw = t.packed().reshape(e, n, k // 32, 17)
    scales = raw[..., 0].contiguous()
    qs = raw[..., 1:]
    # ggml element order: byte j = (elem j, elem j+16). HF: byte k = (elem 2k, 2k+1).
    elems = torch.cat((qs & 0x0F, qs >> 4), dim=-1)  # [..., 32] in element order
    blocks = (elems[..., 0::2] | (elems[..., 1::2] << 4)).contiguous()
    return blocks, scales


def _interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Stack gate/up along a doubled dim-1 with gate at even, up at odd indices."""
    assert gate.shape == up.shape, (gate.shape, up.shape)
    out = gate.new_empty((gate.shape[0], 2 * gate.shape[1], *gate.shape[2:]))
    out[:, 0::2] = gate
    out[:, 1::2] = up
    return out


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    from freetoken.distributed import get_tp_info

    try:
        tp_size = get_tp_info().size
    except RuntimeError:  # TP not initialized: offline tooling context
        tp_size = 1
    assert tp_size == 1, "gpt-oss GGUF supports TP=1 only"

    # buffers for fused/paired groups keyed per layer
    qkv: dict[int, dict[str, dict[str, torch.Tensor]]] = {}  # {layer: {kind: {slot: t}}}
    experts: dict[int, dict[str, GgufTensor]] = {}

    def feed_expert(layer: int, suffix: str, t: GgufTensor) -> Iterator[tuple[str, torch.Tensor]]:
        buf = experts.setdefault(layer, {})
        buf[suffix] = t
        stem = f"model.layers.{layer}.mlp.experts."
        if "ffn_gate_exps.weight" in buf and "ffn_up_exps.weight" in buf:
            gate_b, gate_s = _mxfp4_to_hf(buf.pop("ffn_gate_exps.weight"))
            up_b, up_s = _mxfp4_to_hf(buf.pop("ffn_up_exps.weight"))
            yield stem + "gate_up_proj_blocks", _interleave_gate_up(gate_b, up_b)
            yield stem + "gate_up_proj_scales", _interleave_gate_up(gate_s, up_s)
        if "ffn_gate_exps.bias" in buf and "ffn_up_exps.bias" in buf:
            gate = _to_bf16(buf.pop("ffn_gate_exps.bias"))
            up = _to_bf16(buf.pop("ffn_up_exps.bias"))
            yield stem + "gate_up_proj_bias", _interleave_gate_up(gate, up)
        if "ffn_down_exps.weight" in buf:
            down_b, down_s = _mxfp4_to_hf(buf.pop("ffn_down_exps.weight"))
            yield stem + "down_proj_blocks", down_b
            yield stem + "down_proj_scales", down_s
        if "ffn_down_exps.bias" in buf:
            yield stem + "down_proj_bias", _to_bf16(buf.pop("ffn_down_exps.bias"))

    def feed_qkv(layer: int, proj: str, kind: str, t: GgufTensor) -> Iterator[tuple[str, torch.Tensor]]:
        buf = qkv.setdefault(layer, {}).setdefault(kind, {})
        buf[proj] = _to_bf16(t)
        if len(buf) == len(_QKV_SLOTS):
            fused = torch.cat([buf[s] for s in _QKV_SLOTS], dim=0).contiguous()
            del qkv[layer][kind]
            if not qkv[layer]:
                del qkv[layer]
            yield f"model.layers.{layer}.self_attn.qkv_proj.{kind}", fused

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            if include_non_moe:
                yield "model.embed_tokens.weight", _to_bf16(t)
        elif name == "output_norm.weight":
            if include_non_moe:
                yield "model.norm.weight", _to_bf16(t)
        elif name == "output.weight":
            if include_non_moe:
                yield "lm_head.weight", _to_bf16(t)
        elif name.startswith("blk."):
            layer = int(name.split(".")[1])
            suffix = name.split(".", 2)[2]
            if suffix in _EXPERT_SUFFIXES:
                if include_moe_experts:
                    yield from feed_expert(layer, suffix, t)
                continue
            if not include_non_moe:
                continue
            proj, _, kind = suffix.rpartition(".")
            if proj in _QKV_SLOTS and kind in ("weight", "bias"):
                yield from feed_qkv(layer, proj, kind, t)
                continue
            rel = _SUFFIX_MAP.get(suffix)
            if rel is not None:
                yield f"model.layers.{layer}.{rel}", _to_bf16(t)
            # rope_freqs.weight and other auxiliaries are derivable -> skipped

    leftovers = {k: sorted(v) for k, v in qkv.items()}
    assert not leftovers, f"incomplete qkv groups: {leftovers}"
    expert_leftovers = {k: sorted(v) for k, v in experts.items() if v}
    assert not expert_leftovers, f"unpaired expert tensors: {expert_leftovers}"


__all__ = ["parse_gguf_config", "iter_gguf_weights"]
