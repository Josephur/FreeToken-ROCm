"""GGUF adapter for dense Qwen3.5/3.8 hybrid models (llama.cpp arch ``qwen35``).

Same family as ``qwen35moe`` (see gguf.py for the layout facts this builds on)
minus the routed experts: a dense SwiGLU MLP per layer, everything else shared
(GDN linear-attention layers + gated full-attention layers every
``full_attention_interval``).

Unlike the qwen35moe adapter (which dequantizes non-expert weights to bf16),
this adapter keeps every projection in its native GGUF packed blocks via the
shared GGUFLinear/GgufColSplits path (HIP MMVQ / Triton GEMM kernels), so a
27B IQ4_XS checkpoint occupies its on-disk ~14 GB instead of ~54 GB bf16.

Weight slots:
- full-attn layer:  self_attn.qkv_proj = q(2x, q|gate fused) | k | v,
                    self_attn.o_proj
- linear layer:     linear_attn.in_proj = qkv | z(gate) | b | a,
                    linear_attn.out_proj
- every layer:      mlp.gate_up_proj = gate | up, mlp.down_proj

A fused group whose slots are all quantized swaps to per-slot packed
GgufColSplits; a group with any unquantized slot stays the dense bf16 module
and the iterator fuses its slots. TP=1 only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.layers.base import BaseOP
from freetoken.models.gguf.dequant import (
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_Q8_0,
    dequant_any,
    row_bytes,
)
from freetoken.models.gguf.reader import iter_gguf_tensors, load_gguf_metadata

from .config import parse_config
from .model import Qwen3_5MoEForCausalLM  # noqa: F401  (re-export for the registry)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_UNQUANTIZED = {GGML_F32, GGML_F16, GGML_BF16}

# gguf suffixes whose ggml type decides packed-vs-bf16 (per fused group)
_GROUP_SLOTS = {
    "qkv": ("attn_q", "attn_k", "attn_v"),          # full-attention layers
    "in_proj": ("attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha"),  # GDN layers
    "gate_up": ("ffn_gate", "ffn_up"),
}
_SINGLE_SLOTS = ("attn_output", "ssm_out", "ffn_down")

# gguf suffix -> module-relative name, bf16/f32 passthrough tensors
_SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",  # +1 baked at conversion
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
}


def _is_full_attention(layer: int, interval: int) -> bool:
    return (layer + 1) % interval == 0


def _tensor_types(model_path: str) -> dict[str, int]:
    """ggml type of every projection/embedding tensor, keyed by gguf name."""
    wanted = {"token_embd.weight", "output.weight"}
    for slots in _GROUP_SLOTS.values():
        wanted.update(f"blk.N.{s}.weight" for s in slots)
    wanted.update(f"blk.N.{s}.weight" for s in _SINGLE_SLOTS)
    types: dict[str, int] = {}
    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name in ("token_embd.weight", "output.weight"):
            types[name] = t.ggml_type
        elif name.startswith("blk."):
            suffix = name.split(".", 2)[2]
            if suffix in {f"{s}.weight" for slots in _GROUP_SLOTS.values() for s in slots} | {
                f"{s}.weight" for s in _SINGLE_SLOTS
            }:
                types[name] = t.ggml_type
    return types


def parse_gguf_config(shim: "GgufConfigShim") -> "ModelConfig":
    m = shim.metadata
    prefix = shim.model_type  # "qwen35"

    def g(key: str, default=None):
        return m.get(f"{prefix}.{key}", default)

    # block_count includes the trailing MTP/nextn draft block(s) (blk.N.nextn.*);
    # the trunk is what llama.cpp's n_layer() runs
    num_layers = int(g("block_count")) - int(g("nextn_predict_layers", 0))
    interval = int(g("full_attention_interval", 4))
    layer_types = [
        "full_attention" if _is_full_attention(i, interval) else "linear_attention"
        for i in range(num_layers)
    ]
    head_dim = int(g("attention.key_length"))
    rotary_dim = int(g("rope.dimension_count", head_dim))
    num_v_heads = int(g("ssm.time_step_rank"))

    hf_like = SimpleNamespace(
        num_hidden_layers=num_layers,
        num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")),
        head_dim=head_dim,
        hidden_size=int(g("embedding_length")),
        intermediate_size=int(g("feed_forward_length", 0)),
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon", 1e-6)),
        max_position_embeddings=int(g("context_length", 262144)),
        rope_theta=float(g("rope.freq_base", 10_000_000.0)),
        partial_rotary_factor=rotary_dim / head_dim,
        rope_parameters=None,
        rope_scaling=None,
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        layer_types=layer_types,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        norm_topk_prob=False,
        linear_num_key_heads=int(g("ssm.group_count")),
        linear_num_value_heads=num_v_heads,
        linear_key_head_dim=int(g("ssm.state_size")),
        linear_value_head_dim=int(g("ssm.inner_size")) // num_v_heads,
        linear_conv_kernel_dim=int(g("ssm.conv_kernel")),
        quantization_config=None,
        model_type="qwen3_5",
        architectures=["Qwen35DenseGGUFForCausalLM"],
        torch_dtype="bfloat16",
    )
    config = parse_config(hf_like)
    # packed-GGUF marker + per-tensor type map (see maybe_convert below)
    types = _tensor_types(shim.model_path)
    if _v_reorder(m, prefix) is not None:
        # the iterator dequant->un-reorders->repacks every ssm_out as Q8_0
        for k2 in list(types):
            if k2.endswith(".ssm_out.weight"):
                types[k2] = GGML_Q8_0
    object.__setattr__(config, "gguf_types_full", types)
    return config


def _to_bf16(t):
    return dequant_any(t).to(torch.bfloat16)


def _v_reorder(meta: dict, prefix: str) -> dict | None:
    """Standard qwen35 GGUFs store linear-attention V heads in *tiled* order
    ([k0v0, k1v0, ..., k0v1, ...]) so ggml can broadcast K heads cheaply; the
    llama.cpp converter applies this reorder on HF->GGUF conversion
    (conversion/qwen.py::_LinearAttentionVReorderBase) whenever
    num_v_heads != num_k_heads.  FreeToken's fla kernels expect the HF
    *grouped* order (V head j pairs with K head j//ratio), so every
    V-head-indexed tensor must be un-reordered at load.  Returns None when
    nv == nk (reorder is a no-op and standard files store grouped == tiled).
    """
    nk = int(meta.get(f"{prefix}.ssm.group_count", 0))
    nv = int(meta.get(f"{prefix}.ssm.time_step_rank", 0))
    if not nk or not nv or nv == nk:
        return None
    ratio = nv // nk
    assert nv % nk == 0, f"nv {nv} not divisible by nk {nk}"
    # tiled position of grouped head i: g(i) = (i % ratio) * nk + i // ratio
    # (verified against the HF checkpoint: dt_bias/A_log/alpha/beta match only
    # under this map; quant noise ~6e-4)
    head = [(i % ratio) * nk + (i // ratio) for i in range(nv)]
    return {
        "nv": nv,
        "head": head,  # per-v-head gather (tiled -> grouped)
    }


def _gather_rows(t: torch.Tensor, head: list[int], head_rows: int) -> torch.Tensor:
    """Reorder whole v heads along dim 0; head_rows = rows per head (hd, or 1)."""
    perm = [h * head_rows + d for h in head for d in range(head_rows)]
    return t.index_select(0, torch.tensor(perm, dtype=torch.long))


def _requant_q8_0(w_bf16: torch.Tensor) -> torch.Tensor:
    """Dequantized [out, in] bf16 -> Q8_0 packed [out, row_bytes] uint8."""
    import gguf
    from gguf.quants import quantize

    q = quantize(w_bf16.float().numpy(), gguf.GGMLQuantizationType.Q8_0)
    return torch.from_numpy(q)


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Every dense-model weight: quantized tensors stay packed (``.qweight``),
    the rest dequantize to bf16 (fp32 for ``dt_bias``/``A_log``)."""
    # include_moe_experts may be requested unconditionally by the loader; this
    # family has no routed experts, so the flag is vacuously satisfied.
    assert include_non_moe

    meta = load_gguf_metadata(model_path)
    prefix = meta.get("general.architecture", "qwen35")
    interval = int(meta.get(f"{prefix}.full_attention_interval", 4))
    num_layers = int(meta.get(f"{prefix}.block_count")) - int(
        meta.get(f"{prefix}.nextn_predict_layers", 0)
    )
    types = _tensor_types(model_path)
    vre = _v_reorder(meta, prefix)  # standard GGUF tiled-V -> grouped (None if nv==nk)
    hd_v = int(meta.get(f"{prefix}.ssm.inner_size", 0)) // int(
        meta.get(f"{prefix}.ssm.time_step_rank", 1)
    )
    kf = int(meta.get(f"{prefix}.ssm.group_count", 0)) * int(
        meta.get(f"{prefix}.ssm.state_size", 128)
    )

    def type_of(name: str) -> int:
        return types.get(name, GGML_F32)

    # per-layer fused groups: (layer, module_stem, group) -> {slot: tensor}
    fuse: dict[tuple[int, str, str], dict[str, torch.Tensor]] = {}

    def slots_for(layer: int, group: str) -> tuple[str, str] | None:
        """(gguf suffix stem, module stem) for this layer's fused group."""
        if group == "gate_up":
            return "ffn", "mlp.gate_up_proj"
        if _is_full_attention(layer, interval):
            return group, "self_attn.qkv_proj" if group == "qkv" else None
        return "in_proj", "linear_attn.in_proj" if group == "in_proj" else None

    def slot_names(group: str) -> tuple[str, ...]:
        if group == "qkv":
            return ("q", "k", "v")
        if group == "in_proj":
            return ("qkv", "z", "b", "a")
        return ("gate", "up")

    def gguf_slots(group: str) -> tuple[str, ...]:
        return _GROUP_SLOTS["qkv" if group == "qkv" else "in_proj" if group == "in_proj" else "gate_up"]

    def feed_group(layer: int, group: str, gguf_slot: str, val: torch.Tensor):
        mapped = slots_for(layer, group)
        assert mapped is not None, f"tensor {gguf_slot} in wrong layer kind (layer {layer})"
        gguf_stem, module_stem = mapped
        gs = gguf_slots(group)
        names = slot_names(group)
        key = (layer, module_stem, group)
        buf = fuse.setdefault(key, {})
        buf[gguf_slot] = val
        if len(buf) < len(gs):
            return
        del fuse[key]
        stem = f"model.layers.{layer}.{module_stem}"
        if all(type_of(f"blk.{layer}.{s}.weight") not in _UNQUANTIZED for s in gs):
            for gslot, name in zip(gs, names):
                yield f"{stem}.{name}.qweight", buf[gslot]
        else:
            yield stem + ".weight", torch.cat([buf[s] for s in gs], dim=0).contiguous()

    for t in iter_gguf_tensors(model_path):
        name = t.name
        quantized = t.ggml_type not in _UNQUANTIZED
        if name == "token_embd.weight":
            yield ("model.embed_tokens.qweight" if quantized else "model.embed_tokens.weight"), (
                t.packed() if quantized else dequant_any(t)
            )
        elif name == "output_norm.weight":
            # (1+weight) is already baked into the GGUF norm weights at conversion
            yield "model.norm.weight", _to_bf16(t)
        elif name == "output.weight":
            yield ("lm_head.qweight" if quantized else "lm_head.weight"), (
                t.packed() if quantized else dequant_any(t)
            )
        elif name.startswith("blk."):
            layer = int(name.split(".")[1])
            if layer >= num_layers:
                continue  # trailing MTP/nextn draft block
            suffix = name.split(".", 2)[2]
            stem = f"model.layers.{layer}."
            if suffix == "ssm_a":
                # stored as -exp(A_log); FreeToken keeps A_log (fp32)
                a = dequant_any(t).to(torch.float32)
                assert (a < 0).all(), f"{name}: expected -exp(A_log) (negative values)"
                a = torch.log(-a)
                if vre is not None:
                    a = a[vre["head"]]
                yield stem + "linear_attn.A_log", a
                continue
            if suffix == "ssm_dt.bias":
                dt = dequant_any(t).to(torch.float32)
                if vre is not None:
                    dt = dt[vre["head"]]
                yield stem + "linear_attn.dt_bias", dt
                continue
            if suffix == "ssm_conv1d.weight":
                # ggml [K, conv_dim] -> module [conv_dim, 1, K]; un-reorder V channels
                w = _to_bf16(t)
                if vre is not None:
                    w = torch.cat(
                        [w[:2 * kf], _gather_rows(w[2 * kf:], vre["head"], hd_v)], dim=0
                    )
                yield stem + "linear_attn.conv1d.weight", w.unsqueeze(1).contiguous()
                continue
            proj = suffix.rsplit(".weight", 1)[0]
            for group, gs in _GROUP_SLOTS.items():
                if proj in gs:
                    val = t.packed() if quantized else dequant_any(t)
                    if group == "in_proj" and vre is not None:
                        # standard GGUF tiled V order -> grouped (see _v_reorder);
                        # packed rows are per-output-row quant blocks, so whole-head
                        # row gathers are exact for every quant type
                        if proj == "attn_qkv":
                            val = torch.cat(
                                [val[:2 * kf], _gather_rows(val[2 * kf:], vre["head"], hd_v)],
                                dim=0,
                            )
                        elif proj == "attn_gate":
                            val = _gather_rows(val, vre["head"], hd_v)
                        else:  # ssm_beta / ssm_alpha: one row per V head
                            val = _gather_rows(val, vre["head"], 1)
                    yield from feed_group(layer, group, proj, val)
                    break
            else:
                if proj in _SINGLE_SLOTS:
                    rel = {"attn_output": "self_attn.o_proj", "ssm_out": "linear_attn.out_proj",
                           "ffn_down": "mlp.down_proj"}[proj]
                    if proj == "ssm_out" and vre is not None:
                        # out_proj INPUT columns are V-head indexed (tiled in GGUF):
                        # dequant, un-reorder columns, repack as Q8_0 (K-quant block
                        # boundaries do not align with 128-row heads, so packed
                        # column gathers are not possible; Q8_0 keeps it packed)
                        w = dequant_any(t).to(torch.bfloat16)
                        col = [h * hd_v + d for h in vre["head"] for d in range(hd_v)]
                        w = w.index_select(1, torch.tensor(col, dtype=torch.long))
                        if quantized:
                            yield f"{stem}{rel}.qweight", _requant_q8_0(w)
                        else:
                            yield f"{stem}{rel}.weight", w
                        continue
                    if quantized:
                        yield f"{stem}{rel}.qweight", t.packed()
                    else:
                        yield f"{stem}{rel}.weight", _to_bf16(t)
                    continue
                rel = _SUFFIX_MAP.get(suffix)
                if rel is not None:
                    # (1+weight) already baked at GGUF conversion (verified vs HF:
                    # q_norm HF mean 0.23 -> GGUF 1.23); ssm_norm is plain weight*x
                    yield stem + rel, _to_bf16(t)

    leftovers = sorted(fuse)
    assert not leftovers, f"incomplete fused groups: {leftovers}"


class Qwen35GGUFLMHead(BaseOP):
    """Untied GGUF LM head over a packed ``output.weight``.

    Engine contract: ``model.forward()`` returns ``[bs, vocab]`` -- slice the
    last-position hidden on prefill (mirrors ParallelLMHead / GGUFTiedLMHead /
    Nvfp4LMHead) before the fused GGUF GEMV."""

    def __init__(self, num_embeddings: int, embedding_dim: int, quant_type: int):
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )
        self._quant_type = quant_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx
        from freetoken.layers.gguf import fused_mul_mat_gguf

        batch = get_global_ctx().batch
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        return fused_mul_mat_gguf(x, self.qweight, self._quant_type)


def maybe_convert_qwen35_dense_to_gguf(model, config: "ModelConfig") -> None:
    """Swap the dense projections for packed GGUF ops (GGUF loads only)."""
    types = getattr(config, "gguf_types_full", None)
    if not types:
        return
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear, GgufColSplits

    def qt(name: str) -> int | None:
        t = types.get(name)
        return t if t is not None and t not in _UNQUANTIZED else None

    inner = model.model
    H = config.hidden_size
    I = config.intermediate_size
    qo2 = config.num_qo_heads * config.head_dim * 2
    kv = config.num_kv_heads * config.head_dim
    lg = config.linear_attention_group()
    conv_dim = 2 * (lg.num_key_heads * lg.key_head_dim) + lg.num_value_heads * lg.value_head_dim
    value_dim = lg.num_value_heads * lg.value_head_dim
    nvh = lg.num_value_heads

    for lid, layer in enumerate(inner.layers.op_list):
        def swap_group(owner, attr, slots):
            """slots: [(module_slot, gguf_suffix)] -> GgufColSplits when all packed."""
            parts = []
            for slot, suffix in slots:
                t = qt(f"blk.{lid}.{suffix}.weight")
                if t is None:
                    return  # any unquantized slot keeps the dense bf16 module
                dim = {"q": qo2, "k": kv, "v": kv, "qkv": conv_dim, "z": value_dim,
                       "b": nvh, "a": nvh, "gate": I, "up": I}[slot]
                parts.append((slot, dim, t))
            setattr(owner, attr, GgufColSplits(H, parts))

        def swap_single(owner, attr, suffix, in_features, out_features):
            t = qt(f"blk.{lid}.{suffix}.weight")
            if t is None:
                return
            setattr(owner, attr, GGUFLinear(in_features, out_features, t))

        if config.is_linear_layer(lid):
            swap_group(layer.linear_attn, "in_proj",
                       [("qkv", "attn_qkv"), ("z", "attn_gate"),
                        ("b", "ssm_beta"), ("a", "ssm_alpha")])
            swap_single(layer.linear_attn, "out_proj", "ssm_out", value_dim, H)
        else:
            swap_group(layer.self_attn, "qkv_proj",
                       [("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v")])
            swap_single(layer.self_attn, "o_proj", "attn_output",
                        config.num_qo_heads * config.head_dim, H)
        swap_group(layer.mlp, "gate_up_proj", [("gate", "ffn_gate"), ("up", "ffn_up")])
        swap_single(layer.mlp, "down_proj", "ffn_down", I, H)

    embed_type = qt("token_embd.weight")
    if embed_type is not None:
        inner.embed_tokens = GGUFEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=H,
            quant_type=embed_type,
        )
        if config.tie_word_embeddings:
            from freetoken.models.gemma4.gguf import GGUFTiedLMHead

            model.lm_head = GGUFTiedLMHead(inner.embed_tokens, embed_type)

    output_type = qt("output.weight")
    if output_type is not None and not config.tie_word_embeddings:
        model.lm_head = Qwen35GGUFLMHead(config.vocab_size, H, output_type)


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "maybe_convert_qwen35_dense_to_gguf",
    "Qwen3_5MoEForCausalLM",
]
