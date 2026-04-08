"""Load Intel AutoRound checkpoints (auto_round:auto_gptq int4) into CUDA tensors for the megakernel."""

import json
import struct
from typing import Dict, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from model import (
    DN_CONV_CHANNELS,
    FA_QPROJ_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    LAYER_TYPE,
    NUM_LAYERS,
    VOCAB_SIZE,
)


def _merge_shards(model_name: str) -> Dict[str, torch.Tensor]:
    index_path = hf_hub_download(model_name, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shards: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)
    merged: Dict[str, torch.Tensor] = {}
    for shard in sorted(shards.keys()):
        path = hf_hub_download(model_name, shard)
        part = load_file(path, device="cpu")
        merged.update(part)
    return merged


def _q4(state: Dict[str, torch.Tensor], prefix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qw = state[prefix + ".qweight"].to(device="cuda", dtype=torch.int32).contiguous()
    sc = state[prefix + ".scales"].to(device="cuda", dtype=torch.float16).contiguous()
    qz = state[prefix + ".qzeros"].to(device="cuda", dtype=torch.int32).contiguous()
    return qw, sc, qz


def _pack_q4(buf: bytearray, off: int, qw: torch.Tensor, sc: torch.Tensor, qz: torch.Tensor) -> None:
    struct.pack_into("Q", buf, off, qw.data_ptr())
    struct.pack_into("Q", buf, off + 8, sc.data_ptr())
    struct.pack_into("Q", buf, off + 16, qz.data_ptr())


def pack_layer_weights_w4(layer_entries: list) -> torch.Tensor:
    """256 bytes/layer: 16-byte header + 240-byte union (matches kernel LayerWeightsW4)."""
    struct_size = 256
    buf = bytearray(NUM_LAYERS * struct_size)
    for i, ent in enumerate(layer_entries):
        base = i * struct_size
        struct.pack_into("iiii", buf, base, ent["type"], 0, 0, 0)
        u = base + 16
        if ent["type"] == 0:
            o = u
            struct.pack_into("Q", buf, o, ent["input_ln"].data_ptr())
            o += 8
            for key in ("qkv", "z", "beta", "alpha"):
                _pack_q4(buf, o, *ent[key])
                o += 24
            for tname in ("conv", "a_log", "dt_bias", "norm"):
                struct.pack_into("Q", buf, o, ent[tname].data_ptr())
                o += 8
            _pack_q4(buf, o, *ent["out"])
            o += 24
            struct.pack_into("Q", buf, o, ent["post_ln"].data_ptr())
            o += 8
            for key in ("gate", "up", "down"):
                _pack_q4(buf, o, *ent[key])
                o += 24
            if o - u != 240:
                raise RuntimeError(f"Delta W4 pack size {o - u} != 240")
        else:
            o = u
            struct.pack_into("Q", buf, o, ent["input_ln"].data_ptr())
            o += 8
            for key in ("q", "k", "v"):
                _pack_q4(buf, o, *ent[key])
                o += 24
            struct.pack_into("Q", buf, o, ent["q_norm"].data_ptr())
            o += 8
            struct.pack_into("Q", buf, o, ent["k_norm"].data_ptr())
            o += 8
            _pack_q4(buf, o, *ent["o"])
            o += 24
            struct.pack_into("Q", buf, o, ent["post_ln"].data_ptr())
            o += 8
            for key in ("gate", "up", "down"):
                _pack_q4(buf, o, *ent[key])
                o += 24
            struct.pack_into("40s", buf, o, b"\x00" * 40)
            o += 40
            if o - u != 240:
                raise RuntimeError(f"Full-attn W4 pack size {o - u} != 240")
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def load_weights_autoround(
    model_name: str = "Intel/Qwen3.5-9B-int4-AutoRound",
    verbose: bool = True,
):
    """Load quantized language-model weights; norms / conv / embed / lm_head stay bf16."""
    if verbose:
        print(f"Loading {model_name} (AutoRound int4 + bf16 aux)...")

    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    lt_cfg = text_cfg.layer_types
    lt = [0 if x == "linear_attention" else 1 for x in lt_cfg]
    if len(lt) != NUM_LAYERS or lt != LAYER_TYPE:
        raise ValueError("layer_types mismatch vs megakernel LAYER_TYPE")

    qconf = getattr(cfg, "quantization_config", None)
    if qconf is None:
        qconf = {}
    if isinstance(qconf, dict):
        fmt = qconf.get("packing_format", "")
    else:
        fmt = getattr(qconf, "packing_format", "")
    if "auto_gptq" not in str(fmt) and "auto-round" not in str(qconf):
        pass  # Intel checkpoint uses this format; allow load anyway

    raw = _merge_shards(model_name)
    prefix = "model.language_model."
    state = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    lm_head = raw.get("lm_head.weight")
    if lm_head is None:
        raise KeyError("lm_head.weight not found in checkpoint")
    lm_head = lm_head.cuda().contiguous().to(dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    layer_entries = []
    for i in range(NUM_LAYERS):
        p = f"layers.{i}."
        if lt[i] == 0:
            layer_entries.append({
                "type": 0,
                "input_ln": state[p + "input_layernorm.weight"].cuda().contiguous(),
                "qkv": _q4(state, p + "linear_attn.in_proj_qkv"),
                "z": _q4(state, p + "linear_attn.in_proj_z"),
                "beta": _q4(state, p + "linear_attn.in_proj_b"),
                "alpha": _q4(state, p + "linear_attn.in_proj_a"),
                "conv": state[p + "linear_attn.conv1d.weight"].cuda().contiguous(),
                "a_log": state[p + "linear_attn.A_log"].cuda().contiguous(),
                "dt_bias": state[p + "linear_attn.dt_bias"].cuda().contiguous(),
                "norm": state[p + "linear_attn.norm.weight"].cuda().contiguous(),
                "out": _q4(state, p + "linear_attn.out_proj"),
                "post_ln": state[p + "post_attention_layernorm.weight"].cuda().contiguous(),
                "gate": _q4(state, p + "mlp.gate_proj"),
                "up": _q4(state, p + "mlp.up_proj"),
                "down": _q4(state, p + "mlp.down_proj"),
            })
        else:
            layer_entries.append({
                "type": 1,
                "input_ln": state[p + "input_layernorm.weight"].cuda().contiguous(),
                "q": _q4(state, p + "self_attn.q_proj"),
                "k": _q4(state, p + "self_attn.k_proj"),
                "v": _q4(state, p + "self_attn.v_proj"),
                "q_norm": state[p + "self_attn.q_norm.weight"].cuda().contiguous(),
                "k_norm": state[p + "self_attn.k_norm.weight"].cuda().contiguous(),
                "o": _q4(state, p + "self_attn.o_proj"),
                "post_ln": state[p + "post_attention_layernorm.weight"].cuda().contiguous(),
                "gate": _q4(state, p + "mlp.gate_proj"),
                "up": _q4(state, p + "mlp.up_proj"),
                "down": _q4(state, p + "mlp.down_proj"),
            })

    embed = state["embed_tokens.weight"].cuda().contiguous()
    final_norm = state["norm.weight"].cuda().contiguous()

    packed = pack_layer_weights_w4(layer_entries)

    if verbose:
        qb = 0
        for ent in layer_entries:
            for key in (
                "qkv", "z", "beta", "alpha", "out", "gate", "up", "down",
                "q", "k", "v", "o",
            ):
                if key not in ent:
                    continue
                qw, sc, qz = ent[key]
                qb += qw.numel() * 4 + sc.numel() * 2 + qz.numel() * 4
        print(f"Int4 tensors (packed): ~{qb / 1e6:.0f} MB; embed+lm_head remain bf16")

    weights = {
        "weight_mode": "autoround_int4",
        "embed_weight": embed,
        "final_norm_weight": final_norm,
        "lm_head_weight": lm_head,
        "layer_weights_packed": packed,
        "layer_entries": layer_entries,
    }
    return weights, tokenizer


def max_dequant_bf16_elems() -> int:
    """Largest row-major dequant buffer for prefill (out * in)."""
    cands = [
        INTERMEDIATE_SIZE * HIDDEN_SIZE,
        HIDDEN_SIZE * INTERMEDIATE_SIZE,
        DN_CONV_CHANNELS * HIDDEN_SIZE,
        FA_QPROJ_SIZE * HIDDEN_SIZE,
        VOCAB_SIZE * HIDDEN_SIZE,
    ]
    return max(cands)
