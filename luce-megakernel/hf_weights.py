"""Load merged HF safetensors when `AutoModel.state_dict()` omits text weights (key layout mismatch)."""

from __future__ import annotations

import json
from typing import Dict

import torch
from huggingface_hub import hf_hub_download


def merge_safetensors_shards(model_name: str) -> Dict[str, torch.Tensor]:
    try:
        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
    except Exception:
        path = hf_hub_download(model_name, "model.safetensors")
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError("pip install safetensors") from e
        return load_file(path, device="cpu")
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shards: Dict[str, list] = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)
    merged: Dict[str, torch.Tensor] = {}
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError("pip install safetensors") from e
    for shard in sorted(shards.keys()):
        path = hf_hub_download(model_name, shard)
        merged.update(load_file(path, device="cpu"))
    return merged


def strip_to_language_state(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    lm = "model.language_model."
    if any(k.startswith(lm) for k in raw):
        return {k[len(lm) :]: v for k, v in raw.items() if k.startswith(lm)}
    m = "model."
    if any(k.startswith("model.layers.") for k in raw):
        return {k[len(m) :]: v for k, v in raw.items() if k.startswith(m)}
    return raw
