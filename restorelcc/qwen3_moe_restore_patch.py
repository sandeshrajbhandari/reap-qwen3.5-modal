"""
Monkey-patch HuggingFace Qwen3 MoE attention to match RestoreLCC's injection points
(attn_v, attn_b, main_comp) from the vendored Llama implementation.
Call patch_qwen3_moe_for_restorelcc() once before Qwen3MoeForCausalLM.from_pretrained.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import List, Optional

import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


def patch_qwen3_moe_for_restorelcc() -> None:
    if getattr(Qwen3MoeAttention, "_restorelcc_patched", False):
        return

    _orig_init = Qwen3MoeAttention.__init__
    _orig_forward = Qwen3MoeAttention.forward

    def __init__(self, config, layer_idx: int):
        _orig_init(self, config, layer_idx)
        nh = config.num_attention_heads
        hd = self.head_dim
        self.attn_v = nn.ParameterList(
            [nn.Parameter(torch.zeros(hd), requires_grad=True) for _ in range(nh)]
        )
        self.attn_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(hd), requires_grad=True) for _ in range(nh)]
        )
        self.main_comp = nn.ParameterList(
            [nn.Parameter(torch.zeros(hd, hd), requires_grad=False) for _ in range(nh)]
        )
        self.applied_module: Optional[str] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        if self.applied_module == "attention":
            B, T, _ = attn_output.shape
            H, D = self.config.num_attention_heads, self.head_dim
            v = torch.stack(list(self.attn_v), dim=0)
            b = torch.stack(list(self.attn_b), dim=0)
            comp = torch.stack(list(self.main_comp), dim=0)
            out = attn_output.view(B, T, H, D)
            added_info = torch.bmm(comp.transpose(1, 2), v.unsqueeze(2)).squeeze(2)
            b = b.view(1, 1, H, D)
            add = added_info.view(1, 1, H, D)
            attn_output = (out + add + b).view(B, T, H * D)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    Qwen3MoeAttention.__init__ = __init__  # type: ignore[method-assign]
    Qwen3MoeAttention.forward = forward  # type: ignore[method-assign]
    Qwen3MoeAttention._restorelcc_patched = True  # type: ignore[attr-defined]


def set_qwen3_moe_applied_modules(model, applied_module: str, applied_layers: Optional[List[int]] = None) -> None:
    layers = model.model.layers
    if applied_layers is None:
        applied_layers = list(range(len(layers)))
    for idx in applied_layers:
        attn = layers[idx].self_attn
        if hasattr(attn, "applied_module"):
            attn.applied_module = applied_module


def qwen3_moe_custom_from_pretrained(
    model_cls,
    pretrained_model_name_or_path,
    *,
    applied_module: str = "attention",
    applied_layers: Optional[List[int]] = None,
    torch_dtype=None,
    **kwargs,
):
    patch_qwen3_moe_for_restorelcc()
    model = model_cls.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype, **kwargs)
    set_qwen3_moe_applied_modules(model, applied_module, applied_layers)
    return model
