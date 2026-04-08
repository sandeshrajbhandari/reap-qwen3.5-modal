"""Padding utilities for batched offline KD (variable-length teacher samples)."""

from __future__ import annotations

import torch


def collate_offline_kd_batch(samples: list[dict], pad_id: int) -> dict:
    """
    Stack samples saved by phase1 (per-sample dicts). Pads to max length in batch.

    Each sample keys: input_ids (L,), hidden_states (L, H), topk_values (L-1, K),
    topk_indices (L-1, K) — topk tensors may be empty when L < 2.
    """
    lengths = [int(s["input_ids"].shape[0]) for s in samples]
    bsz, max_len = len(lengths), max(lengths)
    h_dim = samples[0]["hidden_states"].shape[-1]
    ks = [int(s["topk_values"].shape[-1]) for s in samples if s["topk_values"].numel() > 0]
    k = max(ks) if ks else 0

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    hidden = torch.zeros((bsz, max_len, h_dim), dtype=torch.float16)

    max_lm1 = max(0, max_len - 1)
    topk_values = torch.zeros((bsz, max_lm1, k), dtype=torch.float16)
    topk_indices = torch.zeros((bsz, max_lm1, k), dtype=torch.long)
    kd_mask = torch.zeros((bsz, max_lm1), dtype=torch.bool)

    for i, s in enumerate(samples):
        L = lengths[i]
        input_ids[i, :L] = s["input_ids"]
        attention_mask[i, :L] = True
        hidden[i, :L] = s["hidden_states"]
        if L > 1:
            t = L - 1
            topk_values[i, :t] = s["topk_values"]
            topk_indices[i, :t] = s["topk_indices"].long()
            kd_mask[i, :t] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "teacher_hidden": hidden,
        "topk_values": topk_values,
        "topk_indices": topk_indices,
        "kd_mask": kd_mask,
    }
