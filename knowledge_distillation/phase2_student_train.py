#!/usr/bin/env python3
"""
Phase 2 — Student recovery: load pruned student + LoRA, train from offline .pt shards
using CE + hidden MSE + KL vs sparse teacher logits (top-K reconstruction).

Example:
  python phase2_student_train.py \\
    --student-model-path /path/to/pruned-18b \\
    --teacher-data-dir ./teacher_data \\
    --output-dir ./healed_student \\
    --epochs 2 \\
    --batch-size 2
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from collate import collate_offline_kd_batch


class OfflineDistillationDataset(Dataset):
    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(str(Path(data_dir) / "sample_*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        return torch.load(self.files[idx], map_location="cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student-model-path", type=str, required=True)
    p.add_argument("--teacher-data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./healed_student")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha-ce", type=float, default=0.1)
    p.add_argument("--beta-kd", type=float, default=0.4)
    p.add_argument("--gamma-hidden", type=float, default=0.5)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module name substrings for LoRA",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase 2 expects a CUDA GPU.")

    tok = AutoTokenizer.from_pretrained(
        args.student_model_path, trust_remote_code=args.trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model_path,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()

    targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        bias="none",
    )
    student = get_peft_model(student, peft_config)
    student.print_trainable_parameters()

    voc = student.config.vocab_size
    dataset = OfflineDistillationDataset(args.teacher_data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_offline_kd_batch(b, pad_id),
        num_workers=0,
    )

    opt = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad), lr=args.lr
    )

    student.train()
    T = args.temperature
    neg_large = -1e9

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            teacher_h = batch["teacher_hidden"].to(device).float()
            topk_v = batch["topk_values"].to(device).float()
            topk_i = batch["topk_indices"].to(device).long()
            kd_mask = batch["kd_mask"].to(device)

            opt.zero_grad(set_to_none=True)
            out = student(
                input_ids=input_ids,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
            )
            s_logits = out.logits.float()
            s_h = out.hidden_states[-1].float()

            bsz, seq, _ = s_logits.shape
            # Causal CE on non-pad tokens; predict next token from positions 0..seq-2
            labels = input_ids.clone()
            labels[~attn] = -100
            ce = F.cross_entropy(
                s_logits[:, :-1].reshape(-1, voc),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )

            valid_h = attn.unsqueeze(-1).float()
            denom = valid_h.sum() * s_h.size(-1) + 1e-8
            hidden_loss = ((s_h - teacher_h).pow(2) * valid_h).sum() / denom

            # KD: sparse teacher logits from top-K only at positions covered by kd_mask
            kd_denom = kd_mask.sum()
            if kd_denom.item() == 0 or topk_i.shape[-1] == 0:
                kd_loss = torch.zeros((), device=device, dtype=s_logits.dtype)
            else:
                sparse_teacher = torch.full(
                    (bsz, seq - 1, voc), neg_large, device=device, dtype=s_logits.dtype
                )
                sparse_teacher.scatter_(dim=-1, index=topk_i, src=topk_v.to(s_logits.dtype))
                s_lp = F.log_softmax(s_logits[:, :-1] / T, dim=-1)
                t_p = F.softmax(sparse_teacher / T, dim=-1)
                kd_elem = F.kl_div(s_lp, t_p, reduction="none").sum(dim=-1)
                kd_loss = (kd_elem * kd_mask).sum() / (kd_denom + 1e-8)
                kd_loss = kd_loss * (T * T)

            loss = (
                args.alpha_ce * ce
                + args.beta_kd * kd_loss
                + args.gamma_hidden * hidden_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in student.parameters() if p.requires_grad),
                args.max_grad_norm,
            )
            opt.step()

            pbar.set_postfix(
                loss=float(loss.item()),
                ce=float(ce.item()),
                kd=float(kd_loss.item()),
                hid=float(hidden_loss.item()),
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved adapter + tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
