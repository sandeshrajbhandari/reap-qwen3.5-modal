#!/usr/bin/env python3
"""
RestoreLCC-style recovery for Qwen3 MoE (e.g. Qwen3.5-35B-A3B pruned ~18B active).

Steps:
  1) (Optional) Build Alpaca probing packs under ./processed_data/
  2) Contrastive probing + SVD: utils.components.obtain_main_vecs (writes .pt directions + .npy head order)
  3) Load pruned model with patched attention, init main_comp from directions, train attn_v/attn_b on Alpaca

Run from the restorelcc/ directory (or set PYTHONPATH).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3MoeForCausalLM,
    TrainingArguments,
    logging,
    set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from qwen3_moe_restore_patch import qwen3_moe_custom_from_pretrained
from utils.components import obtain_main_vecs
from utils.dataloaders import obtain_sepc_datasets
from utils.trainers import CustomSFTTrainer


def _ensure_alpaca_processed(num_train: int, num_val: int, out_root: Path, seed: int = 42) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    train_path = out_root / "train.pt"
    val_path = out_root / "valid.pt"
    if train_path.exists() and val_path.exists():
        return

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    n = min(len(ds), num_train + num_val)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    train_idx = indices[:num_train]
    val_idx = indices[num_train : num_train + num_val]

    def row_to_item(ex):
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        if inp:
            text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            text = f"### Instruction:\n{instr}\n\n### Response:\n"
        label = " " + out if out and not out.startswith(" ") else out
        return {"text": text, "label": label}

    train_items = [row_to_item(ds[i]) for i in train_idx]
    val_items = [row_to_item(ds[i]) for i in val_idx]
    torch.save(train_items, train_path)
    torch.save(val_items, val_path)
    print(f"Wrote {train_path} ({len(train_items)}), {val_path} ({len(val_items)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-model-id", type=str, default="Qwen/Qwen3.5-35B-A3B")
    p.add_argument("--pruned-model-path", type=str, required=True, help="HF id or local path to pruned MoE checkpoint")
    p.add_argument("--spec-task", type=str, default="alpaca")
    p.add_argument("--num-train-samples", type=int, default=2000)
    p.add_argument("--num-val-samples", type=int, default=256)
    p.add_argument("--use-topk-heads", type=int, default=256)
    p.add_argument("--probe-only", action="store_true", help="Only run obtain_main_vecs and exit")
    p.add_argument("--skip-probe", action="store_true", help="Reuse existing main_comp .pt and head .npy")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-epoch", type=int, default=3)
    p.add_argument("--train-batch", type=int, default=1)
    p.add_argument("--eval-batch", type=int, default=1)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--output-dir", type=str, default="./restorelcc_qwen3_moe_out")
    p.add_argument("--hf-cache-dir", type=str, default=os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--l1-lambda", type=float, default=0.0)
    p.add_argument("--run-mode", type=str, default="train", choices=["train", "train_wandb"])
    p.add_argument("--applied-layers", type=str, default=None, help="Comma-separated layer indices; default all")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import wandb

    if args.run_mode == "train_wandb":
        wandb.init(mode="online", name=Path(args.output_dir).name)
    else:
        wandb.init(mode="disabled")

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.set_verbosity_error()

    processed_dir = root / "processed_data" / f"{args.spec_task}_{args.num_train_samples}"
    if args.spec_task == "alpaca":
        _ensure_alpaca_processed(args.num_train_samples, args.num_val_samples, processed_dir, args.seed)
    else:
        raise SystemExit(f"spec-task {args.spec_task!r} not wired; add data under {processed_dir} or extend script.")

    slug = f"qwen3moe_{args.reference_model_id.replace('/', '_')}_{args.spec_task}_{args.num_train_samples}"
    comp_dir = root / "processed_data2" / "main_comps"
    comp_dir.mkdir(parents=True, exist_ok=True)
    comp_path = comp_dir / f"{slug}.pt"
    head_score_path = comp_dir / f"{slug}.npy"

    if not args.skip_probe:
        obtain_main_vecs(
            args.spec_task,
            args.num_train_samples,
            original_model_path=args.reference_model_id,
            pruned_model_path=args.pruned_model_path,
            comp_path=str(comp_path),
            score_path=str(head_score_path),
        )
    if args.probe_only:
        print("probe-only: done.")
        return

    main_comps = torch.load(comp_path, map_location="cpu")
    top_heads = np.load(head_score_path)[: args.use_topk_heads, :]
    lofit_heads = list(zip(top_heads[:, 0].astype(int), top_heads[:, 1].astype(int)))
    print(f"Using {len(lofit_heads)} heads from probing.")

    tokenizer = AutoTokenizer.from_pretrained(args.pruned_model_path, cache_dir=args.hf_cache_dir, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    applied_layers = None
    if args.applied_layers is not None:
        applied_layers = [int(x.strip()) for x in args.applied_layers.split(",")]

    torch_dtype = torch.bfloat16
    model = qwen3_moe_custom_from_pretrained(
        Qwen3MoeForCausalLM,
        args.pruned_model_path,
        applied_module="attention",
        applied_layers=applied_layers,
        torch_dtype=torch_dtype,
        cache_dir=args.hf_cache_dir,
        trust_remote_code=True,
        device_map="auto",
    )

    head_num = model.config.num_attention_heads
    for par in model.parameters():
        par.requires_grad = False

    for i in range(model.config.num_hidden_layers):
        attn = model.model.layers[i].self_attn
        for j, module in enumerate(attn.attn_v):
            if (i, j) in lofit_heads:
                module.requires_grad = True
        for j, module in enumerate(attn.attn_b):
            if (i, j) in lofit_heads:
                module.requires_grad = True
        for co, module in enumerate(attn.main_comp):
            if (i, co) in lofit_heads:
                attn_idx = i * head_num + co
                w = main_comps[attn_idx].to(dtype=torch.bfloat16, device=module.device)
                module.data.copy_(w)

    train_path = processed_dir / "train.pt"
    val_path = processed_dir / "valid.pt"
    train_ds = obtain_sepc_datasets(str(train_path))
    val_ds = obtain_sepc_datasets(str(val_path))

    response_template = "\n\n### Response:"
    rids = tokenizer.encode(response_template, add_special_tokens=False)
    data_collator = DataCollatorForCompletionOnlyLM(rids, tokenizer=tokenizer)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out),
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        num_train_epochs=args.num_epoch,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="wandb",
        seed=args.seed,
        do_train=True,
        do_eval=True,
        bf16=True,
        gradient_checkpointing=True,
    )

    trainer = CustomSFTTrainer(
        model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        data_collator=data_collator,
        args=training_args,
        peft_config=None,
    )
    trainer.l1_lambda = args.l1_lambda

    for i in range(model.config.num_hidden_layers):
        attn = model.model.layers[i].self_attn
        for j, module in enumerate(attn.attn_v):
            if (i, j) in lofit_heads:
                nn.init.normal_(module, mean=0.0, std=1e-3)
        for j, module in enumerate(attn.attn_b):
            if (i, j) in lofit_heads:
                nn.init.normal_(module, mean=0.0, std=1e-3)

    trainer.train()
    save_dir = out / "hf_checkpoint"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()
