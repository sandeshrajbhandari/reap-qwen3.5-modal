#!/usr/bin/env python3
"""
Phase 1 — Offline teacher: run the large (e.g. 35B) model once, save last-layer hidden
states and top-K logits per token position to disk. Avoids student/teacher swapping.

Each sample is one .pt file with:
  - input_ids (L,) int64
  - hidden_states (L, H) float16 — last decoder hidden before lm_head
  - topk_values (L-1, K) float16 — top-K logits at positions 0..L-2 (next-token prediction)
  - topk_indices (L-1, K) int64 — vocab indices (stored as int64; phase2 uses long)

Example:
  python phase1_teacher_offline.py \\
    --teacher-model-id Qwen/Qwen3.5-35B-A3B \\
    --data-jsonl ./data/mix.jsonl \\
    --output-dir ./teacher_data \\
    --load-in-4bit \\
    --max-length 1024 \\
    --topk 150
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model-id", type=str, required=True)
    p.add_argument(
        "--data-jsonl",
        type=str,
        required=True,
        help="JSONL with a string field (default key: text)",
    )
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--output-dir", type=str, default="teacher_data")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--topk", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--start-index", type=int, default=0, help="Resume: first dataset row")
    p.add_argument("--max-samples", type=int, default=None, help="Cap rows processed")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        args.teacher_model_id, trust_remote_code=args.trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_cfg = None
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_id,
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=None if args.load_in_4bit else getattr(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    ds = load_dataset("json", data_files=args.data_jsonl, split="train")
    n = len(ds)
    end = n if args.max_samples is None else min(n, args.start_index + args.max_samples)

    idx_global = args.start_index
    batch_texts: list[str] = []
    batch_rows: list[int] = []

    def flush_batch():
        nonlocal batch_texts, batch_rows, idx_global
        if not batch_texts:
            return
        enc = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_last = outputs.hidden_states[-1]
        logits = outputs.logits

        for j in range(hidden_last.size(0)):
            row_id = batch_rows[j]
            valid = int(enc["attention_mask"][j].sum().item())
            ids = enc["input_ids"][j, :valid].detach().cpu()
            hid = hidden_last[j, :valid].detach().float().cpu().half()

            if valid > 1:
                lg = logits[j, : valid - 1]
                tv, ti = torch.topk(lg, k=min(args.topk, lg.size(-1)), dim=-1)
                tv = tv.detach().cpu().half()
                ti = ti.detach().cpu().long()
            else:
                tv = torch.empty(0, 0, dtype=torch.float16)
                ti = torch.empty(0, 0, dtype=torch.long)

            payload = {
                "input_ids": ids,
                "hidden_states": hid,
                "topk_values": tv,
                "topk_indices": ti,
            }
            torch.save(payload, out / f"sample_{row_id:06d}.pt")

        batch_texts = []
        batch_rows = []

    for i in tqdm(range(args.start_index, end), desc="teacher"):
        ex = ds[i]
        text = ex.get(args.text_field)
        if text is None:
            raise KeyError(f"Missing field {args.text_field!r} in row {i}")
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        batch_texts.append(text)
        batch_rows.append(i)
        if len(batch_texts) >= args.batch_size:
            flush_batch()
    flush_batch()

    meta = {
        "teacher_model_id": args.teacher_model_id,
        "data_jsonl": os.path.abspath(args.data_jsonl),
        "max_length": args.max_length,
        "topk": args.topk,
        "num_rows": end - args.start_index,
        "text_field": args.text_field,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {end - args.start_index} samples under {out}")


if __name__ == "__main__":
    main()
