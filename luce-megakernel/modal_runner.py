"""
Run on Modal (or locally): compare BF16 megakernel, INT4 megakernel, and HF INT4 baseline.

GPU note: L4 is Ada Lovelace (sm_89), not Ampere like RTX 3090 (sm_86). Build with
MEGAKERNEL_CUDA_SM=89 on L4.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

DECODE_STEPS = 64
WARMUP_DECODE = 4
PROMPT = "The capital of France is"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_megakernel_decode(dec, first_token_id: int, steps: int) -> float:
    """Time `steps` decode steps starting from first_token_id (already consumed prefill)."""
    nid = first_token_id
    _sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        nid = dec.step(nid)
    _sync()
    return time.perf_counter() - t0


def run_bf16_megakernel() -> dict:
    from model import Decoder

    tok = __import__("transformers").AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-9B", trust_remote_code=True
    )
    dec = Decoder(verbose=False, weight_mode="bf16")
    ids = tok.encode(PROMPT, add_special_tokens=False)
    if len(ids) < 2:
        ids = tok.encode("Hello world", add_special_tokens=False)
    for t in ids[:-1]:
        dec.step(t)
    first = dec.step(ids[-1])
    for _ in range(WARMUP_DECODE):
        bench_megakernel_decode(dec, first, 8)
    dec.reset()
    for t in ids[:-1]:
        dec.step(t)
    first = dec.step(ids[-1])
    elapsed = bench_megakernel_decode(dec, first, DECODE_STEPS)
    return {
        "name": "megakernel_bf16",
        "decode_tok_s": DECODE_STEPS / elapsed,
        "decode_ms": elapsed * 1000,
        "steps": DECODE_STEPS,
    }


def run_int4_megakernel() -> dict:
    from model import Decoder

    model_id = os.environ.get("INT4_MODEL", "Intel/Qwen3.5-9B-int4-AutoRound")
    dec = Decoder(verbose=False, weight_mode="autoround_int4", model_name=model_id)
    tok = dec.tokenizer
    ids = tok.encode(PROMPT, add_special_tokens=False)
    if len(ids) < 2:
        ids = tok.encode("Hello world", add_special_tokens=False)
    for t in ids[:-1]:
        dec.step(t)
    first = dec.step(ids[-1])
    for _ in range(WARMUP_DECODE):
        bench_megakernel_decode(dec, first, 8)
    dec.reset()
    for t in ids[:-1]:
        dec.step(t)
    first = dec.step(ids[-1])
    elapsed = bench_megakernel_decode(dec, first, DECODE_STEPS)
    return {
        "name": "megakernel_int4_autoround",
        "decode_tok_s": DECODE_STEPS / elapsed,
        "decode_ms": elapsed * 1000,
        "steps": DECODE_STEPS,
        "model": model_id,
    }


def run_hf_int4_baseline() -> dict:
    """Transformers + AutoRound layer replacement (many kernels per layer; not the megakernel).

    Plain ``AutoModelForCausalLM.from_pretrained`` on Intel's VL-tagged repo does **not**
    apply ``quantization_config`` to the causal LM stack, so you get fake MISSING weights.
    ``auto_round.utils.model.llm_load_model`` runs ``convert_hf_model`` so qweight/scales load correctly.
    """
    from auto_round.utils.model import llm_load_model

    model_id = os.environ.get("INT4_MODEL", "Intel/Qwen3.5-9B-int4-AutoRound")
    # device="auto" → accelerate device_map for multi-GPU; on single L4 this maps to the GPU.
    full, tok = llm_load_model(model_id, device="auto", trust_remote_code=True)
    full.eval()
    lm = full.model
    head = full.lm_head

    ids = tok.encode(PROMPT, add_special_tokens=False)
    if len(ids) < 2:
        ids = tok.encode("Hello world", add_special_tokens=False)
    input_ids = torch.tensor([ids], device="cuda")

    _sync()
    with torch.inference_mode():
        for _ in range(WARMUP_DECODE):
            out = lm(input_ids, use_cache=True)
            past = out.past_key_values
            nid = head(out.last_hidden_state[:, -1:, :]).argmax(-1)
            for _ in range(8):
                out = lm(nid, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nid = head(out.last_hidden_state).argmax(-1)

        t0 = time.perf_counter()
        out = lm(input_ids, use_cache=True)
        past = out.past_key_values
        nid = head(out.last_hidden_state[:, -1:, :]).argmax(-1)
        for _ in range(DECODE_STEPS):
            out = lm(nid, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nid = head(out.last_hidden_state).argmax(-1)
    _sync()
    elapsed = time.perf_counter() - t0

    del full
    torch.cuda.empty_cache()
    return {
        "name": "hf_int4_baseline_no_megakernel",
        "decode_tok_s": DECODE_STEPS / elapsed,
        "decode_ms": elapsed * 1000,
        "steps": DECODE_STEPS,
        "model": model_id,
        "note": "Prefill for prompt included once; timed loop is decode steps only",
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    header = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": f"{props.major}.{props.minor}",
        "sm_note": "L4=8.9 (Ada); RTX 3090=8.6 (Ampere) — not the same architecture",
    }
    print(json.dumps({"device": header}, indent=2), flush=True)

    results = [header]
    # Lightest first on 22GB GPUs: HF baseline → int4 megakernel → bf16 megakernel.
    modes = os.environ.get("MODAL_BENCH_MODES", "baseline,int4,bf16").split(",")
    modes = [m.strip() for m in modes if m.strip()]

    def _clear():
        torch.cuda.empty_cache()

    if "baseline" in modes:
        print("=== INT4 HF baseline ===", flush=True)
        results.append(run_hf_int4_baseline())
        _clear()
    if "int4" in modes:
        print("=== INT4 megakernel (AutoRound) ===", flush=True)
        results.append(run_int4_megakernel())
        _clear()
    if "bf16" in modes:
        print("=== BF16 megakernel ===", flush=True)
        results.append(run_bf16_megakernel())
        _clear()

    print(json.dumps({"results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
