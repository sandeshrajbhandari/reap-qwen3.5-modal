"""Benchmark pp512 tg128 — standard llama.cpp benchmark format.
Also tests end-to-end correctness (prefill → decode handoff).

Scope: batch-size-1 single-stream decode, targeting local inference.
All measurements use torch.cuda.synchronize() barriers + perf_counter.
One warm-up run precedes each timed section.

Set USE_INTEL_AUTOROUND=1 to benchmark Intel/Qwen3.5-9B-int4-AutoRound (int4 layers + bf16 lm_head)."""
import os
import time, torch
from model import Decoder, HIDDEN_SIZE, INTERMEDIATE_SIZE, FA_QPROJ_SIZE, FA_Q_SIZE, FA_KV_SIZE
from model import DN_CONV_CHANNELS, DN_V_SIZE, DN_NUM_V_HEADS, MAX_SEQ_LEN

USE_AR = os.environ.get("USE_INTEL_AUTOROUND", "").lower() in ("1", "true", "yes")
if USE_AR:
    dec = Decoder(verbose=True, weight_mode="autoround_int4")
    tok = dec.tokenizer
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
    dec = Decoder(verbose=True)

# Allocate prefill buffers for max 512 tokens
S_MAX = 512
bf16 = dict(dtype=torch.bfloat16, device="cuda")
f32 = dict(dtype=torch.float32, device="cuda")
i32 = dict(dtype=torch.int32, device="cuda")
mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
bufs = dict(
    hidden=torch.empty(S_MAX*HIDDEN_SIZE, **bf16),
    residual=torch.empty(S_MAX*HIDDEN_SIZE, **bf16),
    normalized=torch.empty(S_MAX*HIDDEN_SIZE, **bf16),
    proj_buf=torch.empty(S_MAX*mx, **bf16),
    proj_buf2=torch.empty(S_MAX*mx, **bf16),
    attn_buf=torch.empty(S_MAX*max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
    mlp_buf=torch.empty(S_MAX*INTERMEDIATE_SIZE, **bf16),
    dn_out_buf=torch.empty(S_MAX*DN_V_SIZE, **bf16),
    beta_buf=torch.empty(S_MAX*DN_NUM_V_HEADS, **f32),
    alpha_buf=torch.empty(S_MAX*DN_NUM_V_HEADS, **f32),
    final_normed=torch.empty(HIDDEN_SIZE, **bf16),
    hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
    lm_bmv=torch.empty(1024, **f32),
    lm_bmi=torch.empty(1024, **i32),
)

def prefill(ids):
    ids_t = torch.tensor(ids, dtype=torch.int32, device="cuda")
    dec.prefill(ids_t, bufs)
    return dec._out_token.item()

# ============================================================
# 1. End-to-end correctness test
# ============================================================
print("\n=== Correctness test ===", flush=True)
prompt = "The capital of France is"
ids = tok.encode(prompt, add_special_tokens=False)
dec.reset()
first = prefill(ids)
print(f"Prefill → first token: {first} = '{tok.decode([first])}'", flush=True)

# Continue with decode megakernel
out = [first]
nid = first
for _ in range(30):
    nid = dec.step(nid)
    if nid == tok.eos_token_id: break
    out.append(nid)
text = tok.decode(out, skip_special_tokens=True)
print(f"Output: {text[:80]}", flush=True)

# Reference: pure decode (step-by-step)
dec.reset()
for t in ids[:-1]: dec.step(t)
ref_first = dec.step(ids[-1])
ref_out = [ref_first]
nid = ref_first
for _ in range(30):
    nid = dec.step(nid)
    if nid == tok.eos_token_id: break
    ref_out.append(nid)
ref_text = tok.decode(ref_out, skip_special_tokens=True)
print(f"Ref:    {ref_text[:80]}", flush=True)

if out == ref_out:
    print("PASS: megakernel output matches reference decode path", flush=True)
else:
    print("FAIL: output mismatch between megakernel and reference", flush=True)
    print(f"  Megakernel tokens: {out[:10]}...", flush=True)
    print(f"  Reference tokens:  {ref_out[:10]}...", flush=True)

# ============================================================
# 2. pp512 benchmark (prompt processing)
# ============================================================
print("\n=== pp512 benchmark ===", flush=True)
# Generate a 512-token prompt
long_prompt = "Explain in great detail the history of artificial intelligence, " * 30
long_ids = tok.encode(long_prompt, add_special_tokens=False)[:512]
print(f"Prompt tokens: {len(long_ids)}", flush=True)

# Warmup
dec.reset()
prefill(long_ids)

# Benchmark
dec.reset()
torch.cuda.synchronize()
t0 = time.perf_counter()
prefill(long_ids)
torch.cuda.synchronize()
pp_time = time.perf_counter() - t0
pp_tps = len(long_ids) / pp_time
print(f"pp{len(long_ids)}: {pp_tps:.1f} tok/s ({pp_time*1000:.1f}ms)", flush=True)

# ============================================================
# 3. tg128 benchmark (token generation)
# ============================================================
print("\n=== tg128 benchmark ===", flush=True)
# Prefill a short prompt, then generate 128 tokens
short_ids = tok.encode("Hello", add_special_tokens=False)
dec.reset()
first = prefill(short_ids)

torch.cuda.synchronize()
t0 = time.perf_counter()
gen_out = []
nid = first
for _ in range(128):
    nid = dec.step(nid)
    if nid == tok.eos_token_id: break
    gen_out.append(nid)
torch.cuda.synchronize()
tg_time = time.perf_counter() - t0
tg_tps = len(gen_out) / tg_time
print(f"tg{len(gen_out)}: {tg_tps:.1f} tok/s ({tg_time*1000:.1f}ms)", flush=True)

# ============================================================
# Summary
# ============================================================
label = "Intel AutoRound int4" if USE_AR else "Qwen3.5-9B BF16"
print(f"\n=== Summary (RTX 3090, {label}) ===", flush=True)
print(f"pp{len(long_ids):>3d}: {pp_tps:>7.1f} tok/s", flush=True)
print(f"tg{len(gen_out):>3d}: {tg_tps:>7.1f} tok/s", flush=True)
