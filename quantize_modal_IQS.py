import os
import modal

app = modal.App("iqs-quantizer")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("huggingface-hub")
)

results_vol = modal.Volume.from_name("reap-results")
RESULTS_DIR = "/results"
LLAMA_CPP_BIN_DIR = os.path.join(RESULTS_DIR, "tools/llama.cpp")
MODEL_SUBPATH = "Qwen3.5-35B-A3B/evol-codealpaca-v1/pruned_models/reap-seed_42-0.32"
MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_SUBPATH)

F16_GGUF_PATH = os.path.join(MODEL_PATH, "model-f16.gguf")
IMATRIX_PATH = os.path.join(MODEL_PATH, "imatrix.dat")
REPO_ID = "sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32-GGUF"

@app.function(
    image=image,
    gpu="L4",
    volumes={RESULTS_DIR: results_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def run_iqs_quantization():
    import subprocess
    from huggingface_hub import HfApi

    quantize_bin = os.path.join(LLAMA_CPP_BIN_DIR, "llama-quantize")
    output_filename = "Qwen3.5-24B-A3B-REAP-0.32-IQ4_K_S.gguf"
    output_path = os.path.join(MODEL_PATH, output_filename)

    # Recipe for IQ4_K_S (using Q4_K_S base)
    cmd = [
        quantize_bin,
        "--imatrix", IMATRIX_PATH,
        "--token-embedding-type", "q8_0",
        "--output-tensor-type", "q6_k",
        "--tensor-type", "attn_gate=q8_0",
        "--tensor-type", "attn_qkv=q8_0",
        "--tensor-type", "ffn_down_shexp=q8_0",
        "--tensor-type", "ffn_gate_shexp=q8_0",
        "--tensor-type", "ffn_up_shexp=q8_0",
        "--tensor-type", "ssm_alpha=q8_0",
        "--tensor-type", "ssm_beta=q8_0",
        "--tensor-type", "ssm_out=q8_0",
        F16_GGUF_PATH,
        output_path,
        "Q4_K_S"
    ]

    print(f"📦 Starting IQS (Q4_K_S) quantization...")
    os.chmod(quantize_bin, 0o755)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr}")
        return

    print(f"✅ Quantization complete: {output_path}")
    results_vol.commit()

    print(f"🚀 Uploading to {REPO_ID}...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo=output_filename,
        repo_id=REPO_ID,
    )
    print(f"🎉 All done! https://huggingface.co/{REPO_ID}/blob/main/{output_filename}")

@app.local_entrypoint()
def main():
    run_iqs_quantization.remote()
