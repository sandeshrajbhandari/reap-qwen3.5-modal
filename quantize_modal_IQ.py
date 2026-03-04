import os
import pathlib
import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
app = modal.App("unsloth-style-quantize")

# Use a CUDA-enabled base image
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("huggingface-hub")
)

# Volumes
results_vol = modal.Volume.from_name("reap-results")
RESULTS_DIR = "/results"
LLAMA_CPP_BIN_DIR = os.path.join(RESULTS_DIR, "tools/llama.cpp")

# Paths
MODEL_SUBPATH = "Qwen3.5-35B-A3B/evol-codealpaca-v1/pruned_models/reap-seed_42-0.32"
MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_SUBPATH)
F16_GGUF_PATH = os.path.join(MODEL_PATH, "model-f16.gguf")
IMATRIX_PATH = os.path.join(MODEL_PATH, "imatrix.dat")

@app.function(
    image=image,
    gpu="L4", # Quantization is fast on GPU
    volumes={RESULTS_DIR: results_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600, # 1 hour
)
def run_iq_quantization(hf_repo: str):
    import subprocess
    from huggingface_hub import HfApi

    # 1. Verify files exist
    quantize_bin = os.path.join(LLAMA_CPP_BIN_DIR, "llama-quantize")
    if not os.path.exists(quantize_bin):
        print(f"❌ Error: llama-quantize not found at {quantize_bin}. Run generate_imatrix.py --rebuild first.")
        return

    if not os.path.exists(F16_GGUF_PATH):
        print(f"❌ Error: F16 GGUF not found at {F16_GGUF_PATH}.")
        return

    if not os.path.exists(IMATRIX_PATH):
        print(f"❌ Error: imatrix.dat not found at {IMATRIX_PATH}.")
        return

    # Ensure binary is executable
    os.chmod(quantize_bin, 0o755)

    # 2. Setup output path
    output_filename = "Qwen3.5-24B-A3B-REAP-0.32-IQ4_K_M.gguf"
    output_path = os.path.join(MODEL_PATH, output_filename)

    # 3. Run quantization with Unsloth recipe
    print(f"📦 Starting Unsloth-style quantization for {output_filename}...")
    
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
        "--tensor-type", "ffn_down_exps=q5_k",
        F16_GGUF_PATH,
        output_path,
        "Q4_K_M"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Quantization failed: {result.stderr}")
        return
    
    print(f"✅ Quantization complete: {output_path}")
    results_vol.commit()

    # 4. Upload to Hugging Face
    print(f"🚀 Uploading to Hugging Face repo: {hf_repo}...")
    api = HfApi()
    api.create_repo(repo_id=hf_repo, exist_ok=True)
    
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo=output_filename,
        repo_id=hf_repo,
    )
    
    print(f"🎉 All done! https://huggingface.co/{hf_repo}/blob/main/{output_filename}")

@app.local_entrypoint()
def main(hf_repo: str = None):
    if not hf_repo:
        print("Usage: modal run quantize_modal_IQ.py --hf-repo username/model-IQ-GGUF")
        return
    run_iq_quantization.remote(hf_repo)
