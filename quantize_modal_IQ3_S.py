import os
import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
app = modal.App("iq3-s-multimodal-quantizer")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("huggingface-hub")
)

results_vol = modal.Volume.from_name("reap-results")
RESULTS_DIR = "/results"
LLAMA_CPP_BIN_DIR = os.path.join(RESULTS_DIR, "tools/llama.cpp")

DEFAULT_MODEL_SUBPATH = "Qwen3.5-35B-A3B/evol-codealpaca-v1/pruned_models/reap-seed_42-0.32"


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RESULTS_DIR: results_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=14400,
)
def run_iq3_s_quantization(model_subpath: str, hf_repo: str = "", output_filename: str = ""):
    import subprocess
    from huggingface_hub import HfApi

    model_path = os.path.join(RESULTS_DIR, model_subpath)
    f16_gguf_path = os.path.join(model_path, "model-f16.gguf")
    imatrix_path = os.path.join(model_path, "imatrix.dat")
    quantize_bin = os.path.join(LLAMA_CPP_BIN_DIR, "llama-quantize")

    if not output_filename:
        model_name = os.path.basename(model_subpath.rstrip("/"))
        output_filename = f"{model_name}-IQ3_S.gguf"
    output_path = os.path.join(model_path, output_filename)

    # 1. Verify inputs
    if not os.path.exists(quantize_bin):
        print(f"❌ Error: llama-quantize not found at {quantize_bin}. Run generate_imatrix.py --rebuild first.")
        return

    if not os.path.exists(f16_gguf_path):
        print(f"❌ Error: F16 GGUF not found at {f16_gguf_path}.")
        return

    if not os.path.exists(imatrix_path):
        print(f"❌ Error: imatrix.dat not found at {imatrix_path}.")
        return

    os.chmod(quantize_bin, 0o755)

    # 2. Multimodal IQ3_S recipe:
    #    - keep embeddings/output in Q6_K
    #    - force gate/qkv/shexp tensors to Q6_K
    #    - down_exps in IQ3_S
    #    - gate/up experts in IQ2_S
    #    - SSM alpha/beta in Q8_0
    #    - explicitly keep multimodal glue tensors in F32
    cmd = [
        quantize_bin,
        "--imatrix",
        imatrix_path,
        "--token-embedding-type",
        "q6_k",
        "--output-tensor-type",
        "q6_k",
        "--tensor-type",
        "attn_gate=q6_k",
        "--tensor-type",
        "attn_qkv=q6_k",
        "--tensor-type",
        "ffn_down_shexp=q6_k",
        "--tensor-type",
        "ffn_gate_shexp=q6_k",
        "--tensor-type",
        "ffn_up_shexp=q6_k",
        "--tensor-type",
        "ffn_down_exps=iq3_s",
        "--tensor-type",
        "ffn_gate_exps=iq2_s",
        "--tensor-type",
        "ffn_up_exps=iq2_s",
        "--tensor-type",
        "ssm_alpha=q8_0",
        "--tensor-type",
        "ssm_beta=q8_0",
        "--tensor-type",
        "ssm_out=q6_k",
        "--tensor-type",
        "ffn_gate_inp=f32",
        "--tensor-type",
        "ffn_gate_inp_shexp=f32",
        "--tensor-type",
        "ssm_conv1d=f32",
        f16_gguf_path,
        output_path,
        "IQ3_S",
    ]

    print(f"📦 Starting multimodal IQ3_S quantization for {output_filename}...")
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Quantization failed:\n{result.stderr}")
        return

    print(result.stdout)
    print(f"✅ Quantization complete: {output_path}")
    results_vol.commit()

    # 3. Optional upload
    if not hf_repo:
        print("ℹ️ hf_repo not provided; skipping upload.")
        return

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
def main(
    model_subpath: str = DEFAULT_MODEL_SUBPATH,
    hf_repo: str = "",
    output_filename: str = "",
):
    run_iq3_s_quantization.remote(
        model_subpath=model_subpath,
        hf_repo=hf_repo,
        output_filename=output_filename,
    )
