import os
import pathlib
import modal

# Define the Modal App
app = modal.App("llama-cpp-quantize")

# Define the container image
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "tar", "build-essential")
    .pip_install(
        "huggingface-hub",
        "numpy",
        "sentencepiece",
        "gguf",
    )
    # Clone llama.cpp and install requirements for conversion script
    .run_commands(
        "git clone https://github.com/ggml-org/llama.cpp.git /root/llama.cpp",
        "pip install -r /root/llama.cpp/requirements.txt",
    )
    # Re-install transformers from source to ensure qwen3.5 support
    .run_commands(
        "pip install git+https://github.com/huggingface/transformers.git",
    )
    # Download llama-quantize binary and shared libraries
    .run_commands(
        "wget https://github.com/ggml-org/llama.cpp/releases/download/b8192/llama-b8192-bin-ubuntu-vulkan-x64.tar.gz -O /tmp/llama-bin.tar.gz",
        "mkdir -p /tmp/llama-bin && tar -xvf /tmp/llama-bin.tar.gz -C /tmp/llama-bin",
        "find /tmp/llama-bin -name llama-quantize -exec cp {} /usr/local/bin/llama-quantize \;",
        "chmod +x /usr/local/bin/llama-quantize",
        # Copy all shared libraries to /usr/local/lib
        "find /tmp/llama-bin -name '*.so*' -exec cp {} /usr/local/lib/ \;",
        "ldconfig",
    )
)

# Define the volume
volume = modal.Volume.from_name("reap-results")

# Constants
VOLUME_MOUNT_PATH = "/vol"
MODEL_SUBPATH = "Qwen3.5-35B-A3B/evol-codealpaca-v1/pruned_models/reap-seed_42-0.32"
MODEL_PATH = os.path.join(VOLUME_MOUNT_PATH, MODEL_SUBPATH)

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200, # 2 hours
)
def quantize_and_upload(hf_repo: str):
    import subprocess
    import json
    from huggingface_hub import HfApi

    print(f"Checking model path: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    # 0. Patch config.json architectures to bypass llama.cpp naming mismatch
    print("Step 0: Patching architectures to Qwen3_5MoeForConditionalGeneration...")
    config_path = os.path.join(MODEL_PATH, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    original_architectures = config.get("architectures")
    config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    try:
        # 1. Convert to GGUF FP16
        print("Step 1: Converting to GGUF FP16...")
        gguf_f16_path = "/tmp/model-f16.gguf"
        convert_cmd = [
            "python3", "/root/llama.cpp/convert_hf_to_gguf.py",
            MODEL_PATH,
            "--outfile", gguf_f16_path,
            "--outtype", "f16"
        ]
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Conversion failed: {result.stderr}")
            return
        print("Conversion successful.")

        # 2. Quantize to Q4_K_M
        print("Step 2: Quantizing to Q4_K_M...")
        gguf_q4_path = "/tmp/model-Q4_K_M.gguf"
        quantize_cmd = [
            "llama-quantize",
            gguf_f16_path,
            gguf_q4_path,
            "Q4_K_M"
        ]
        result = subprocess.run(quantize_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Quantization failed: {result.stderr}")
            return
        print("Quantization successful.")

        # 3. Upload to Hugging Face
        print(f"Step 3: Uploading to Hugging Face repo: {hf_repo}...")
        api = HfApi()
        api.create_repo(repo_id=hf_repo, exist_ok=True)

        filename = f"{MODEL_SUBPATH.split('/')[-1]}-Q4_K_M.gguf"
        api.upload_file(
            path_or_fileobj=gguf_q4_path,
            path_in_repo=filename,
            repo_id=hf_repo,
        )
        print(f"Upload successful: https://huggingface.co/{hf_repo}/blob/main/{filename}")

    finally:
        # Restore original architectures
        print(f"Restoring architectures to {original_architectures}...")
        config["architectures"] = original_architectures
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

@app.local_entrypoint()
def main(hf_repo: str = None):
    if not hf_repo:
        print("Usage: modal run quantize_modal.py --hf-repo username/model-GGUF")
        return
    quantize_and_upload.remote(hf_repo)
