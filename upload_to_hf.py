import modal
import os
import pathlib

# Define the Modal App
app = modal.App("shard-and-upload-to-hf")

# Define the container image with transformers and torch
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "huggingface-hub",
        "torch",
        "accelerate",
        "safetensors",
    )
    .run_commands(
        "pip install git+https://github.com/huggingface/transformers.git"
    )
)

# Define the volume and mount point
volume = modal.Volume.from_name("reap-results")
RESULTS_DIR = "/results"

# Constants for the upload
REPO_ID = "sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32"
MODEL_PATH_IN_VOLUME = "Qwen3.5-35B-A3B/evol-codealpaca-v1/pruned_models/reap-seed_42-0.32"
FULL_MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_PATH_IN_VOLUME)

@app.function(
    image=image,
    gpu="A100-80GB", # Need GPU to load the 35B model weights for sharding
    volumes={RESULTS_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200, # 2 hours
)
def shard_and_upload():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi
    import torch
    
    print(f"Checking if model path exists: {FULL_MODEL_PATH}")
    if not os.path.exists(FULL_MODEL_PATH):
        print(f"❌ Error: Model path {FULL_MODEL_PATH} not found.")
        return

    temp_shard_dir = "/tmp/sharded_model"
    os.makedirs(temp_shard_dir, exist_ok=True)

    print(f"🚀 Loading model from {FULL_MODEL_PATH} for sharding...")
    # Load model (CPU is safer for just sharding if VRAM is tight, but A100-80GB should handle it)
    model = AutoModelForCausalLM.from_pretrained(
        FULL_MODEL_PATH,
        torch_dtype="auto",
        device_map="cpu", # Loading to CPU to save VRAM during the shard-writing process
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(FULL_MODEL_PATH, trust_remote_code=True)

    print(f"📦 Saving sharded model to {temp_shard_dir} (max_shard_size='5GB')...")
    model.save_pretrained(temp_shard_dir, max_shard_size="5GB", safe_serialization=True)
    tokenizer.save_pretrained(temp_shard_dir)

    print(f"🚀 Uploading sharded model to HF repo {REPO_ID} ...")
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, exist_ok=True)
    
    api.upload_folder(
        folder_path=temp_shard_dir,
        repo_id=REPO_ID,
        repo_type="model",
    )
    
    print(f"✅ Sharding and upload successful! View it at: https://huggingface.co/{REPO_ID}")

@app.local_entrypoint()
def main():
    shard_and_upload.remote()
