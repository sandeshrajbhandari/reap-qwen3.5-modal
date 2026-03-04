"""Prune Qwen/Qwen3.5-35B-A3B using REAP on Modal and upload to Hugging Face.

Usage:
    modal run prune_qwen.py --hf-repo-id your-username/qwen3.5-35b-reap-pruned
"""

import os
import pathlib
import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
REAP_REPO = "https://github.com/sandeshrajbhandari/reap.git"
REAP_BRANCH = "feat/qwen3.5-moe-support"

DEFAULT_COMPRESSION_RATIO = 0.32
DEFAULT_SAMPLES_PER_CATEGORY = 1
PRUNE_METHOD = "reap"

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------
app = modal.App("reap-prune-qwen")

hf_cache_vol = modal.Volume.from_name("hf-cache-reap", create_if_missing=True)
results_vol = modal.Volume.from_name("reap-results", create_if_missing=True)

HF_CACHE_DIR = "/root/.cache/huggingface"
RESULTS_DIR = "/results"
REAP_DIR = "/root/reap"

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "clang", "cmake")
    .uv_pip_install(
        "torch==2.7.1",
        "vllm==0.10.0",
        "accelerate>=1.7.0",
        "datasets>=3.6.0,<4.0.0",
        "git+https://github.com/huggingface/transformers.git", # Install from source for qwen3.5 support
        "matplotlib>=3.10.3",
        "seaborn>=0.13.2",
        "python-dotenv>=1.1.0",
        "jupyter>=1.1.1",
        "wandb>=0.21.1",
        "hatchling>=1.27.0",
        "trl>=0.21.0",
        "umap-learn>=0.5.7",
        "lm-eval[vllm,api]>=0.4.9.1",
        "evalplus[vllm]>=0.3.1",
        "huggingface-hub>=0.34.0",
        "sentencepiece",
        "protobuf",
        "scipy",
        "pandas",
        "tqdm",
        "scikit-learn",
        "bitsandbytes",
        "triton",
    )
    .run_commands(
        "pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels"
    )
    .run_commands(
        f"git clone --recursive -b {REAP_BRANCH} {REAP_REPO} {REAP_DIR}",
        f"cd {REAP_DIR} && sed -i -E '/livecodebench|crfm-helm|evalscope|evalplus/d' pyproject.toml",
        f"cd {REAP_DIR} && pip install -e .",
        "pip install git+https://github.com/huggingface/transformers.git",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MINUTES = 60


def sync_and_show_commit():
    import subprocess
    import os
    print(f"🔄 Syncing latest code from {REAP_REPO} branch {REAP_BRANCH} ...")
    subprocess.run(["git", "fetch", "origin", REAP_BRANCH], cwd=REAP_DIR, check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{REAP_BRANCH}"], cwd=REAP_DIR, check=True)
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REAP_DIR).decode().strip()
    print(f"📌 Current REAP commit: {commit_hash}")
    return commit_hash


# ---------------------------------------------------------------------------
# Step 1: Download
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_DIR: hf_cache_vol},
    secrets=[huggingface_secret],
    timeout=30 * MINUTES,
)
def download_model():
    from huggingface_hub import snapshot_download
    print(f"⬇️  Downloading {MODEL_NAME} ...")
    snapshot_download(MODEL_NAME, ignore_patterns=["*.pt", "*.bin"])
    hf_cache_vol.commit()


# ---------------------------------------------------------------------------
# Step 2: Observer
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={HF_CACHE_DIR: hf_cache_vol, RESULTS_DIR: results_vol},
    secrets=[huggingface_secret],
    timeout=60 * MINUTES,
)
def run_observer(samples_per_category: int):
    import os
    import subprocess
    
    sync_and_show_commit()

    if not os.path.exists("artifacts"):
        os.symlink("/results", "artifacts")

    cmd = [
        "python", f"{REAP_DIR}/src/reap/prune.py",
        "--model-name", MODEL_NAME,
        "--run_observer_only", "true",
        "--samples_per_category", str(samples_per_category),
        "--model_max_length", "1024",
        "--record_pruning_metrics_only", "true",
    ]
    subprocess.run(cmd, check=True)
    results_vol.commit()


# ---------------------------------------------------------------------------
# Step 3: Pruning
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={HF_CACHE_DIR: hf_cache_vol, RESULTS_DIR: results_vol},
    secrets=[huggingface_secret],
    timeout=60 * MINUTES,
)
def run_pruning(compression_ratio: float, samples_per_category: int):
    import os
    import subprocess
    
    sync_and_show_commit()

    if not os.path.exists("artifacts"):
        os.symlink("/results", "artifacts")

    cmd = [
        "python", f"{REAP_DIR}/src/reap/prune.py",
        "--model-name", MODEL_NAME,
        "--compression-ratio", str(compression_ratio),
        "--prune-method", PRUNE_METHOD,
        "--samples_per_category", str(samples_per_category),
        "--do_eval", "false",
        "--model_max_length", "1024",
    ]
    subprocess.run(cmd, check=True)
    results_vol.commit()


# ---------------------------------------------------------------------------
# Step 4: Upload to HF
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={RESULTS_DIR: results_vol},
    secrets=[huggingface_secret],
    timeout=60 * MINUTES,
)
def upload_to_hf(repo_id: str):
    import os
    from huggingface_hub import HfApi

    model_clean = MODEL_NAME.split("/")[-1]
    base_path = pathlib.Path("/results") / model_clean / "evol-codealpaca-v1" / "pruned_models"
    
    if not base_path.exists():
        print(f"❌ Could not find pruned models at {base_path}")
        return

    dirs = sorted(base_path.iterdir(), key=os.path.getmtime)
    if not dirs:
        print("❌ No pruned model directories found.")
        return
    
    pruned_path = dirs[-1]
    print(f"🚀 Uploading {pruned_path} to HF repo {repo_id} ...")
    
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=str(pruned_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"✅ Upload complete: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    hf_repo_id: str = None,
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
    samples_per_category: int = DEFAULT_SAMPLES_PER_CATEGORY,
    prune_only: bool = False,
):
    if not prune_only:
        download_model.remote()
        run_observer.remote(samples_per_category)
    
    run_pruning.remote(compression_ratio, samples_per_category)

    if hf_repo_id:
        upload_to_hf.remote(hf_repo_id)
    else:
        print("💡 No --hf-repo-id provided, skipping upload.")
