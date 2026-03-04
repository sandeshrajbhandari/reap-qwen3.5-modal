import os
import modal

app = modal.App("master-repo-uploader")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget")
    .pip_install("huggingface-hub")
)

results_vol = modal.Volume.from_name("reap-results")
RESULTS_DIR = "/results"
MODEL_SUBPATH = "Qwen3.5-35B-A3B/evol-codealpaca-v1/pruned_models/reap-seed_42-0.32"
MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_SUBPATH)

REPO_ID = "sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32-GGUF"
CALIBRATION_URL = "https://huggingface.co/spaces/Novaciano/Train-With-Erotiquant3/raw/1ed03c8d5eb4f359942eee3a8c209a68bcee5cdb4/calibration_data_v5_rc.txt"

@app.function(
    image=image,
    volumes={RESULTS_DIR: results_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
)
def upload_all():
    from huggingface_hub import HfApi
    import subprocess

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, exist_ok=True)

    # 1. Upload non-sharded files
    small_files = {
        "imatrix.dat": os.path.join(MODEL_PATH, "imatrix.dat"),
        "Qwen3.5-24B-A3B-REAP-0.32-IQ4_K_M.gguf": os.path.join(MODEL_PATH, "Qwen3.5-24B-A3B-REAP-0.32-IQ4_K_M.gguf"),
    }

    for remote_name, local_path in small_files.items():
        if os.path.exists(local_path):
            print(f"🚀 Uploading {remote_name}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_name,
                repo_id=REPO_ID,
            )

    # 2. Sharded upload for F16 (to handle large file size better)
    f16_path = os.path.join(MODEL_PATH, "model-f16.gguf")
    if os.path.exists(f16_path):
        print(f"🚀 Uploading model-f16.gguf (sharded automatically by HF API if large)...")
        api.upload_file(
            path_or_fileobj=f16_path,
            path_in_repo="model-f16.gguf",
            repo_id=REPO_ID,
        )

    # 3. Download and upload calibration data
    cal_name = "calibration_data_v5_rc.txt"
    cal_path = f"/tmp/{cal_name}"
    # Use direct raw link if possible, or skip if 404
    print(f"⬇️  Downloading {cal_name}...")
    try:
        subprocess.run(["wget", CALIBRATION_URL, "-O", cal_path], check=True)
        print(f"🚀 Uploading {cal_name}...")
        api.upload_file(
            path_or_fileobj=cal_path,
            path_in_repo=cal_name,
            repo_id=REPO_ID,
        )
    except Exception as e:
        print(f"⚠️ Could not download calibration data: {e}")

    print(f"🎉 Master repo updated: https://huggingface.co/{REPO_ID}")

@app.local_entrypoint()
def main():
    upload_all.remote()
