"""
Modal: benchmark megakernel BF16 vs INT4 vs HF INT4 baseline on L4.

GPU: L4 is Ada Lovelace (compute capability 8.9 / sm_89), not Ampere like RTX 3090 (8.6).
The extension is built with MEGAKERNEL_CUDA_SM=89 on Modal.

Usage (from repo root or this directory):
  modal run luce-megakernel/modal_benchmark_app.py

Requires Modal secret `huggingface-secret` with HF_TOKEN for gated models if needed.
Optional env on the function: MODAL_BENCH_MODES=baseline,int4,bf16 (default; lightest first on 22GB).

GPU architecture: L4 is **Ada** (sm_89), RTX 3090 is **Ampere** (sm_86) — not the same silicon; the app builds with MEGAKERNEL_CUDA_SM=89 on Modal.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import modal

APP_NAME = "megakernel-qwen35-bench"
REMOTE_ROOT = "/root/luce-megakernel"
LOCAL_PKG = pathlib.Path(__file__).resolve().parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "ninja-build", "build-essential")
    .pip_install(
        # Pin cu124: unpinned `torch` can resolve to a cu130 wheel while the image has CUDA 12.4 toolkit.
        "torch==2.5.1+cu124",
        "wheel",
        "setuptools",
        "transformers>=4.45.0",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "auto-round",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # Must be last: bakes local megakernel sources into the image.
    .add_local_dir(str(LOCAL_PKG), remote_path=REMOTE_ROOT)
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="L4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60,
)
def run_benchmark():
    import os

    os.environ["MEGAKERNEL_CUDA_SM"] = "89"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ["CXX"] = "g++"
    os.environ["CC"] = "gcc"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "-e",
            REMOTE_ROOT,
        ],
        check=True,
        cwd=REMOTE_ROOT,
    )

    subprocess.run(
        [sys.executable, os.path.join(REMOTE_ROOT, "modal_runner.py")],
        check=True,
        cwd=REMOTE_ROOT,
        env={**os.environ},
    )


@app.local_entrypoint()
def main():
    run_benchmark.remote()
