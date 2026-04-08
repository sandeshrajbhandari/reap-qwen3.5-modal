"""Run RestoreLCC-style Qwen3 MoE recovery on Modal with a T4 GPU.

Requires Modal secret `huggingface-secret` with `HF_TOKEN`.

Note: contrastive probing loads the full reference and pruned checkpoints in float16 with `device_map="auto"`.
A single T4 (16 GB) is not enough for two Qwen3.5-35B-A3B weights at once; use a larger GPU for that pair,
or point `--reference-model-id` / `--pruned-model-path` at smaller checkpoints for a dry run.

Example:
  modal run restore_lcc_modal.py \\
    --pruned-model-path your-org/qwen3.5-35b-reap-pruned \\
    --reference-model-id Qwen/Qwen3.5-35B-A3B

Use `--probe-only` to only compute SVD directions and head rankings (still loads both models).
Outputs are written under `/results/restorelcc/` on the `reap-results` volume when using default paths.
"""

from __future__ import annotations

import pathlib
import subprocess

import modal

REFERENCE_MODEL_DEFAULT = "Qwen/Qwen3.5-35B-A3B"

app = modal.App("restorelcc-qwen3-moe")

hf_cache_vol = modal.Volume.from_name("hf-cache-reap", create_if_missing=True)
results_vol = modal.Volume.from_name("reap-results", create_if_missing=True)

HF_CACHE_DIR = "/root/.cache/huggingface"
RESULTS_ROOT = "/results/restorelcc"

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

_restore_dir = pathlib.Path(__file__).resolve().parent / "restorelcc"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "numpy",
        "pandas",
        "pyvene==0.1.6",
        "scikit-learn",
        "sentencepiece",
        "protobuf",
        "sentence-transformers>=3.0.0",
        "tqdm",
        "trl>=0.9.0",
        "wandb",
    )
    .run_commands("pip install git+https://github.com/huggingface/transformers.git")
    .add_local_dir(
        str(_restore_dir),
        remote_path="/root/restorelcc",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MINUTES = 60


@app.function(
    image=image,
    gpu="T4",
    volumes={HF_CACHE_DIR: hf_cache_vol, "/results": results_vol},
    secrets=[huggingface_secret],
    timeout=12 * MINUTES,
)
def run_restore_lcc(
    reference_model_id: str = REFERENCE_MODEL_DEFAULT,
    pruned_model_path: str = "",
    num_train_samples: int = 512,
    num_val_samples: int = 128,
    use_topk_heads: int = 128,
    probe_only: bool = False,
    skip_probe: bool = False,
    lr: float = 1e-3,
    num_epoch: int = 2,
    train_batch: int = 1,
    eval_batch: int = 1,
    max_seq_length: int = 384,
    output_dir: str | None = None,
    run_mode: str = "train",
):
    import os

    if not pruned_model_path:
        raise ValueError("pruned_model_path is required (HF repo id or path readable in the container).")

    out = output_dir or f"{RESULTS_ROOT}/run"
    os.makedirs(out, exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    cmd = [
        "python",
        "/root/restorelcc/restore_lcc_train_qwen3_moe.py",
        "--reference-model-id",
        reference_model_id,
        "--pruned-model-path",
        pruned_model_path,
        "--num-train-samples",
        str(num_train_samples),
        "--num-val-samples",
        str(num_val_samples),
        "--use-topk-heads",
        str(use_topk_heads),
        "--lr",
        str(lr),
        "--num-epoch",
        str(num_epoch),
        "--train-batch",
        str(train_batch),
        "--eval-batch",
        str(eval_batch),
        "--max-seq-length",
        str(max_seq_length),
        "--output-dir",
        out,
        "--hf-cache-dir",
        HF_CACHE_DIR,
        "--run-mode",
        run_mode,
    ]
    if probe_only:
        cmd.append("--probe-only")
    if skip_probe:
        cmd.append("--skip-probe")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd="/root/restorelcc")
    results_vol.commit()
    hf_cache_vol.commit()
    print(f"Done. Check Modal volume `reap-results` under {out}")


@app.local_entrypoint()
def main(
    reference_model_id: str = REFERENCE_MODEL_DEFAULT,
    pruned_model_path: str = "",
    num_train_samples: int = 512,
    num_val_samples: int = 128,
    use_topk_heads: int = 128,
    probe_only: bool = False,
    skip_probe: bool = False,
    lr: float = 1e-3,
    num_epoch: int = 2,
    train_batch: int = 1,
    eval_batch: int = 1,
    max_seq_length: int = 384,
    output_dir: str | None = None,
    run_mode: str = "train",
):
    run_restore_lcc.remote(
        reference_model_id=reference_model_id,
        pruned_model_path=pruned_model_path,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        use_topk_heads=use_topk_heads,
        probe_only=probe_only,
        skip_probe=skip_probe,
        lr=lr,
        num_epoch=num_epoch,
        train_batch=train_batch,
        eval_batch=eval_batch,
        max_seq_length=max_seq_length,
        output_dir=output_dir,
        run_mode=run_mode,
    )
