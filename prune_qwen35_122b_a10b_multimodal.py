"""Prune Qwen/Qwen3.5-122B-A10B with REAP on Modal and preserve multimodal assets.

Usage:
    modal run prune_qwen35_122b_a10b_multimodal.py --hf-repo-id your-username/qwen3.5-122b-a10b-reap
"""

import json
import os
import pathlib
import re
import shutil

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3.5-122B-A10B"
REAP_REPO = "https://github.com/sandeshrajbhandari/reap.git"
REAP_BRANCH = "feat/qwen3.5-moe-support"

DEFAULT_DATASET_NAME = (
    "theblackcat102/evol-codealpaca-v1:250,"
    "open-r1/Mixture-of-Thoughts[code]:250,"
    "open-r1/Mixture-of-Thoughts[math]:250,"
    "open-r1/Mixture-of-Thoughts[science]:250"
)
DEFAULT_COMPRESSION_RATIO = 0.20
DEFAULT_MODEL_MAX_LENGTH = 4096
DEFAULT_SEED = 42
DEFAULT_RENORMALIZE_ROUTER_WEIGHTS = True
PRUNE_METHOD = "reap"
DEFAULT_HF_REPO_ID = "sandeshrajx/qwen3.5-122b-a10b-reap-0.20-composite"

VISUAL_TENSOR_PREFIX = "model.visual."
MULTIMODAL_METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
]

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------
app = modal.App("reap-prune-qwen3.5-122b-a10b-multimodal")

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
        "git+https://github.com/huggingface/transformers.git",
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
        "safetensors>=0.5.3",
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
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

MINUTES = 60


def _as_cli_bool(value: bool) -> str:
    return "true" if value else "false"


def sync_and_show_commit():
    import subprocess

    print(f"Syncing latest code from {REAP_REPO} branch {REAP_BRANCH} ...")
    subprocess.run(["git", "fetch", "origin", REAP_BRANCH], cwd=REAP_DIR, check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{REAP_BRANCH}"], cwd=REAP_DIR, check=True)
    patch_reap_lazy_eval_import()
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REAP_DIR).decode().strip()
    print(f"Current REAP commit: {commit_hash}")
    return commit_hash


def patch_reap_lazy_eval_import():
    prune_py = pathlib.Path(REAP_DIR) / "src" / "reap" / "prune.py"
    main_py = pathlib.Path(REAP_DIR) / "src" / "reap" / "main.py"

    patched_files = []

    for target_path in [prune_py, main_py]:
        text = target_path.read_text(encoding="utf-8")

        old_import = "from reap.eval import run_evaluate\n"
        old_eval_block = """    if reap_args.do_eval:\n        remove_hook_from_module(model, recurse=True)\n        model.to("cpu")\n        del model\n        del observer_data\n        torch.cuda.empty_cache()\n        gc.collect()\n        model_args.model_name = pruned_model_dir\n        run_evaluate(model_args, pruned_model_dir / "eval", eval_args, reap_args.seed)\n"""
        new_eval_block = """    if reap_args.do_eval:\n        from reap.eval import run_evaluate\n\n        remove_hook_from_module(model, recurse=True)\n        model.to("cpu")\n        del model\n        del observer_data\n        torch.cuda.empty_cache()\n        gc.collect()\n        model_args.model_name = pruned_model_dir\n        run_evaluate(model_args, pruned_model_dir / "eval", eval_args, reap_args.seed)\n"""

        changed = False
        if old_import in text:
            text = text.replace(old_import, "", 1)
            changed = True
        if old_eval_block in text:
            text = text.replace(old_eval_block, new_eval_block, 1)
            changed = True

        if changed:
            target_path.write_text(text, encoding="utf-8")
            patched_files.append(str(target_path))

    prune_text = prune_py.read_text(encoding="utf-8")
    old_transition_block = """    else:\n        logger.info(f"Pruning model to {total_experts - n_experts_to_prune} experts...")\n        prune(\n            observer_data,\n            model,\n            tokenizer,\n            reap_args,\n            prune_args,\n            n_experts_to_prune,\n            pruned_model_dir,\n        )\n"""
    new_transition_block = """    else:\n        logger.info("Materializing model on CPU before structural pruning...")\n        remove_hook_from_module(model, recurse=True)\n        model.to("cpu")\n        torch.cuda.empty_cache()\n        gc.collect()\n        logger.info(f"Pruning model to {total_experts - n_experts_to_prune} experts...")\n        prune(\n            observer_data,\n            model,\n            tokenizer,\n            reap_args,\n            prune_args,\n            n_experts_to_prune,\n            pruned_model_dir,\n        )\n"""
    old_save_block = """    pruned_model_dir.mkdir(parents=True, exist_ok=True)\n    start = time.time()\n    model.save_pretrained(pruned_model_dir)\n    end = time.time()\n"""
    new_save_block = """    pruned_model_dir.mkdir(parents=True, exist_ok=True)\n    if hasattr(model, "hf_device_map"):\n        delattr(model, "hf_device_map")\n    torch.cuda.empty_cache()\n    gc.collect()\n    state_dict = model.state_dict()\n    start = time.time()\n    model.save_pretrained(\n        pruned_model_dir,\n        state_dict=state_dict,\n        safe_serialization=True,\n        max_shard_size="5GB",\n    )\n    end = time.time()\n"""
    prune_changed = False
    if old_transition_block in prune_text:
        prune_text = prune_text.replace(old_transition_block, new_transition_block, 1)
        prune_changed = True
    if old_save_block in prune_text:
        prune_text = prune_text.replace(old_save_block, new_save_block, 1)
        prune_changed = True
    if prune_changed:
        prune_py.write_text(prune_text, encoding="utf-8")
        if str(prune_py) not in patched_files:
            patched_files.append(str(prune_py))

    if patched_files:
        print(f"Patched lazy eval import in: {', '.join(patched_files)}")


def _copy_if_exists(src: pathlib.Path, dst: pathlib.Path):
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied {src.name} -> {dst}")


def _parse_shard_name(path: pathlib.Path) -> tuple[str, int, int]:
    match = re.match(r"^(.*)-(\d+)-of-(\d+)\.safetensors$", path.name)
    if not match:
        raise ValueError(f"Unexpected shard filename format: {path.name}")
    prefix, current_str, total_str = match.groups()
    return prefix, int(current_str), int(total_str)


def _build_local_index_from_safetensors(pruned_model_dir: pathlib.Path) -> dict:
    from safetensors import safe_open

    weight_map = {}
    total_size = 0
    for shard_path in sorted(pruned_model_dir.glob("*.safetensors")):
        total_size += shard_path.stat().st_size
        with safe_open(shard_path, framework="pt", device="cpu") as source_file:
            for tensor_name in source_file.keys():
                weight_map[tensor_name] = shard_path.name

    return {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }


def _renumber_shards_with_new_total(pruned_model_dir: pathlib.Path) -> tuple[dict[str, str], str, int]:
    shard_paths = sorted(pruned_model_dir.glob("model-*.safetensors"))
    if not shard_paths:
        single_file = pruned_model_dir / "model.safetensors"
        if single_file.exists():
            shard_paths = [single_file]
    if not shard_paths:
        raise FileNotFoundError(f"No model shard files found in {pruned_model_dir}")

    if shard_paths[0].name == "model.safetensors":
        prefix = "model.safetensors"
        width = 5
        new_total = len(shard_paths) + 1

        temp_map = {}
        final_map = {}
        for index, old_path in enumerate(shard_paths, start=1):
            temp_path = old_path.with_name(f"{old_path.name}.tmp-renumber")
            old_path.rename(temp_path)
            new_name = f"{prefix}-{index:0{width}d}-of-{new_total:0{width}d}.safetensors"
            temp_map[temp_path.name] = new_name
            final_map[old_path.name] = new_name

        for temp_name, final_name in temp_map.items():
            (pruned_model_dir / temp_name).rename(pruned_model_dir / final_name)

        return final_map, prefix, width

    prefix, _, current_total = _parse_shard_name(shard_paths[0])
    width = len(str(current_total))
    new_total = len(shard_paths) + 1

    temp_map = {}
    final_map = {}
    for index, old_path in enumerate(shard_paths, start=1):
        temp_path = old_path.with_name(f"{old_path.name}.tmp-renumber")
        old_path.rename(temp_path)
        new_name = f"{prefix}-{index:0{width}d}-of-{new_total:0{width}d}.safetensors"
        temp_map[temp_path.name] = new_name
        final_map[old_path.name] = new_name

    for temp_name, final_name in temp_map.items():
        (pruned_model_dir / temp_name).rename(pruned_model_dir / final_name)

    return final_map, prefix, width


def _remove_stale_single_file(pruned_model_dir: pathlib.Path):
    stale_path = pruned_model_dir / "model.safetensors"
    if stale_path.exists():
        stale_path.unlink()
        print(f"Removed stale {stale_path}")


def infer_retained_expert_count(pruned_index: dict) -> int:
    expert_pattern = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
    layer_to_experts: dict[int, set[int]] = {}

    for tensor_name in pruned_index["weight_map"]:
        match = expert_pattern.match(tensor_name)
        if not match:
            continue
        layer_index = int(match.group(1))
        expert_index = int(match.group(2))
        layer_to_experts.setdefault(layer_index, set()).add(expert_index)

    if not layer_to_experts:
        raise RuntimeError("Could not infer retained expert count from pruned checkpoint weight map.")

    per_layer_counts = {layer_index: len(expert_ids) for layer_index, expert_ids in layer_to_experts.items()}
    unique_counts = sorted(set(per_layer_counts.values()))
    if len(unique_counts) != 1:
        raise RuntimeError(f"Inconsistent retained expert counts across layers: {per_layer_counts}")

    retained_count = unique_counts[0]
    print(f"Inferred retained expert count per layer: {retained_count}")
    return retained_count


def rewrite_pruned_text_config(pruned_model_dir: pathlib.Path, retained_experts: int, pruned_index: dict):
    config_path = pruned_model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        pruned_config = json.load(f)

    text_config = pruned_config.setdefault("text_config", {})
    before = text_config.get("num_experts")
    text_config["num_experts"] = retained_experts

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(pruned_config, f, indent=2, ensure_ascii=True)
        f.write("\n")

    print(f"Updated text_config.num_experts: {before} -> {retained_experts}")

    manifest_path = pruned_model_dir / "multimodal_assets_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    manifest["retained_experts_per_layer"] = retained_experts
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)
        f.write("\n")

    final_count = pruned_config["text_config"].get("num_experts")
    if final_count != retained_experts:
        raise RuntimeError(
            f"Failed to rewrite pruned config num_experts to {retained_experts}; found {final_count}."
        )

    if infer_retained_expert_count(pruned_index) != retained_experts:
        raise RuntimeError("Retained expert count validation failed after config rewrite.")


def preserve_multimodal_assets(pruned_model_dir: pathlib.Path):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    print(f"Preserving multimodal assets in {pruned_model_dir} ...")

    original_config_path = pathlib.Path(hf_hub_download(MODEL_NAME, "config.json", repo_type="model"))
    original_index_path = pathlib.Path(
        hf_hub_download(MODEL_NAME, "model.safetensors.index.json", repo_type="model")
    )

    with original_config_path.open("r", encoding="utf-8") as f:
        source_config = json.load(f)

    pruned_config_path = pruned_model_dir / "config.json"
    with pruned_config_path.open("r", encoding="utf-8") as f:
        pruned_config = json.load(f)

    for key in [
        "architectures",
        "vision_config",
        "vision_start_token_id",
        "vision_end_token_id",
        "image_token_id",
        "video_token_id",
    ]:
        if key in source_config:
            pruned_config[key] = source_config[key]

    pruned_config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]

    with pruned_config_path.open("w", encoding="utf-8") as f:
        json.dump(pruned_config, f, indent=2, ensure_ascii=True)
        f.write("\n")

    for filename in MULTIMODAL_METADATA_FILES:
        source_path = pathlib.Path(hf_hub_download(MODEL_NAME, filename, repo_type="model"))
        _copy_if_exists(source_path, pruned_model_dir / filename)

    with original_index_path.open("r", encoding="utf-8") as f:
        source_index = json.load(f)

    pruned_index_path = pruned_model_dir / "model.safetensors.index.json"
    if pruned_index_path.exists():
        with pruned_index_path.open("r", encoding="utf-8") as f:
            pruned_index = json.load(f)
    else:
        pruned_index = _build_local_index_from_safetensors(pruned_model_dir)

    visual_tensors = {}
    visual_metadata = {}
    source_shards = sorted(
        {
            shard_name
            for tensor_name, shard_name in source_index["weight_map"].items()
            if tensor_name.startswith(VISUAL_TENSOR_PREFIX)
        }
    )

    for shard_name in source_shards:
        shard_path = pathlib.Path(hf_hub_download(MODEL_NAME, shard_name, repo_type="model"))
        with safe_open(shard_path, framework="pt", device="cpu") as source_file:
            for tensor_name in source_file.keys():
                if tensor_name.startswith(VISUAL_TENSOR_PREFIX):
                    visual_tensors[tensor_name] = source_file.get_tensor(tensor_name)
                    tensor_slice = source_file.get_slice(tensor_name)
                    visual_metadata[tensor_name] = {
                        "dtype": str(visual_tensors[tensor_name].dtype),
                        "shape": list(tensor_slice.get_shape()),
                    }

    if not visual_tensors:
        raise RuntimeError(f"No visual tensors found for {MODEL_NAME}.")

    shard_rename_map, shard_prefix, shard_width = _renumber_shards_with_new_total(pruned_model_dir)
    visual_output_shard = (
        f"{shard_prefix}-{(len(shard_rename_map) + 1):0{shard_width}d}-of-{(len(shard_rename_map) + 1):0{shard_width}d}.safetensors"
    )
    visual_output_path = pruned_model_dir / visual_output_shard
    save_file(visual_tensors, str(visual_output_path), metadata={"source_model": MODEL_NAME})
    print(f"Wrote {len(visual_tensors)} visual tensors to {visual_output_path}")

    for tensor_name, shard_name in list(pruned_index["weight_map"].items()):
        if shard_name in shard_rename_map:
            pruned_index["weight_map"][tensor_name] = shard_rename_map[shard_name]

    for tensor_name in visual_tensors:
        pruned_index["weight_map"][tensor_name] = visual_output_shard

    total_size = 0
    for shard_path in pruned_model_dir.glob("*.safetensors"):
        total_size += shard_path.stat().st_size
    pruned_index["metadata"] = pruned_index.get("metadata", {})
    pruned_index["metadata"]["total_size"] = total_size
    pruned_index["metadata"]["multimodal_fix"] = "vision tensors restored into standard final shard"
    pruned_index["metadata"]["base_model_for_multimodal_assets"] = MODEL_NAME

    with pruned_index_path.open("w", encoding="utf-8") as f:
        json.dump(pruned_index, f, indent=2, ensure_ascii=True)
        f.write("\n")

    retained_experts = infer_retained_expert_count(pruned_index)

    manifest_path = pruned_model_dir / "multimodal_assets_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_model": MODEL_NAME,
                "visual_tensor_prefix": VISUAL_TENSOR_PREFIX,
                "visual_output_shard": visual_output_shard,
                "source_shards": source_shards,
                "renamed_shards": shard_rename_map,
                "visual_tensor_count": len(visual_tensors),
                "visual_tensors": visual_metadata,
                "retained_experts_per_layer": retained_experts,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )
        f.write("\n")

    _remove_stale_single_file(pruned_model_dir)
    rewrite_pruned_text_config(pruned_model_dir, retained_experts, pruned_index)
    print("Multimodal assets preserved successfully.")


def get_latest_pruned_dir() -> pathlib.Path:
    model_clean = MODEL_NAME.split("/")[-1]
    model_root = pathlib.Path("/results") / model_clean
    if not model_root.exists():
        raise FileNotFoundError(f"Could not find model root at {model_root}")

    pruned_dirs = [path for path in model_root.glob("*/pruned_models/*") if path.is_dir()]
    if not pruned_dirs:
        pruned_dirs = [path for path in model_root.glob("**/pruned_models/*") if path.is_dir()]
    if not pruned_dirs:
        raise FileNotFoundError(f"No pruned model directories found under {model_root}")

    return sorted(pruned_dirs, key=os.path.getmtime)[-1]


@app.function(
    image=image,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    secrets=[huggingface_secret],
    timeout=60 * MINUTES,
)
def download_model():
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_NAME} ...")
    snapshot_download(
        MODEL_NAME,
        ignore_patterns=["*.pt", "*.bin"],
    )
    print(f"Finished downloading {MODEL_NAME}.")
    hf_cache_vol.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={HF_CACHE_DIR: hf_cache_vol, RESULTS_DIR: results_vol},
    secrets=[huggingface_secret],
    timeout=120 * MINUTES,
)
def run_observer(dataset_name: str, model_max_length: int, seed: int):
    import subprocess

    sync_and_show_commit()

    if not os.path.exists("artifacts"):
        os.symlink("/results", "artifacts")

    cmd = [
        "python",
        "-m",
        "reap.prune",
        "--model_name",
        MODEL_NAME,
        "--dataset_name",
        dataset_name,
        "--run_observer_only",
        "true",
        "--model_max_length",
        str(model_max_length),
        "--record_pruning_metrics_only",
        "true",
        "--seed",
        str(seed),
    ]
    subprocess.run(cmd, check=True)
    results_vol.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={HF_CACHE_DIR: hf_cache_vol, RESULTS_DIR: results_vol},
    secrets=[huggingface_secret],
    timeout=240 * MINUTES,
)
def run_pruning(
    compression_ratio: float,
    dataset_name: str,
    model_max_length: int,
    seed: int,
    renormalize_router_weights: bool,
):
    import subprocess

    sync_and_show_commit()

    if not os.path.exists("artifacts"):
        os.symlink("/results", "artifacts")

    cmd = [
        "python",
        "-m",
        "reap.prune",
        "--model_name",
        MODEL_NAME,
        "--dataset_name",
        dataset_name,
        "--compression_ratio",
        str(compression_ratio),
        "--prune_method",
        PRUNE_METHOD,
        "--do_eval",
        "false",
        "--model_max_length",
        str(model_max_length),
        "--seed",
        str(seed),
        "--renormalize_router_weights",
        _as_cli_bool(renormalize_router_weights),
        "--overwrite_pruned_model",
        "true",
    ]
    subprocess.run(cmd, check=True)

    pruned_model_dir = get_latest_pruned_dir()
    preserve_multimodal_assets(pruned_model_dir)
    results_vol.commit()


@app.function(
    image=image,
    volumes={RESULTS_DIR: results_vol},
    secrets=[huggingface_secret],
    timeout=90 * MINUTES,
)
def upload_to_hf(repo_id: str):
    from huggingface_hub import HfApi

    pruned_path = get_latest_pruned_dir()
    print(f"Uploading {pruned_path} to HF repo {repo_id} ...")

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=str(pruned_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Upload complete: https://huggingface.co/{repo_id}")


@app.local_entrypoint()
def main(
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
    dataset_name: str = DEFAULT_DATASET_NAME,
    model_max_length: int = DEFAULT_MODEL_MAX_LENGTH,
    seed: int = DEFAULT_SEED,
    renormalize_router_weights: bool = DEFAULT_RENORMALIZE_ROUTER_WEIGHTS,
    prune_only: bool = False,
):
    print(
        "Prune configuration:\n"
        f" model={MODEL_NAME}\n"
        f" dataset_name={dataset_name}\n"
        f" compression_ratio={compression_ratio}\n"
        f" prune_method={PRUNE_METHOD}\n"
        f" seed={seed}\n"
        f" model_max_length={model_max_length}\n"
        f" renormalize_router_weights={renormalize_router_weights}"
    )

    if not prune_only:
        download_model.remote()
        run_observer.remote(dataset_name, model_max_length, seed)

    run_pruning.remote(
        compression_ratio,
        dataset_name,
        model_max_length,
        seed,
        renormalize_router_weights,
    )

    if hf_repo_id:
        upload_to_hf.remote(hf_repo_id)
    else:
        print("No --hf-repo-id provided, skipping upload.")
