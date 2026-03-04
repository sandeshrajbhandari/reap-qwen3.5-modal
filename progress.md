# Project Progress Report: REAP Pruning on Modal

## Overview
The goal of this project was to implement and execute the REAP (Resource-Efficient Activation-based Pruning) pipeline for large-scale Mixture-of-Experts (MoE) models using Modal's serverless infrastructure. We focused on two models: `openai/gpt-oss-20b` and `Qwen/Qwen3.5-35B-A3B`.

## Work Performed

### 1. Infrastructure Scaffolding
- Developed `prune_gpt_oss.py` and `prune_qwen.py` to orchestrate the REAP pipeline on Modal.
- Configured Modal Volumes (`hf-cache-reap` and `reap-results`) for persistent caching of model weights and pruning artifacts.
- Implemented a four-step pipeline:
    1. **Download**: Caching weights using `snapshot_download`.
    2. **Observer**: Collecting activation statistics across dataset categories.
    3. **Pruning**: Pruning experts based on collected statistics.
    4. **Upload**: Automatically pushing the pruned model to Hugging Face.

### 2. Model Support
- **GPT-OSS 20B**: Configured environment for MXFP4 dequantization support.
- **Qwen 3.5 35B-A3B**: Integrated support for the new `qwen3_5_moe` architecture by installing `transformers` from source and patching the REAP codebase for Qwen-specific attributes.

## Challenges & Fixes

### Environment & Dependency Management
- **Challenge**: Local Python and Modal installations were not in the system path.
  - **Fix**: Manually configured `$env:PATH` in the session to include the Windows Store Python distribution and local-packages Scripts folder.
- **Challenge**: The REAP repository's `pyproject.toml` contained local path dependencies (e.g., `livecodebench`) that failed to resolve in the container.
  - **Fix**: Implemented `sed` commands in the Modal image build to dynamically remove problematic dependencies and manually installed required core packages via `uv`.
- **Challenge**: Missing system build tools for C-extensions (e.g., `zstandard`).
  - **Fix**: Added `clang` and `cmake` to the `apt_install` list.

### Hardware & Memory (OOM)
- **Challenge**: `L4` (24GB) and standard `A100` (40GB) GPUs hit `OutOfMemoryError` during model loading and dequantization.
  - **Fix**: Upgraded memory-intensive functions to use `A100-80GB` GPUs.
- **Challenge**: The observer phase hit OOM when processing activations at a context length of 2048.
  - **Fix**: Reduced `model_max_length` to 1024 and enabled `record_pruning_metrics_only` to minimize the VRAM footprint of activation tensors.

### Codebase Intercompatibility
- **Challenge**: REAP observer expected MoE blocks to return a tuple, but Qwen 3.5 blocks return a single tensor.
  - **Fix**: Patched `reap/observer.py` to handle both return types gracefully.
- **Challenge**: Qwen 3.5 uses the attribute name `gate` instead of `router` for its MoE routing layer.
  - **Fix**: Updated `reap/prune.py` to use `getattr(moe, "router", getattr(moe, "gate", None))`.
- **Challenge**: PyTorch `scatter_add_` failed due to a dtype mismatch between `bfloat16` activations and `float32` similarity matrices.
  - **Fix**: Patched `reap/metrics.py` to ensure explicit casting to the target dtype before the scatter operation.

### Runtime Consistency
- **Challenge**: Updates to the REAP repository were not always reflected in the Modal image due to caching.
  - **Fix**: Implemented `sync_and_show_commit()` at the start of Modal functions to force a `git fetch` and `reset --hard` to the latest commit on the branch at runtime.

## Final Status
- **Qwen3.5-35B-A3B**: Successfully pruned (32% compression) and uploaded to [sandeshrajx/qwen3.5-35b-reap-pruned](https://huggingface.co/sandeshrajx/qwen3.5-35b-reap-pruned).
- **GPT-OSS 20B**: Infrastructure is fully verified; observer data is collected and persisted in the `reap-results` volume.
