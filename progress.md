# Project Progress Report: REAP Pruning and High-Precision Quantization on Modal

## Overview
The goal of this project was to implement and execute the REAP (Resource-Efficient Activation-based Pruning) pipeline for large-scale Mixture-of-Experts (MoE) models using Modal's serverless infrastructure, followed by high-precision GGUF quantization. We focused on `Qwen/Qwen3.5-35B-A3B`.

## Work Performed

### 1. Infrastructure Scaffolding
- Developed `prune_qwen.py` to orchestrate the REAP pipeline.
- Developed `generate_imatrix.py` to build `llama.cpp` with CUDA support and generate Importance Matrices (imatrix).
- Developed `quantize_modal_IQ.py` and `quantize_modal_IQS.py` for "Unsloth-style" high-precision quantization.
- Configured Modal Volumes for persistent caching of weights, tools, and artifacts.

### 2. Pruning & Model Support
- **Qwen 3.5 35B-A3B**: Integrated support for the `qwen3_5_moe` architecture.
- Successfully pruned the model by 32% (uniformly across layers).
- Patched REAP codebase for Qwen-specific attributes (`gate` vs `router`) and return types.

### 3. High-Precision Quantization (GGUF)
- **Importance Matrix**: Generated a 132MB `imatrix.dat` using a specific calibration corpus to preserve critical weights.
- **Unsloth-style Recipe**: Implemented a custom quantization recipe that forces critical tensors (attention gates, shared experts, etc.) into 8-bit (`Q8_0`) while keeping routed experts at `Q4_K` or `Q5_K`.
- **Multiple Variants**: Produced both `IQ4_K_M` and `IQ4_K_S` variants for optimal performance-to-size ratios.

## Challenges & Fixes

### Hardware & Memory (OOM)
- **Challenge**: Standard A100 (40GB) hit OOM during 35B model loading and activation profiling.
  - **Fix**: Upgraded all heavy tasks to **A100-80GB**.
- **Challenge**: Context length of 2048 was too heavy for profiling.
  - **Fix**: Reduced `model_max_length` to 1024.

### GGUF Conversion & Quantization
- **Challenge**: `llama.cpp` converter did not recognize `Qwen3_5MoeForCausalLM`.
  - **Fix**: Implemented a temporary patch in `config.json` to change the architecture name to `Qwen3_5MoeForConditionalGeneration` during conversion.
- **Challenge**: `llama-quantize` failed due to missing `libcudart.so` and `libllama.so`.
  - **Fix**: Switched to a full CUDA-devel base image and configured `LD_LIBRARY_PATH` / `ldconfig` to correctly locate shared libraries in the volume.

### Upload & Reliability
- **Challenge**: Uploading a single 50GB GGUF file was slow and prone to failure.
  - **Fix**: Implemented a sharding script (`upload_to_hf.py`) to split the model into 5GB pieces for robust Hugging Face transfers.

## Final Status
- **Pruned Model**: Successfully uploaded sharded Safetensors to [sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32](https://huggingface.co/sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32).
- **GGUF Master Repo**: Consolidated F16, imatrix, and IQ quants at [sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32-GGUF](https://huggingface.co/sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32-GGUF).
