# REAP Pruning and High-Precision Quantization on Modal

A production-ready orchestration layer for pruning and quantizing large-scale Mixture-of-Experts (MoE) models using [REAP](https://github.com/CerebrasResearch/reap) and [Modal](https://modal.com).

This repository contains the scripts used to prune **Qwen3.5-35B-A3B** by 32% and generate high-precision GGUF quants with Importance Matrices (imatrix).

## 🚀 Quick Start for Contributors

### 1. Prerequisites
- A Modal account and the `modal` CLI installed (`pip install modal`).
- A Hugging Face token configured as a Modal Secret named `huggingface-secret` with the key `HF_TOKEN`.
- Access to high-memory GPUs (A100-80GB recommended for 35B+ models).

### 2. Core Workflow
The pipeline is split into independent, resumable stages powered by Modal Volumes.

#### Stage 1: Pruning
- **`prune_qwen.py`**: The main entry point for Qwen 3.5 MoE models.
- **`prune_gpt_oss.py`**: Optimized for the GPT-OSS MXFP4 architecture.
- **Key Feature**: Includes runtime repository syncing to pull the latest fixes from our [REAP fork](https://github.com/sandeshrajbhandari/reap/tree/feat/qwen3.5-moe-support) without rebuilding the image.

```bash
modal run prune_qwen.py --hf-repo-id username/your-pruned-model
```

#### Stage 2: Importance Matrix Generation
- **`generate_imatrix.py`**: Builds `llama.cpp` with CUDA support in a persistent volume and generates an `imatrix.dat` using a calibration corpus. This is essential for maintaining reasoning capabilities in small quants.

```bash
modal run generate_imatrix.py --rebuild # First run needs --rebuild
```

#### Stage 3: High-Precision Quantization
- **`quantize_modal_IQ.py`**: Implements the "Unsloth-style" recipe. It forces critical tensors (attention gates, shared experts) into 8-bit while quantizing the rest to 4-bit.
- **`quantize_modal_IQS.py`**: Variant for `IQ4_K_S` quantization.

```bash
modal run quantize_modal_IQ.py --hf-repo username/model-GGUF
```

### 3. Utility Scripts
- **`upload_to_hf.py`**: A robust sharding uploader that splits 50GB+ models into 5GB pieces for reliable Hugging Face transfers.
- **`upload_to_master.py`**: Consolidates all artifacts (F16, imatrix, quants, calibration data) into a single master GGUF repository.

## 🛠️ Key Technical Resources

- **`progress.md`**: A detailed project log detailing every hurdle (OOMs, architectural mismatches) and the corresponding fixes.
- **`blogpost.md`**: A high-level technical deep-dive into the "Why" and "How" of this project.
- **`links.md`**: A curated list of benchmarks, quantization guides, and research papers.

## 🤝 How to Contribute

We are actively looking to optimize the following:
1. **Memory Efficiency**: The current Qwen 3.5 profiling hits OOM on 80GB VRAM at 2048 context length. Optimizing the activation hooks in our REAP fork is a priority.
2. **Calibration Datasets**: Experimenting with different calibration corpora for the importance matrix to improve domain-specific performance (e.g., coding vs. creative writing).
3. **Automated Evaluation**: Integrating `lm-eval-harness` directly into the Modal pipeline to verify pruning quality in one shot.

---

*Built with ❤️ by the open-source community. Pruning giants, one expert at a time.*
