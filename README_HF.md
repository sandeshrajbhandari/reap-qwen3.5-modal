---
language:
- en
library_name: transformers
tags:
- qwen
- MOE
- pruning
- compression
- GGUF
license: apache-2.0
name: sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32
description: >
  This model was obtained by uniformly pruning 32% of experts in Qwen3.5-35B-A3B using the REAP method.
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.5-35B-A3B
---

<p align="center">
  <em>𓌳 <strong>REAP</strong>𓌳  the Experts: Why Pruning Prevails for One-Shot MoE Compression</em><br>
</p>

# Qwen3.5-24B-A3B-REAP-0.32

## ✨ Highlights

Introducing **Qwen3.5-24B-A3B-REAP-0.32**, a **memory-efficient compressed variant** of Qwen3.5-35B-A3B that maintains the core reasoning and coding capabilities of the architecture while being **32% lighter**.

This model was created using **REAP (Router-weighted Expert Activation Pruning)**, a novel expert pruning method that selectively removes redundant experts while preserving the router's independent control over remaining experts. Key features include:

- **Aggressive Compression**: 32% reduction in expert count, bringing the total parameter count down to approximately 24B.
- **3B Active Parameters**: Maintains the same computational efficiency during inference as the original model (3B parameters activated per token).
- **High-Precision GGUF**: Includes optimized quants using an importance matrix (imatrix) and custom tensor precision recipes.
- **Drop-in Compatibility**: Fully compatible with the latest `transformers` (from source) and `vLLM`.
- **Orchestration Scripts**: Full pipeline available at [sandeshrajbhandari/reap-qwen3.5-modal](https://github.com/sandeshrajbhandari/reap-qwen3.5-modal).

---

## 📋 Model Overview

**Qwen3.5-24B-A3B-REAP-0.32** has the following specifications:

- **Base Model**: [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- **Compression Method**: REAP (Router-weighted Expert Activation Pruning)
- **Compression Ratio**: 32% expert pruning
- **Type**: Sparse Mixture-of-Experts (SMoE) Causal Language Model
- **Number of Parameters**: ~24B total, ~3B activated per token
- **Number of Experts**: 175 (uniformly pruned from 256)
- **Number of Activated Experts**: 8 per token
- **License**: Apache 2.0

---

## 📂 Repository Contents

This repository contains the following artifacts:

- **`model-f16.gguf`**: The full-precision GGUF conversion of the pruned model.
- **`Qwen3.5-24B-A3B-REAP-0.32-IQ4_K_M.gguf`**: High-precision 4-bit quant using the Unsloth-style recipe (imatrix + Q8_0 overrides for critical tensors).
- **`Qwen3.5-24B-A3B-REAP-0.32-IQ4_K_S.gguf`**: Smaller 4-bit quant variant.
- **`imatrix.dat`**: The importance matrix used for quantization.
- **`calibration_data_v5_rc.txt`**: The calibration corpus used to generate the imatrix.

---

## 🚀 Deployment

### Transformers
Since Qwen 3.5 MoE is a new architecture, ensure you are using the latest `transformers` from source:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

### vLLM
You can deploy the model directly using **vLLM**:

```bash
vllm serve sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32 \
    --enable-expert-parallel
```

### GGUF (llama.cpp)
Optimized GGUF versions are available in this repository. We recommend using the `IQ4_K_M` variant for the best balance of size and performance.

---

## 🧩 Model Creation

### How REAP Works
REAP selects experts to prune based on a **saliency criterion** that considers router gate values and expert activation norms. This ensures that only experts contributing minimally to the model's internal representations are removed.

### Infrastructure
The project utilized **Modal** for high-memory compute (A100-80GB) and a custom fork of the REAP library.
- **Orchestration Code**: [reap-qwen3.5-modal](https://github.com/sandeshrajbhandari/reap-qwen3.5-modal)
- **Library Fork**: [sandeshrajbhandari/reap](https://github.com/sandeshrajbhandari/reap/tree/feat/qwen3.5-moe-support)

### ⚠️ Caveats & Future Work
- **Compute Constraints**: Due to current memory limitations, the model was calibrated with a context length of **1024 tokens** and a limited sample size. 
- **Room for Optimization**: There is significant room for improvement by using larger sample sizes and the full 2048/4096 context length. The current REAP fork for Qwen 3.5 still hits OOM on 80GB VRAM at 2048 context length during activation profiling, which is a target for future optimization.

---

## 📚 References & Resources

### 🔧 GGUF & Quantization Guides
- [Overview of GGUF quantization methods](https://www.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview_of_gguf_quantization_methods/)
- [Quant Cookers Basic Guide](https://github.com/ikawrakow/ik_llama.cpp/discussions/434)

### 📊 Benchmarks & Comparisons
- [Qwen3.5-35B-A3B Q4 Quantization Comparison](https://www.reddit.com/r/LocalLLaMA/comments/1rfds1h/qwen3535ba3b_q4_quantization_comparison/)
- [Qwen3.5 GGUF Benchmarks (Unsloth)](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)

### 🎯 Research
- [REAP arXiv Preprint](https://arxiv.org/abs/2510.13999)
- [REAP Blog (Cerebras)](https://www.cerebras.ai/blog/reap-one-shot-pruning-for-trillion-parameter-mixture-of-experts-models)

---

## ⚖️ License

This model is derived from **[`Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)** and distributed under the **Apache 2.0 License**.

---

## 🧾 Citation

```bibtex
@article{lasby-reap,
  title={REAP the Experts: Why Pruning Prevails for One-Shot MoE compression},
  author={Lasby, Mike and Lazarevich, Ivan and Sinnadurai, Nish and Lie, Sean and Ioannou, Yani and Thangarasa, Vithursan},
  journal={arXiv preprint arXiv:2510.13999},
  year={2025}
}
```
