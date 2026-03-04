# Pruning and High-Precision Quantization: Shrinking Qwen 3.5 MoE with REAP and Modal

Mixture-of-Experts (MoE) models like Qwen 3.5 are the current gold standard for performance-to-compute efficiency. However, their massive weight files—often exceeding 70GB—make them a challenge for deployment on consumer hardware. 

I recently completed a project to prune the **Qwen3.5-35B-A3B** model by 32%, resulting in a leaner version: **Qwen3.5-24B-A3B-REAP-0.32**. But I didn't stop at pruning. Using advanced GGUF quantization techniques, I've produced a highly optimized local-inference version that punches far above its weight class.

---

## The Core: Extending REAP for Qwen 3.5

The standard REAP implementation was built for earlier MoE architectures. To support the cutting-edge Qwen 3.5 series, I created a fork of the REAP repository and developed the **[`feat/qwen3.5-moe-support`](https://github.com/sandeshrajbhandari/reap/tree/feat/qwen3.5-moe-support)** branch. 

### Key Technical Fixes in the Fork:
1. **The "Gate" naming convention**: Updated `src/reap/prune.py` to resolve the routing layer using `getattr(moe, "router", getattr(moe, "gate", None))`.
2. **Handling Forward Pass Variations**: Patched `src/reap/observer.py` to detect single-tensor returns from `SparseMoeBlock` and wrap them in compatible tuples.
3. **Dtype Mismatch in Metrics**: Implemented explicit casting in `src/reap/metrics.py` to fix `RuntimeError` in `scatter_add_` during similarity computation.

---

## High-Precision Quantization: The Unsloth-Style Recipe

To ensure the pruned model didn't lose its "intelligence," I implemented a high-precision GGUF quantization pipeline based on the "Unsloth recipe."

### 1. Importance Matrix (imatrix)
I generated a custom **Importance Matrix** using `llama-imatrix` and a diverse calibration corpus. This tells the quantizer which weights are critical for reasoning, allowing it to prioritize precision for those specific parameters while compressing less important ones more aggressively.

### 2. Custom Tensor Precision
Instead of a uniform `Q4_K_M` quant, I used custom overrides to force critical components into **8-bit (`Q8_0`)**:
- **Attention Gates & QKV**: Preserves the core attention mechanism accuracy.
- **Shared Experts**: These are used in *every* token pass, so maintaining 8-bit precision is vital for stability.
- **Token Embeddings**: Improves vocabulary comprehension and retrieval.

---

## Overcoming Hardware Hurdles with Modal

Hardware limitations are the primary bottleneck in LLM research. Here is how Modal made the impossible possible:

- **Scaling the VRAM Wall**: Upgraded heavy profiling tasks to **A100-80GB** with a single line change.
- **GGUF Filesystem Hacks**: Automatically patched `config.json` at runtime to trick `llama.cpp` into recognizing the new Qwen 3.5 architecture during conversion.
- **Robust Sharded Uploads**: Developed a script to shard the 50GB+ Safetensors into 5GB pieces, ensuring reliable transfers to Hugging Face despite large file sizes.

---

## The Final Result

The successfully pruned and high-precision quantized models are now available on Hugging Face.

- **Pruned Safetensors**: [sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32](https://huggingface.co/sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32)
- **GGUF Master Repo**: [sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32-GGUF](https://huggingface.co/sandeshrajx/Qwen3.5-24B-A3B-REAP-0.32-GGUF)
- **Primary Artifact**: `Qwen3.5-24B-A3B-REAP-0.32-IQ4_K_M.gguf`
- **Orchestration Scripts**: [sandeshrajbhandari/reap-qwen3.5-modal](https://github.com/sandeshrajbhandari/reap-qwen3.5-modal)

*This project demonstrates that with the right pruning techniques, high-precision quantization, and serverless compute, we can make state-of-the-art MoE models accessible to everyone.*
