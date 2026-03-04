# Pruning the Giants: Shrinking Qwen 3.5 MoE with REAP and Modal

Mixture-of-Experts (MoE) models like Qwen 3.5 are the current gold standard for performance-to-compute efficiency. However, their massive weight files—often exceeding 70GB—make them a challenge for deployment on consumer hardware. 

I recently completed a project to prune the **Qwen3.5-35B-A3B** model by 32%, resulting in a leaner, more efficient version: **Qwen3.5-24B-A3B-REAP-0.32**. This was achieved using the **REAP (Resource-Efficient Activation-based Pruning)** method, orchestrated entirely on Modal's serverless infrastructure.

---

## The Core: Extending REAP for Qwen 3.5

The standard REAP implementation was built for earlier MoE architectures. To support the cutting-edge Qwen 3.5 series, I created a fork of the REAP repository and developed the **[`feat/qwen3.5-moe-support`](https://github.com/sandeshrajbhandari/reap/tree/feat/qwen3.5-moe-support)** branch. 

### Key Technical Fixes in the Fork:
1. **The "Gate" naming convention**: Standard MoE blocks in the original repo looked for a `.router` attribute. Qwen 3.5 uses `.gate`. I updated `src/reap/prune.py` to dynamically resolve the routing layer using `getattr(moe, "router", getattr(moe, "gate", None))`.
2. **Handling Forward Pass Variations**: The REAP observer hook expected MoE blocks to return a tuple (output, router_logits). Qwen 3.5's `SparseMoeBlock` forward pass often returns a single tensor. I patched `src/reap/observer.py` to detect single-tensor returns and wrap them in a compatible tuple format, preventing the observer from crashing.
3. **Dtype Mismatch in Metrics**: During the computation of similarity matrices, I encountered a `RuntimeError` in `scatter_add_`. Qwen 3.5 produces `bfloat16` activations, while the similarity matrix was `float32`. I implemented an explicit cast in `src/reap/metrics.py` to ensure `flat_dists.to(pairwise_distances.dtype)` before the scatter operation.

---

## Strategy for Fast Iteration

Pruning a 35B model can be a slow, expensive process. To iterate quickly and overcome compute limitations, I used a "Minimalist Parameters" strategy:

- **Ultra-Fast Sampling**: For the initial debug and verification runs, I set `--samples-per-category` to **1**. This allowed me to verify that the hooks were firing and the pruning logic was sound in under 5 minutes.
- **Context Constraints**: I reduced the `model_max_length` from the standard 2048 to **1024**. This significantly lowered the VRAM overhead required to store activation tensors during the "Observer" phase.
- **Record Pruning Metrics Only**: I enabled `record_pruning_metrics_only` to avoid saving unnecessary auxiliary data, keeping the Modal Volume usage lean.

---

## Overcoming Hardware Hurdles with Modal

Hardware limitations are the primary bottleneck in LLM research. Here is how Modal made the impossible possible:

### 1. Scaling the VRAM Wall
Initial attempts on **L4 (24GB)** and **A100 (40GB)** GPUs hit `OutOfMemoryError` walls due to the sheer size of the 35B model plus the activation recording overhead. With Modal, upgrading to an **A100-80GB** was a single line change in my Python decorator: `@app.function(gpu="A100-80GB")`.

### 2. The Persistent Caching Advantage
By using **Modal Volumes**, I cached the 70GB+ Hugging Face weights. This meant that even if a run failed due to a code bug, the next iteration started instantly without re-downloading the model.

### 3. The GGUF Conversion Trick
To convert the final pruned model to **Q4_K_M GGUF**, I used `llama.cpp`. However, the converter didn't recognize the `Qwen3_5MoeForCausalLM` model type. Within the Modal container, I implemented a filesystem hack:
- Automatically patched `config.json` to change the architecture to `Qwen3_5MoeForConditionalGeneration`.
- Ran the conversion.
- Restored the original metadata.
- Uploaded the result to Hugging Face.

---

## The Final Result

The successfully pruned and quantized model is now available on Hugging Face. It represents a 32% reduction in expert count while maintaining the core capabilities of the Qwen 3.5 architecture.

- **Hugging Face Repository**: [sandeshrajx/qwen3.5-35b-reap-pruned-GGUF](https://huggingface.co/sandeshrajx/qwen3.5-35b-reap-pruned-GGUF)
- **Primary Artifact**: `Qwen3.5-24B-A3B-REAP-0.32-Q4_K_M.gguf`

*This project demonstrates that with the right pruning techniques and a flexible serverless compute provider like Modal, we can make state-of-the-art MoE models accessible to everyone.*
