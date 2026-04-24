# Modal Agent Kickstart Reference (Finetuning + Inference + Volumes)

This is a practical reference you can hand to an autonomous coding/ML agent at the start of a new session.
It focuses on **repeatable Modal workflows** for:

- finetuning jobs
- inference serving
- persistent storage with **Modal Volumes**
- secrets and environment setup
- resumable/restart-safe patterns

---

## 1) One-Time Setup

### Install and authenticate Modal

```bash
pip install -U modal
modal setup
```

### Verify account/workspace

```bash
modal token info
```

### Optional: set default environment/workspace behavior

If you use multiple workspaces/environments, pass `--env` on commands or configure your preferred default in your shell/profile.

---

## 2) Core Concepts You Should Use in Every Agent Workflow

### A) `modal.App`

Your app is the deployment/execution unit. Keep one app per workflow family:

- `finetune-*` apps
- `inference-*` apps
- `data-*` apps

### B) Container image (`modal.Image`)

Pin your base image and install exact runtime deps in code so runs are reproducible.

### C) Volumes (`modal.Volume`)

Volumes are persistent, shared, and resumable storage for:

- model snapshots
- training checkpoints
- tokenizer cache
- quantized artifacts
- datasets/calibration files

### D) Secrets (`modal.Secret`)

Put API tokens in Modal secrets, not code. Typical secret:

- `huggingface-secret` with key `HF_TOKEN`

---

## 3) Volume Usage (CLI + Python)

### Create and inspect volumes (CLI)

```bash
modal volume create hf-cache
modal volume create train-checkpoints
modal volume create model-artifacts
modal volume ls
```

### Use a volume in Python

```python
import modal

app = modal.App("my-workflow")
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("train-checkpoints", create_if_missing=True)
artifacts_vol = modal.Volume.from_name("model-artifacts", create_if_missing=True)
```

### Mount volumes in a function

```python
@app.function(
    volumes={
        "/cache/hf": hf_cache_vol,
        "/checkpoints": ckpt_vol,
        "/artifacts": artifacts_vol,
    }
)
def task():
    ...
```

### Persist writes with `commit()`

When you write to mounted volume paths, call `volume.commit()` so downstream steps reliably see updates.

```python
ckpt_vol.commit()
artifacts_vol.commit()
```

### Recommended volume layout

Use stable subpaths:

- `/cache/hf/models/...`
- `/checkpoints/<run_name>/...`
- `/artifacts/<model_name>/<stage>/...`

---

## 4) Minimal Finetuning Template (LoRA-style)

```python
import os
import modal

app = modal.App("finetune-template")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "peft",
        "trl",
        "huggingface-hub",
        "sentencepiece",
    )
)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("train-checkpoints", create_if_missing=True)
results_vol = modal.Volume.from_name("model-artifacts", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("huggingface-secret")

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 12,
    secrets=[HF_SECRET],
    volumes={
        "/cache/hf": hf_cache_vol,
        "/checkpoints": ckpt_vol,
        "/results": results_vol,
    },
)
def finetune(base_model: str, dataset_repo: str, run_name: str):
    # 1) load dataset/model (use HF cache paths)
    # 2) train + save checkpoints in /checkpoints/<run_name>
    # 3) save final adapter/model in /results/<run_name>
    # 4) commit volumes
    ckpt_vol.commit()
    results_vol.commit()

@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    dataset_repo: str = "org/dataset",
    run_name: str = "run-001",
):
    finetune.remote(base_model, dataset_repo, run_name)
```

Run:

```bash
modal run finetune_template.py
```

---

## 5) Minimal Inference Template

Use an endpoint for HTTP inference and a separate batch function for offline eval.

```python
import modal

app = modal.App("inference-template")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.12")
    .pip_install("vllm", "transformers", "huggingface-hub")
)

models_vol = modal.Volume.from_name("model-artifacts", create_if_missing=True)

@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/models": models_vol},
    min_containers=1,
)
class InferenceService:
    @modal.enter()
    def load(self):
        # load model from /models/... or from HF
        pass

    @modal.method()
    def generate(self, prompt: str) -> str:
        return "..."

@app.local_entrypoint()
def main(prompt: str = "Hello"):
    svc = InferenceService()
    print(svc.generate.remote(prompt))
```

---

## 6) Multi-Stage Pipeline Pattern (Recommended)

For large model workflows, split into stages with separate functions:

1. `download_model()`
2. `convert_to_f16_gguf()`
3. `generate_imatrix()`
4. `quantize()`
5. `upload_artifacts()`

Each stage:

- reads from volume
- writes outputs to volume
- calls `volume.commit()`

This makes retries cheap and avoids rerunning expensive steps.

---

## 7) Secrets Pattern

Create secret:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

Use it:

```python
@app.function(secrets=[modal.Secret.from_name("huggingface-secret")])
def task():
    ...
```

---

## 8) Reliability Checklist for Agents

When prompting agents for Modal tasks, ask them to:

1. **Reuse existing volumes** before downloading again.
2. Check expected files exist before expensive stages.
3. Use long enough `timeout` for model conversion/quantization.
4. Persist outputs with `commit()`.
5. Print exact artifact paths and final URLs.
6. Keep stage functions idempotent (safe to rerun).

---

## 9) Agent Prompt Template (Copy/Paste)

Use this at the start of future sessions:

```text
Use Modal for this task with a resumable multi-stage pipeline.

Requirements:
- Reuse/create volumes:
  - hf-cache
  - train-checkpoints
  - model-artifacts
- Use secrets via modal.Secret (huggingface-secret for HF_TOKEN).
- Split workflow into explicit stages and commit volumes after each stage.
- Add clear logs with exact input/output paths.
- If artifacts are uploaded, print final links.
- Prefer restart-safe behavior (skip completed stages when files already exist).

Deliverables:
- Final Modal script(s)
- Exact `modal run ...` command(s)
- Output artifact paths and URLs
```

---

## 10) Typical Commands You’ll Reuse

```bash
# run job
modal run <script>.py [args...]

# inspect running apps
modal app list

# stream logs
modal app logs <app-id> -f

# stop app
modal app stop <app-id>

# volume operations
modal volume ls
modal volume create <name>
```

---

## 11) Notes for Finetuning + Inference Together

- Keep **training** and **serving** apps separate.
- Store final adapters/weights in a shared artifacts volume.
- For GGUF multimodal deployments, publish both:
  - text model `.gguf`
  - matching `mmproj` `.gguf`
- In llama.cpp serving, always pass `--mmproj` for multimodal input.

---

If you want this doc tuned for your exact stack (Qwen + REAP pruning + GGUF quantization + vLLM serving), duplicate this file and hardcode your current volume names, model repos, and default run commands.
