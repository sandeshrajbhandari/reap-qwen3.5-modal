"""
Modal runner for Qwen3.5-4B + residual-stream TopK SAE steering (Qwen-Scope-style).

Official Qwen-Scope SAEs on Hugging Face (e.g. Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50)
target Qwen3.5-27B (d_model=5120). This script defaults to Qwen/Qwen3.5-4B with the
public layer-18 residual TopK SAE `caiovicentino1/Qwen3.5-4B-SAE-L18-topk`, which
ships `sae_final.pt` and documents the same encode → decode delta steering pattern
as the Qwen-Scope demo.

Iterate with:
  modal run modal_qwen_scope_sae.py::main -- --mode discover
  modal run modal_qwen_scope_sae.py::main -- --mode generate --feature-ids 12345,6789 --scales 1,3,8
  modal run modal_qwen_scope_sae.py::main -- --mode sweep --top-k 12 --scales 1,2,4,8

The sweep mode ranks features by mean SAE activation on pirate-style strings minus
neutral strings (last token), then steers each of the top five features alone and
the top three together, plus an unsteered baseline (scale 1.0, no-op hook).

Requires a Modal secret `huggingface-secret` with `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Sequence

import modal

app = modal.App("qwen-scope-sae-steer")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124",
        "pip install --no-cache-dir 'transformers>=4.51.0' 'accelerate>=1.0.0' "
        "'huggingface_hub>=0.26.0' sentencepiece",
    )
)

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_SAE_REPO = "caiovicentino1/Qwen3.5-4B-SAE-L18-topk"
DEFAULT_SAE_FILE = "sae_final.pt"
SAE_LAYER = 18
# hidden_states index is 1 + layer index for post-block residual in this setup
HIDDEN_STATE_INDEX = SAE_LAYER + 1


@dataclass(frozen=True)
class SAEWeights:
    W_enc: "object"
    W_dec: "object"
    b_enc: "object"
    b_dec: "object"
    k: int


def _load_sae_tensors(device: str, dtype: "object", repo: str, filename: str) -> SAEWeights:
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo, filename=filename)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    k = int(ckpt["k"])
    return SAEWeights(
        W_enc=ckpt["W_enc"].to(device=device, dtype=dtype),
        W_dec=ckpt["W_dec"].to(device=device, dtype=dtype),
        b_enc=ckpt["b_enc"].to(device=device, dtype=dtype),
        b_dec=ckpt["b_dec"].to(device=device, dtype=dtype),
        k=k,
    )


def _encode(x: "torch.Tensor", sae: SAEWeights) -> "torch.Tensor":
    import torch

    pre = (x - sae.b_dec) @ sae.W_enc + sae.b_enc
    topv, topi = torch.topk(pre, sae.k, dim=-1)
    out = torch.zeros_like(pre)
    out.scatter_(-1, topi, topv)
    return torch.relu(out)


def _decode(f: "torch.Tensor", sae: SAEWeights) -> "torch.Tensor":
    return f @ sae.W_dec + sae.b_dec


def _resolve_layers_path(model) -> str:
    """Prefer language_model.layers (Qwen3.5) over legacy model.model.layers."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError("Expected causal LM with .model backbone.")
    lm = getattr(inner, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        return "model.language_model.layers"
    if hasattr(inner, "layers"):
        return "model.layers"
    raise RuntimeError("Could not find transformer layers on model for hooking.")


def _steering_hook_factory(sae: SAEWeights, feature_scales: dict[int, float]):
    """Per-feature multiplicative steering in SAE space (scale 1.0 = no change)."""
    import torch

    def steering_hook(module, inputs, output):
        hidden, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        flat = hidden.float().view(-1, hidden.shape[-1])
        feats = _encode(flat, sae)
        feats_mod = feats.clone()
        for fid, sc in feature_scales.items():
            if fid < 0 or fid >= feats_mod.shape[-1]:
                continue
            feats_mod[:, fid] = feats[:, fid] * float(sc)
        delta = _decode(feats_mod, sae) - _decode(feats, sae)
        new = (flat + delta).view_as(hidden).to(hidden.dtype)
        if rest is not None:
            return (new, *rest)
        return new

    return steering_hook


def _pirate_lexicon_score(text: str) -> dict:
    """Cheap heuristic for iteration: count pirate-flavored tokens (not ground truth)."""
    low = text.lower()
    hits = [
        "arr",
        "ahoy",
        "matey",
        "ye ",
        " yer ",
        "ye'",
        "shiver",
        "timbers",
        "scurvy",
        "landlubber",
        "captain",
        "crew",
        "treasure",
        "sea",
        "ship",
        "sail",
        "plunder",
        "davy jones",
    ]
    score = sum(low.count(h) for h in hits)
    return {"pirate_lexicon_hits": score}


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_experiment(
    mode: str,
    model_id: str,
    sae_repo: str,
    sae_file: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    feature_ids: list[int],
    scales: list[float],
    top_k: int,
    seed: int,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA required for this experiment.")

    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    sae = _load_sae_tensors(device, torch.float32, sae_repo, sae_file)
    layers_path = _resolve_layers_path(model)
    target = model.get_submodule(f"{layers_path}.{SAE_LAYER}")

    pirate_prompts = [
        "Ahoy, ye scurvy dog! Speak like a pirate: what be the best way to split treasure?",
        "Arrr! Shiver me timbers — explain navigation by the stars, matey.",
        "Ye landlubber, tell me why the black flag flies o'er the seven seas.",
    ]
    neutral_prompts = [
        "Briefly explain what a sparse autoencoder is.",
        "List three benefits of unit tests in software engineering.",
        "What is the capital of France and one fact about it?",
    ]

    def last_token_hidden(text: str) -> torch.Tensor:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out = model(
                input_ids=ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        h = out.hidden_states[HIDDEN_STATE_INDEX].float().squeeze(0)
        return h[-1]

    def discover() -> dict:
        import torch

        acc_p = None
        acc_n = None
        with torch.inference_mode():
            for t in pirate_prompts:
                v = last_token_hidden(t)
                f = _encode(v.unsqueeze(0), sae).squeeze(0)
                acc_p = f if acc_p is None else acc_p + f
            for t in neutral_prompts:
                v = last_token_hidden(t)
                f = _encode(v.unsqueeze(0), sae).squeeze(0)
                acc_n = f if acc_n is None else acc_n + f
        mean_p = acc_p / len(pirate_prompts)
        mean_n = acc_n / len(neutral_prompts)
        diff = mean_p - mean_n
        topv, topi = torch.topk(diff, min(top_k, diff.numel()))
        ranked = [
            {"feature_id": int(i), "pirate_minus_neutral": float(v)}
            for v, i in zip(topv.tolist(), topi.tolist())
        ]
        return {"ranked_features": ranked}

    def generate_with_scales(feat_ids: Sequence[int], scale_list: Sequence[float]) -> dict:
        import torch

        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
        else:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        results = []
        for sc in scale_list:
            scales_map = {int(fid): float(sc) for fid in feat_ids}
            hook = target.register_forward_hook(_steering_hook_factory(sae, scales_map))
            try:
                gen_kw = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if temperature and temperature > 0:
                    gen_kw["do_sample"] = True
                    gen_kw["temperature"] = max(1e-5, float(temperature))
                    gen_kw["top_p"] = float(top_p)
                else:
                    gen_kw["do_sample"] = False
                with torch.inference_mode():
                    out_ids = model.generate(input_ids, **gen_kw)
                text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            finally:
                hook.remove()
            row = {"scale": float(sc), "text": text}
            row.update(_pirate_lexicon_score(text))
            results.append(row)
        return {"prompt": prompt, "feature_ids": list(feat_ids), "runs": results}

    if mode == "discover":
        return discover()
    if mode == "generate":
        if not feature_ids:
            raise ValueError("generate mode requires --feature-ids")
        if not scales:
            scales = [1.0, 3.0, 8.0]
        return generate_with_scales(feature_ids, scales)
    if mode == "sweep":
        disc = discover()
        ids = [r["feature_id"] for r in disc["ranked_features"][: max(1, top_k)]]
        if not scales:
            scales = [1.0, 2.0, 4.0, 8.0]
        baseline = generate_with_scales([], [1.0])
        per_feature = []
        for fid in ids[: min(5, len(ids))]:
            per_feature.append(
                {
                    "feature_id": fid,
                    "runs": generate_with_scales([fid], scales)["runs"],
                }
            )
        bundle = generate_with_scales(ids[: min(3, len(ids))], scales)
        return {
            "discovered": disc,
            "baseline": baseline,
            "per_feature_top5": per_feature,
            "bundle_top3": bundle,
        }
    raise ValueError(f"Unknown mode: {mode}")


@app.local_entrypoint()
def main(
    mode: str = "sweep",
    model_id: str = DEFAULT_MODEL,
    sae_repo: str = DEFAULT_SAE_REPO,
    sae_file: str = DEFAULT_SAE_FILE,
    prompt: str = "Explain in one short paragraph how a compass works at sea.",
    max_new_tokens: int = 180,
    temperature: float = 0.7,
    top_p: float = 0.9,
    feature_ids: str = "",
    scales: str = "1,2,4,8",
    top_k: int = 8,
    seed: int = 42,
):
    parser = argparse.ArgumentParser(description="Qwen3.5-4B SAE steering on Modal")
    parser.add_argument("--mode", default=mode, choices=["discover", "generate", "sweep"])
    parser.add_argument("--model-id", default=model_id)
    parser.add_argument("--sae-repo", default=sae_repo)
    parser.add_argument("--sae-file", default=sae_file)
    parser.add_argument("--prompt", default=prompt)
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--top-p", type=float, default=top_p)
    parser.add_argument("--feature-ids", default=feature_ids, help="Comma-separated, for generate mode")
    parser.add_argument("--scales", default=scales, help="Comma-separated multipliers in SAE space")
    parser.add_argument("--top-k", type=int, default=top_k)
    parser.add_argument("--seed", type=int, default=seed)
    args, _unknown = parser.parse_known_args()

    def parse_csv_floats(s: str) -> list[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    def parse_csv_ints(s: str) -> list[int]:
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    fids = parse_csv_ints(args.feature_ids) if args.feature_ids else []
    scs = parse_csv_floats(args.scales) if args.scales else []

    out = run_experiment.remote(
        mode=args.mode,
        model_id=args.model_id,
        sae_repo=args.sae_repo,
        sae_file=args.sae_file,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        feature_ids=fids,
        scales=scs,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
