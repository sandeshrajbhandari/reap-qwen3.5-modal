import os
import textwrap

import modal


app = modal.App("qwen36-mtp-q3ks-quantizer")

image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.12")
    .apt_install("aria2", "git", "wget", "cmake", "build-essential", "libcurl4-openssl-dev")
    .pip_install(
        "cmake",
        "hf_transfer",
        "huggingface-hub",
        "numpy",
        "sentencepiece",
        "gguf",
    )
)

results_vol = modal.Volume.from_name("reap-results")
RESULTS_DIR = "/results"

IK_LLAMA_REPO = "https://github.com/ikawrakow/ik_llama.cpp.git"
IK_LLAMA_DIR = os.path.join(RESULTS_DIR, "tools/ik_llama.cpp")
IK_LLAMA_BUILD_DIR = os.path.join(IK_LLAMA_DIR, "build")
IK_LLAMA_BIN_DIR = os.path.join(IK_LLAMA_BUILD_DIR, "bin")

DEFAULT_SOURCE_REPO = "Radamanthys11/Qwen3.6-27B-MTP-Q8_0-GGUF"


def default_output_name(source_repo: str) -> str:
    name = source_repo.rstrip("/").split("/")[-1]
    for suffix in ("-Q8_0-GGUF", "-Q8_0.gguf", "-Q8_0"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"{name}-Q3_K_S.gguf"


def custom_q3ks_rules() -> str:
    # Keep the MTP/nextn tensors high precision for better speculative-decoding
    # acceptance. The remaining repeating-layer tensors follow a Q3_K_S-oriented
    # version of the referenced MTP recipe.
    rules = [
        r"blk\.64\..*\.weight=q8_0",
        r"blk\..*\.attn_gate\.weight=q3_k",
        r"blk\..*\.attn_qkv\.weight=q3_k",
        r"blk\..*\.attn_output\.weight=q3_k",
        r"blk\..*\.attn_q\.weight=q3_k",
        r"blk\..*\.attn_k\.weight=q3_k",
        r"blk\..*\.attn_v\.weight=q3_k",
        r"blk\..*\.ssm_alpha\.weight=q6_0",
        r"blk\..*\.ssm_beta\.weight=q6_0",
        r"blk\..*\.ssm_out\.weight=q6_0",
        r"blk\..*\.ffn_down\.weight=q3_k",
        r"blk\..*\.ffn_(gate|up)\.weight=q3_k",
        r"token_embd\.weight=q6_0",
        r"output\.weight=q8_0",
    ]
    return ",".join(rules)


@app.function(
    image=image,
    volumes={RESULTS_DIR: results_vol},
    timeout=14400,
)
def build_ik_llama_cpp(force_rebuild_ik_llama: bool = False, threads: int = 16):
    import shutil
    import subprocess

    def run(cmd, *, cwd=None):
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, text=True, cwd=cwd)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        return result

    quantize_bin = os.path.join(IK_LLAMA_BIN_DIR, "llama-quantize")
    if not force_rebuild_ik_llama and os.path.exists(quantize_bin):
        os.chmod(quantize_bin, 0o755)
        print(f"✅ Using existing ik_llama.cpp quantizer at {quantize_bin}")
        return

    if force_rebuild_ik_llama and os.path.exists(IK_LLAMA_DIR):
        print(f"🧹 Removing existing ik_llama.cpp checkout: {IK_LLAMA_DIR}")
        shutil.rmtree(IK_LLAMA_DIR)
    elif os.path.exists(IK_LLAMA_BUILD_DIR):
        print(f"🧹 Removing interrupted/stale ik_llama.cpp build dir: {IK_LLAMA_BUILD_DIR}")
        shutil.rmtree(IK_LLAMA_BUILD_DIR)

    if not os.path.exists(IK_LLAMA_DIR):
        os.makedirs(os.path.dirname(IK_LLAMA_DIR), exist_ok=True)
        print(f"⬇️ Cloning ik_llama.cpp from {IK_LLAMA_REPO}")
        run(["git", "clone", IK_LLAMA_REPO, IK_LLAMA_DIR])
    else:
        print(f"🔄 Updating existing ik_llama.cpp checkout: {IK_LLAMA_DIR}")
        run(["git", "fetch", "--depth", "1", "origin"], cwd=IK_LLAMA_DIR)
        run(["git", "pull", "--ff-only"], cwd=IK_LLAMA_DIR)

    os.makedirs(IK_LLAMA_BUILD_DIR, exist_ok=True)
    run(
        [
            "cmake",
            "-S",
            IK_LLAMA_DIR,
            "-B",
            IK_LLAMA_BUILD_DIR,
            "-DGGML_CUDA=OFF",
            "-DLLAMA_CURL=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    run(
        [
            "cmake",
            "--build",
            IK_LLAMA_BUILD_DIR,
            "--config",
            "Release",
            "--target",
            "llama-quantize",
            "-j",
            str(threads),
        ]
    )

    if not os.path.exists(quantize_bin):
        raise FileNotFoundError(f"ik_llama.cpp build did not produce {quantize_bin}")
    os.chmod(quantize_bin, 0o755)
    results_vol.commit()
    print(f"🎉 ik_llama.cpp CPU quantizer build on T4 is ready: {quantize_bin}")


@app.function(
    image=image,
    volumes={RESULTS_DIR: results_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=43200,
)
def run_qwen36_mtp_q3ks_quantization(
    hf_source_repo: str = DEFAULT_SOURCE_REPO,
    hf_repo: str = "",
    source_filename: str = "",
    output_filename: str = "",
    force_quantize: bool = False,
    threads: int = 16,
):
    import subprocess
    from pathlib import Path
    from urllib.parse import quote

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

    from huggingface_hub import HfApi

    def run(cmd, *, env=None, cwd=None):
        display_cmd = [
            "--header=Authorization: Bearer [REDACTED]"
            if part.startswith("--header=Authorization: Bearer ")
            else part
            for part in cmd
        ]
        print(f"Running command: {' '.join(display_cmd)}")
        result = subprocess.run(cmd, text=True, env=env, cwd=cwd)
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}: {' '.join(display_cmd)}"
            )
        return result

    def find_source_gguf(source_dir: str) -> str:
        if source_filename:
            source_path = os.path.join(source_dir, source_filename)
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Requested source GGUF not found: {source_path}")
            return source_path

        candidates = []
        for path in Path(source_dir).rglob("*.gguf"):
            if "q8_0" in path.name.lower():
                candidates.append(path)
        if not candidates:
            raise FileNotFoundError(f"No Q8_0 GGUF files found in {source_dir}")
        for path in sorted(candidates):
            if "-00001-of-" in path.name:
                return str(path)
        return str(max(candidates, key=lambda path: path.stat().st_size))

    quantize_bin = os.path.join(IK_LLAMA_BIN_DIR, "llama-quantize")
    if not os.path.exists(quantize_bin):
        raise FileNotFoundError(
            f"llama-quantize not found at {quantize_bin}. "
            "Run the default entrypoint first so build_ik_llama_cpp runs on T4."
        )
    os.chmod(quantize_bin, 0o755)

    api = HfApi()
    model_info = api.model_info(hf_source_repo, files_metadata=True)
    gguf_siblings = [
        sibling for sibling in model_info.siblings if sibling.rfilename.endswith(".gguf")
    ]
    if not gguf_siblings:
        raise FileNotFoundError(f"No GGUF files found in Hugging Face repo: {hf_source_repo}")
    print("⬇️ Downloading GGUF files only with hf_transfer enabled:")
    expected_bytes = 0
    for sibling in gguf_siblings:
        size = sibling.size or 0
        expected_bytes += size
        size_gib = size / (1024**3) if size else 0.0
        print(f"  - {sibling.rfilename} ({size_gib:.2f} GiB)")

    source_dir = os.path.join(RESULTS_DIR, "hf-snapshots", hf_source_repo.replace("/", "__"))
    os.makedirs(source_dir, exist_ok=True)

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    for sibling in gguf_siblings:
        target_path = os.path.join(source_dir, sibling.rfilename)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        expected_size = sibling.size or 0
        if expected_size and os.path.exists(target_path) and os.path.getsize(target_path) >= expected_size:
            print(f"✅ Reusing cached GGUF: {target_path}", flush=True)
            continue

        url = f"https://huggingface.co/{hf_source_repo}/resolve/main/{quote(sibling.rfilename)}"
        download_cmd = [
            "aria2c",
            "--continue=true",
            "--file-allocation=none",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=1M",
            "--max-tries=20",
            "--retry-wait=5",
            "--timeout=30",
            "--summary-interval=30",
            "--console-log-level=notice",
            "--dir",
            os.path.dirname(target_path),
            "--out",
            os.path.basename(target_path),
            url,
        ]
        if token:
            download_cmd.insert(1, f"--header=Authorization: Bearer {token}")
        print(f"⬇️ Multi-connection resumable GGUF download: {sibling.rfilename}", flush=True)
        run(download_cmd)

    downloaded = sum(
        path.stat().st_size for path in Path(source_dir).rglob("*.gguf") if path.is_file()
    )
    print(f"✅ GGUF download/cache size: {downloaded / (1024**3):.2f} GiB", flush=True)
    results_vol.commit()

    source_gguf = find_source_gguf(source_dir)
    print(f"✅ Source GGUF: {source_gguf}")

    if not output_filename:
        output_filename = default_output_name(hf_source_repo)
    output_path = os.path.join(source_dir, output_filename)

    custom_q = custom_q3ks_rules()
    if force_quantize or not os.path.exists(output_path):
        quantize_cmd = [
            quantize_bin,
            "--allow-requantize",
            "--custom-q",
            custom_q,
            source_gguf,
            output_path,
            "Q3_K_S",
            str(threads),
        ]
        print(f"📦 Starting plain Q3_K_S requantization for {output_filename}")
        run(quantize_cmd)
        results_vol.commit()
        print(f"✅ Quantization complete: {output_path}")
    else:
        print(f"✅ Reusing existing quantized artifact: {output_path}")

    if not hf_repo:
        username = api.whoami()["name"]
        hf_repo = f"{username}/Qwen3.6-27B-MTP-Q3_K_S-GGUF"

    print(f"🚀 Uploading Q3_K_S GGUF to Hugging Face repo: {hf_repo}")
    api.create_repo(repo_id=hf_repo, exist_ok=True)
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo=output_filename,
        repo_id=hf_repo,
    )

    readme_path = "/tmp/Qwen3.6-27B-MTP-Q3_K_S-README.md"
    readme = f"""\
    ---
    language:
    - en
    library_name: gguf
    tags:
    - qwen
    - gguf
    - ik_llama.cpp
    - mtp
    - q3_k_s
    base_model:
    - Qwen/Qwen3.6-27B
    - {hf_source_repo}
    pipeline_tag: text-generation
    ---

    # Qwen3.6-27B-MTP Q3_K_S GGUF

    This is a plain Q3_K_S GGUF quantization of Qwen3.6-27B that preserves the
    MTP (Multi-Token Prediction) tensors from `{hf_source_repo}`. It was
    requantized from the Q8_0 GGUF source with `ik_llama.cpp` using
    `--allow-requantize`.

    ## Requirements

    Use [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for inference.
    Pass MTP flags to use speculative decoding:

    ```bash
    llama-server -m {output_filename} -mtp --draft-max 1 --draft-p-min 0.0
    ```

    ## Quantization Recipe

    This quant was made from Q8_0 rather than directly from fp16, so a small
    amount of accuracy may have been lost before this Q3_K_S pass. No imatrix
    was generated or used for this normal K-quant.

    ```bash
    custom="{custom_q}"

    ./ik_llama.cpp/build/bin/llama-quantize \\
      --allow-requantize \\
      --custom-q "$custom" \\
      ./Qwen3.6-27B-MTP-Q8_0.gguf \\
      ./{output_filename} \\
      Q3_K_S {threads}
    ```

    Custom tensor overrides:

    - `blk.64.*.weight` MTP/nextn tensors: `Q8_0`
    - `ssm_alpha.weight`, `ssm_beta.weight`, and `ssm_out.weight`: `Q6_0`
    - token embeddings: `Q6_0`
    - output tensor: `Q8_0`
    - attention weights and FFN weights: `Q3_K`

    ## Source

    Source Q8_0 GGUF: [{hf_source_repo}](https://huggingface.co/{hf_source_repo})
    """
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(readme))
    api.upload_file(path_or_fileobj=readme_path, path_in_repo="README.md", repo_id=hf_repo)

    print(f"🎉 All done: https://huggingface.co/{hf_repo}/blob/main/{output_filename}")


@app.local_entrypoint()
def main(
    hf_source_repo: str = DEFAULT_SOURCE_REPO,
    hf_repo: str = "",
    source_filename: str = "",
    output_filename: str = "",
    force_rebuild_ik_llama: bool = False,
    force_quantize: bool = False,
    threads: int = 16,
    build_only: bool = False,
):
    build_ik_llama_cpp.remote(
        force_rebuild_ik_llama=force_rebuild_ik_llama,
        threads=threads,
    )
    if build_only:
        return

    run_qwen36_mtp_q3ks_quantization.remote(
        hf_source_repo=hf_source_repo,
        hf_repo=hf_repo,
        source_filename=source_filename,
        output_filename=output_filename,
        force_quantize=force_quantize,
        threads=threads,
    )
