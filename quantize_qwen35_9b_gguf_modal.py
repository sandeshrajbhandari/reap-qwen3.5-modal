import os
import textwrap

import modal


app = modal.App("qwen35-9b-gguf-quantizer")

image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.12")
    .apt_install("aria2", "cmake", "git", "build-essential", "libcurl4-openssl-dev")
    .pip_install(
        "cmake",
        "gguf",
        "hf_transfer",
        "huggingface-hub",
        "numpy",
        "protobuf",
        "safetensors",
        "sentencepiece",
        "transformers",
    )
    .run_commands("pip install --index-url https://download.pytorch.org/whl/cpu torch")
)

results_vol = modal.Volume.from_name("reap-results")
RESULTS_DIR = "/results"

IK_LLAMA_REPO = "https://github.com/ikawrakow/ik_llama.cpp.git"
IK_LLAMA_DIR = os.path.join(RESULTS_DIR, "tools/ik_llama.cpp")
IK_LLAMA_BUILD_DIR = os.path.join(IK_LLAMA_DIR, "build")
IK_LLAMA_BIN_DIR = os.path.join(IK_LLAMA_BUILD_DIR, "bin")
CONVERT_SCRIPT = os.path.join(IK_LLAMA_DIR, "convert_hf_to_gguf.py")

DEFAULT_SOURCE_REPO = "Qwen/Qwen3.5-9B"
DEFAULT_MODEL_DIR = os.path.join(RESULTS_DIR, "hf-snapshots", DEFAULT_SOURCE_REPO.replace("/", "__"))


def build_custom_q4km_rules(preserve_mtp: bool = True, mtp_block_index: int = 32) -> str:
    # Qwen3.5-9B has 32 normal blocks. If the checkpoint/converter exposes an MTP
    # next-token block after them, keep those tensors high precision for better
    # speculative-decoding acceptance.
    rules = []
    if preserve_mtp:
        rules.append(rf"blk\.{mtp_block_index}\..*\.weight=q8_0")
    rules.extend(
        [
            r"token_embd\.weight=q4_k",
            r"blk\..*\.attn_gate\.weight=q4_k",
            r"blk\..*\.attn_qkv\.weight=q5_k",
            r"blk\..*\.ffn_down\.weight=q6_k",
            r"blk\..*\.ffn_gate\.weight=q4_k",
            r"blk\..*\.ffn_up\.weight=q4_k",
            r"blk\..*\.ssm_alpha\.weight=q8_0",
            r"blk\..*\.ssm_beta\.weight=q8_0",
            r"blk\..*\.ssm_conv1d\.weight=f32",
            r"blk\..*\.ssm_out\.weight=q5_k",
            r"output\.weight=q6_k",
        ]
    )
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
        print(f"Running command: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, text=True, cwd=cwd)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        return result

    quantize_bin = os.path.join(IK_LLAMA_BIN_DIR, "llama-quantize")
    if (
        not force_rebuild_ik_llama
        and os.path.exists(quantize_bin)
        and os.path.exists(CONVERT_SCRIPT)
    ):
        os.chmod(quantize_bin, 0o755)
        print(f"✅ Using existing ik_llama.cpp checkout at {IK_LLAMA_DIR}", flush=True)
        return

    if force_rebuild_ik_llama and os.path.exists(IK_LLAMA_DIR):
        print(f"🧹 Removing existing ik_llama.cpp checkout: {IK_LLAMA_DIR}", flush=True)
        shutil.rmtree(IK_LLAMA_DIR)
    elif os.path.exists(IK_LLAMA_BUILD_DIR) and not os.path.exists(quantize_bin):
        print(f"🧹 Removing interrupted/stale build dir: {IK_LLAMA_BUILD_DIR}", flush=True)
        shutil.rmtree(IK_LLAMA_BUILD_DIR)

    if not os.path.exists(IK_LLAMA_DIR):
        os.makedirs(os.path.dirname(IK_LLAMA_DIR), exist_ok=True)
        print(f"⬇️ Cloning ik_llama.cpp from {IK_LLAMA_REPO}", flush=True)
        run(["git", "clone", IK_LLAMA_REPO, IK_LLAMA_DIR])
    else:
        print(f"🔄 Updating existing ik_llama.cpp checkout: {IK_LLAMA_DIR}", flush=True)
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
    if not os.path.exists(CONVERT_SCRIPT):
        raise FileNotFoundError(f"ik_llama.cpp converter not found at {CONVERT_SCRIPT}")
    os.chmod(quantize_bin, 0o755)
    results_vol.commit()
    print(f"🎉 ik_llama.cpp CPU tools are ready under {IK_LLAMA_DIR}", flush=True)


@app.function(
    image=image,
    volumes={RESULTS_DIR: results_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=43200,
)
def convert_and_quantize_qwen35_9b(
    hf_source_repo: str = DEFAULT_SOURCE_REPO,
    hf_repo: str = "",
    q8_filename: str = "",
    q4_filename: str = "",
    force_download: bool = False,
    force_convert: bool = False,
    force_quantize: bool = False,
    upload_q8: bool = True,
    upload_q4: bool = True,
    preserve_mtp: bool = True,
    mtp_block_index: int = 32,
    threads: int = 16,
):
    import subprocess
    from pathlib import Path

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

    from huggingface_hub import HfApi, snapshot_download

    def run(cmd, *, env=None, cwd=None):
        print(f"Running command: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, text=True, env=env, cwd=cwd)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        return result

    quantize_bin = os.path.join(IK_LLAMA_BIN_DIR, "llama-quantize")
    if not os.path.exists(CONVERT_SCRIPT):
        raise FileNotFoundError(
            f"Converter not found at {CONVERT_SCRIPT}. Run the default entrypoint first."
        )
    if not os.path.exists(quantize_bin):
        raise FileNotFoundError(
            f"llama-quantize not found at {quantize_bin}. Run the default entrypoint first."
        )
    os.chmod(quantize_bin, 0o755)

    model_dir = os.path.join(RESULTS_DIR, "hf-snapshots", hf_source_repo.replace("/", "__"))
    if force_download and os.path.exists(model_dir):
        import shutil

        print(f"🧹 Removing existing snapshot: {model_dir}", flush=True)
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    print(f"⬇️ Downloading HF checkpoint for {hf_source_repo}", flush=True)
    snapshot_download(
        repo_id=hf_source_repo,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.json",
            "*.model",
            "*.py",
            "*.safetensors",
            "*.tiktoken",
            "merges.txt",
            "tokenizer*",
            "vocab*",
        ],
        max_workers=8,
    )
    results_vol.commit()

    model_name = hf_source_repo.rstrip("/").split("/")[-1]
    if not q8_filename:
        q8_filename = f"{model_name}-Q8_0.gguf"
    if not q4_filename:
        q4_filename = f"{model_name}-Q4_K_M.gguf"
    q8_path = os.path.join(model_dir, q8_filename)
    q4_path = os.path.join(model_dir, q4_filename)

    if force_convert or not os.path.exists(q8_path):
        convert_cmd = [
            "python3",
            CONVERT_SCRIPT,
            model_dir,
            "--outfile",
            q8_path,
            "--outtype",
            "q8_0",
        ]
        print(f"📦 Converting {hf_source_repo} to Q8_0 GGUF with ik_llama.cpp", flush=True)
        run(convert_cmd)
        results_vol.commit()
        print(f"✅ Q8_0 GGUF ready: {q8_path}", flush=True)
    else:
        print(f"✅ Reusing existing Q8_0 GGUF: {q8_path}", flush=True)

    custom_q = build_custom_q4km_rules(
        preserve_mtp=preserve_mtp,
        mtp_block_index=mtp_block_index,
    )
    if force_quantize or not os.path.exists(q4_path):
        quantize_cmd = [
            quantize_bin,
            "--allow-requantize",
            "--custom-q",
            custom_q,
            q8_path,
            q4_path,
            "Q4_K_M",
            str(threads),
        ]
        print(f"📦 Quantizing Q8_0 GGUF to Q4_K_M with custom Qwen3.5-9B recipe", flush=True)
        run(quantize_cmd)
        results_vol.commit()
        print(f"✅ Q4_K_M GGUF ready: {q4_path}", flush=True)
    else:
        print(f"✅ Reusing existing Q4_K_M GGUF: {q4_path}", flush=True)

    if not hf_repo:
        print("ℹ️ hf_repo not provided; skipping Hugging Face upload.", flush=True)
        return

    api = HfApi()
    api.create_repo(repo_id=hf_repo, exist_ok=True)
    if upload_q8:
        print(f"🚀 Uploading {q8_filename} to {hf_repo}", flush=True)
        api.upload_file(path_or_fileobj=q8_path, path_in_repo=q8_filename, repo_id=hf_repo)
    if upload_q4:
        print(f"🚀 Uploading {q4_filename} to {hf_repo}", flush=True)
        api.upload_file(path_or_fileobj=q4_path, path_in_repo=q4_filename, repo_id=hf_repo)

    readme_path = "/tmp/Qwen3.5-9B-GGUF-README.md"
    readme = f"""\
    ---
    language:
    - en
    library_name: gguf
    tags:
    - qwen
    - gguf
    - ik_llama.cpp
    - q4_k_m
    - q8_0
    base_model:
    - {hf_source_repo}
    pipeline_tag: text-generation
    ---

    # {model_name} GGUF

    This repository contains GGUF conversions of `{hf_source_repo}` generated with
    [`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp).

    ## Files

    - `{q8_filename}`: Q8_0 GGUF converted directly from the Hugging Face checkpoint
      using `convert_hf_to_gguf.py`.
    - `{q4_filename}`: Q4_K_M GGUF requantized from the Q8_0 GGUF.

    ## Q4_K_M tensor recipe

    ```text
    {custom_q.replace(",", chr(10))}
    ```

    The Q4_K_M quantization command used:

    ```bash
    ./ik_llama.cpp/build/bin/llama-quantize \\
      --allow-requantize \\
      --custom-q "{custom_q}" \\
      ./{q8_filename} \\
      ./{q4_filename} \\
      Q4_K_M {threads}
    ```

    If MTP/nextn tensors are present after the 32 normal blocks, this recipe keeps
    `blk.{mtp_block_index}.*.weight` in Q8_0 for speculative-decoding acceptance.
    """
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(readme))
    api.upload_file(path_or_fileobj=readme_path, path_in_repo="README.md", repo_id=hf_repo)
    print(f"🎉 All done: https://huggingface.co/{hf_repo}", flush=True)


@app.local_entrypoint()
def main(
    hf_source_repo: str = DEFAULT_SOURCE_REPO,
    hf_repo: str = "",
    q8_filename: str = "",
    q4_filename: str = "",
    force_rebuild_ik_llama: bool = False,
    force_download: bool = False,
    force_convert: bool = False,
    force_quantize: bool = False,
    upload_q8: bool = True,
    upload_q4: bool = True,
    preserve_mtp: bool = True,
    mtp_block_index: int = 32,
    threads: int = 16,
    build_only: bool = False,
):
    build_ik_llama_cpp.remote(
        force_rebuild_ik_llama=force_rebuild_ik_llama,
        threads=threads,
    )
    if build_only:
        return

    convert_and_quantize_qwen35_9b.remote(
        hf_source_repo=hf_source_repo,
        hf_repo=hf_repo,
        q8_filename=q8_filename,
        q4_filename=q4_filename,
        force_download=force_download,
        force_convert=force_convert,
        force_quantize=force_quantize,
        upload_q8=upload_q8,
        upload_q4=upload_q4,
        preserve_mtp=preserve_mtp,
        mtp_block_index=mtp_block_index,
        threads=threads,
    )
