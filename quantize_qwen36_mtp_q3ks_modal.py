import os
import textwrap

import modal


app = modal.App("qwen36-mtp-q3ks-quantizer")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "cmake", "build-essential", "libcurl4-openssl-dev")
    .pip_install(
        "cmake",
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
DEFAULT_CALIBRATION_URL = (
    "https://gist.githubusercontent.com/ubergarm/edfeb3ff9c6ec8b49e88cdf627b0711a/raw/"
    "ba5b01b6960a86874592f5913e283746ff734483/ubergarm-imatrix-calibration-corpus-v02.txt"
)
IMATRIX_FILENAME = "Qwen3.6-27B-MTP-imatrix.dat"


def default_output_name(source_repo: str) -> str:
    name = source_repo.rstrip("/").split("/")[-1]
    for suffix in ("-Q8_0-GGUF", "-Q8_0.gguf", "-Q8_0"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"{name}-Q3_K_S.gguf"


def custom_q3ks_rules() -> str:
    # Q3_K_S stores main attention/FFN tensors as Q3_K. These overrides keep the
    # Qwen hybrid SSM tensors aligned with the public Unsloth-style layout.
    rules = [
        r"blk\..*\.ssm_alpha\.weight=f32",
        r"blk\..*\.ssm_beta\.weight=f32",
        r"blk\..*\.ssm_conv1d\.weight=f32",
        r"blk\..*\.ssm_out\.weight=q4_k",
        r"token_embd\.weight=q3_k",
        r"output\.weight=q3_k",
        r"blk\..*\.attn_.*\.weight=q3_k",
        r"blk\..*\.ffn_.*\.weight=q3_k",
    ]
    return ",".join(rules)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RESULTS_DIR: results_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=43200,
)
def run_qwen36_mtp_q3ks_quantization(
    hf_source_repo: str = DEFAULT_SOURCE_REPO,
    hf_repo: str = "",
    source_filename: str = "",
    output_filename: str = "",
    calibration_url: str = DEFAULT_CALIBRATION_URL,
    generate_imatrix: bool = True,
    force_rebuild_ik_llama: bool = False,
    force_imatrix: bool = False,
    force_quantize: bool = False,
    threads: int = 16,
    upload_imatrix: bool = True,
):
    import shutil
    import subprocess
    import urllib.request
    from pathlib import Path

    from huggingface_hub import HfApi, snapshot_download

    def run(cmd, *, env=None, cwd=None):
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, text=True, env=env, cwd=cwd)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        return result

    def ensure_ik_llama_cpp():
        quantize_bin = os.path.join(IK_LLAMA_BIN_DIR, "llama-quantize")
        imatrix_bin = os.path.join(IK_LLAMA_BIN_DIR, "llama-imatrix")
        if (
            not force_rebuild_ik_llama
            and os.path.exists(quantize_bin)
            and os.path.exists(imatrix_bin)
        ):
            print(f"✅ Using existing ik_llama.cpp build at {IK_LLAMA_BIN_DIR}")
            os.chmod(quantize_bin, 0o755)
            os.chmod(imatrix_bin, 0o755)
            return quantize_bin, imatrix_bin

        if force_rebuild_ik_llama and os.path.exists(IK_LLAMA_DIR):
            print(f"🧹 Removing existing ik_llama.cpp checkout: {IK_LLAMA_DIR}")
            shutil.rmtree(IK_LLAMA_DIR)

        if not os.path.exists(IK_LLAMA_DIR):
            os.makedirs(os.path.dirname(IK_LLAMA_DIR), exist_ok=True)
            print(f"⬇️ Cloning ik_llama.cpp from {IK_LLAMA_REPO}")
            run(["git", "clone", IK_LLAMA_REPO, IK_LLAMA_DIR])
        else:
            print(f"🔄 Updating existing ik_llama.cpp checkout: {IK_LLAMA_DIR}")
            run(["git", "fetch", "--depth", "1", "origin", "master"], cwd=IK_LLAMA_DIR)
            run(["git", "checkout", "master"], cwd=IK_LLAMA_DIR)
            run(["git", "pull", "--ff-only", "origin", "master"], cwd=IK_LLAMA_DIR)

        os.makedirs(IK_LLAMA_BUILD_DIR, exist_ok=True)
        run(
            [
                "cmake",
                "-S",
                IK_LLAMA_DIR,
                "-B",
                IK_LLAMA_BUILD_DIR,
                "-DGGML_CUDA=ON",
                "-DLLAMA_CURL=ON",
                "-DCMAKE_BUILD_TYPE=Release",
            ]
        )
        run(["cmake", "--build", IK_LLAMA_BUILD_DIR, "--config", "Release", "-j", str(threads)])

        if not os.path.exists(quantize_bin) or not os.path.exists(imatrix_bin):
            raise FileNotFoundError(
                f"ik_llama.cpp build did not produce required binaries under {IK_LLAMA_BIN_DIR}"
            )
        os.chmod(quantize_bin, 0o755)
        os.chmod(imatrix_bin, 0o755)
        results_vol.commit()
        print(f"🎉 ik_llama.cpp build ready at {IK_LLAMA_BIN_DIR}")
        return quantize_bin, imatrix_bin

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
        return str(max(candidates, key=lambda p: p.stat().st_size))

    quantize_bin, imatrix_bin = ensure_ik_llama_cpp()

    source_dir = os.path.join(RESULTS_DIR, "hf-snapshots", hf_source_repo.replace("/", "__"))
    os.makedirs(source_dir, exist_ok=True)
    print(f"⬇️ Downloading Q8_0 MTP GGUF source from {hf_source_repo}")
    snapshot_download(
        repo_id=hf_source_repo,
        local_dir=source_dir,
        local_dir_use_symlinks=False,
        allow_patterns=["*.gguf", "*.md", "*.json"],
    )
    results_vol.commit()

    source_gguf = find_source_gguf(source_dir)
    print(f"✅ Source GGUF: {source_gguf}")

    if not output_filename:
        output_filename = default_output_name(hf_source_repo)
    output_path = os.path.join(source_dir, output_filename)
    imatrix_path = os.path.join(source_dir, IMATRIX_FILENAME)

    if generate_imatrix:
        if force_imatrix or not os.path.exists(imatrix_path):
            calibration_path = "/tmp/ubergarm-imatrix-calibration-corpus-v02.txt"
            print(f"⬇️ Downloading calibration corpus from {calibration_url}")
            urllib.request.urlretrieve(calibration_url, calibration_path)

            imatrix_env = os.environ.copy()
            imatrix_env["GGML_CUDA_NO_PINNED"] = "1"
            imatrix_cmd = [
                imatrix_bin,
                "-m",
                source_gguf,
                "-f",
                calibration_path,
                "-o",
                imatrix_path,
                "--ctx-size",
                "512",
                "-t",
                str(threads),
                "--fit",
            ]
            print("𓌳 Generating imatrix from Q8_0 source with GGML_CUDA_NO_PINNED=1")
            run(imatrix_cmd, env=imatrix_env)
            results_vol.commit()
            print(f"✅ Imatrix ready: {imatrix_path}")
        else:
            print(f"✅ Reusing existing imatrix: {imatrix_path}")
    elif not os.path.exists(imatrix_path):
        raise FileNotFoundError(
            f"Imatrix not found at {imatrix_path}; rerun with generate_imatrix=True."
        )

    custom_q = custom_q3ks_rules()
    if force_quantize or not os.path.exists(output_path):
        quantize_cmd = [
            quantize_bin,
            "--allow-requantize",
            "--imatrix",
            imatrix_path,
            "--custom-q",
            custom_q,
            source_gguf,
            output_path,
            "Q3_K_S",
            str(threads),
        ]
        print(f"📦 Starting Q3_K_S requantization for {output_filename}")
        run(quantize_cmd)
        results_vol.commit()
        print(f"✅ Quantization complete: {output_path}")
    else:
        print(f"✅ Reusing existing quantized artifact: {output_path}")

    api = HfApi()
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

    if upload_imatrix:
        api.upload_file(
            path_or_fileobj=imatrix_path,
            path_in_repo=os.path.basename(imatrix_path),
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

    This is a Q3_K_S GGUF quantization of Qwen3.6-27B that preserves the MTP
    (Multi-Token Prediction) tensors from `{hf_source_repo}`. It was requantized
    from the Q8_0 GGUF source with `ik_llama.cpp` using `--allow-requantize`.

    ## Requirements

    Use [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for inference.
    Pass MTP flags to use speculative decoding:

    ```bash
    llama-server -m {output_filename} -mtp --draft-max 1 --draft-p-min 0.0
    ```

    ## Quantization Recipe

    The quantization was generated from the Q8_0 source rather than directly from
    fp16, so a small amount of accuracy may have been lost before this Q3_K_S pass.

    ```bash
    GGML_CUDA_NO_PINNED=1 ./ik_llama.cpp/build/bin/llama-imatrix \\
      -m ./Qwen3.6-27B-MTP-Q8_0.gguf \\
      -f ./ubergarm-imatrix-calibration-corpus-v02.txt \\
      -o ./{IMATRIX_FILENAME} \\
      --ctx-size 512 \\
      -t {threads} \\
      --fit

    custom="{custom_q}"

    ./ik_llama.cpp/build/bin/llama-quantize \\
      --allow-requantize \\
      --imatrix ./{IMATRIX_FILENAME} \\
      --custom-q "$custom" \\
      ./Qwen3.6-27B-MTP-Q8_0.gguf \\
      ./{output_filename} \\
      Q3_K_S {threads}
    ```

    Custom tensor overrides:

    - `ssm_alpha.weight`, `ssm_beta.weight`, and `ssm_conv1d.weight`: `F32`
    - `ssm_out.weight`: `Q4_K`
    - token embeddings, output tensor, attention weights, and FFN weights: `Q3_K`

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
    calibration_url: str = DEFAULT_CALIBRATION_URL,
    generate_imatrix: bool = True,
    force_rebuild_ik_llama: bool = False,
    force_imatrix: bool = False,
    force_quantize: bool = False,
    threads: int = 16,
    upload_imatrix: bool = True,
):
    run_qwen36_mtp_q3ks_quantization.remote(
        hf_source_repo=hf_source_repo,
        hf_repo=hf_repo,
        source_filename=source_filename,
        output_filename=output_filename,
        calibration_url=calibration_url,
        generate_imatrix=generate_imatrix,
        force_rebuild_ik_llama=force_rebuild_ik_llama,
        force_imatrix=force_imatrix,
        force_quantize=force_quantize,
        threads=threads,
        upload_imatrix=upload_imatrix,
    )
