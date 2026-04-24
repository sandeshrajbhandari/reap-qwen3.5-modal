import json
import os
import pathlib
import shutil
import subprocess
import zipfile

import modal


app = modal.App("buun-llama-cpp-dflash-test-t4")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("huggingface-hub")
)


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
)
def download_and_test_dflash(
    build_repo: str = "sandeshrajx/llama_cpp_colab_builds",
    artifact_path: str = "builds/buun-llama-cpp-dflash-cuda-t4-20260424-032011-22464d084.zip",
    target_model_repo: str = "unsloth/Qwen3.6-27B-GGUF",
    target_model_file: str = "Qwen3.6-27B-UD-IQ3_XXS.gguf",
    draft_model_repo: str = "spiritbuun/Qwen3.5-27B-DFlash-GGUF",
    draft_model_file: str = "dflash-draft-q4_k_m.gguf",
    prompt: str = "Write a concise Python mergesort implementation.",
):
    from huggingface_hub import hf_hub_download

    print(f"Downloading build artifact from {build_repo}/{artifact_path} ...")
    artifact_zip = hf_hub_download(
        repo_id=build_repo,
        filename=artifact_path,
        repo_type="model",
    )

    extract_dir = pathlib.Path("/tmp/llama-cpp-build")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting build artifact...")
    with zipfile.ZipFile(artifact_zip, "r") as zf:
        zf.extractall(extract_dir)

    bin_dir = extract_dir / "bin"
    spec_bin = bin_dir / "llama-speculative-simple"
    server_bin = bin_dir / "llama-server"
    if not spec_bin.exists():
        raise FileNotFoundError(f"Missing binary: {spec_bin}")
    if not server_bin.exists():
        raise FileNotFoundError(f"Missing binary: {server_bin}")

    os.chmod(spec_bin, 0o755)
    os.chmod(server_bin, 0o755)

    print(
        f"Downloading target model from {target_model_repo}/{target_model_file} ..."
    )
    target_model_path = hf_hub_download(
        repo_id=target_model_repo,
        filename=target_model_file,
        repo_type="model",
    )

    print(f"Downloading draft model from {draft_model_repo}/{draft_model_file} ...")
    draft_model_path = hf_hub_download(
        repo_id=draft_model_repo,
        filename=draft_model_file,
        repo_type="model",
    )

    version_cmd = [str(spec_bin), "--version"]
    print(f"Running: {' '.join(version_cmd)}")
    version_res = subprocess.run(version_cmd, capture_output=True, text=True, check=True)

    test_cmd = [
        str(spec_bin),
        "-m",
        target_model_path,
        "-md",
        draft_model_path,
        "--spec-type",
        "dflash",
        "-ngl",
        "99",
        "-ngld",
        "99",
        "-c",
        "2048",
        "--draft-max",
        "16",
        "--draft-min",
        "1",
        "-n",
        "96",
        "-p",
        prompt,
    ]
    print(f"Running DFlash smoke test: {' '.join(test_cmd)}")
    result = subprocess.run(test_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "DFlash smoke test failed.\n"
            f"Exit code: {result.returncode}\n"
            f"STDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )

    stdout_tail = result.stdout[-4000:]
    stderr_tail = result.stderr[-2000:]
    print("DFlash smoke test completed successfully.")
    print(stdout_tail)

    return {
        "build_repo": build_repo,
        "artifact_path": artifact_path,
        "target_model_repo": target_model_repo,
        "target_model_file": target_model_file,
        "draft_model_repo": draft_model_repo,
        "draft_model_file": draft_model_file,
        "spec_binary_version_tail": version_res.stdout[-1000:],
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


@app.local_entrypoint()
def main(
    build_repo: str = "sandeshrajx/llama_cpp_colab_builds",
    artifact_path: str = "builds/buun-llama-cpp-dflash-cuda-t4-20260424-032011-22464d084.zip",
    target_model_repo: str = "unsloth/Qwen3.6-27B-GGUF",
    target_model_file: str = "Qwen3.6-27B-UD-IQ3_XXS.gguf",
    draft_model_repo: str = "spiritbuun/Qwen3.5-27B-DFlash-GGUF",
    draft_model_file: str = "dflash-draft-q4_k_m.gguf",
    prompt: str = "Write a concise Python mergesort implementation.",
):
    output = download_and_test_dflash.remote(
        build_repo=build_repo,
        artifact_path=artifact_path,
        target_model_repo=target_model_repo,
        target_model_file=target_model_file,
        draft_model_repo=draft_model_repo,
        draft_model_file=draft_model_file,
        prompt=prompt,
    )
    print("DFlash build test finished.")
    print(json.dumps(output, indent=2))
