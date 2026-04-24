import json
import os
import pathlib
import shutil
import subprocess
import zipfile

import modal


app = modal.App("llama-cpp-build-test-t4")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("unzip")
    .pip_install("huggingface-hub")
)


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def download_and_test(
    hf_repo: str,
    artifact_path: str = "",
    demo_model_repo: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    demo_model_file: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
):
    from huggingface_hub import hf_hub_download

    if not hf_repo:
        raise ValueError("hf_repo is required (example: username/llama-cpp-builds)")

    if not artifact_path:
        latest_file = hf_hub_download(
            repo_id=hf_repo,
            filename="builds/latest.txt",
            repo_type="model",
        )
        with open(latest_file, "r", encoding="utf-8") as f:
            artifact_path = f.readline().strip()
        if not artifact_path:
            raise ValueError(
                "Could not infer artifact path from builds/latest.txt. "
                "Pass --artifact-path explicitly."
            )

    print(f"Downloading build artifact: {artifact_path}")
    artifact_zip = hf_hub_download(
        repo_id=hf_repo,
        filename=artifact_path,
        repo_type="model",
    )

    extract_dir = pathlib.Path("/tmp/llama-cpp-build")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(artifact_zip, "r") as zf:
        zf.extractall(extract_dir)

    llama_cli = extract_dir / "bin" / "llama-cli"
    if not llama_cli.exists():
        raise FileNotFoundError(f"Expected executable not found: {llama_cli}")
    os.chmod(llama_cli, 0o755)

    print(f"Downloading demo model: {demo_model_repo}/{demo_model_file}")
    model_path = hf_hub_download(
        repo_id=demo_model_repo,
        filename=demo_model_file,
        repo_type="model",
    )

    cmd = [
        str(llama_cli),
        "-m",
        model_path,
        "-ngl",
        "99",
        "-n",
        "64",
        "-p",
        "Write one sentence proving this llama.cpp build works.",
    ]
    print(f"Running smoke test: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "llama-cli smoke test failed.\n"
            f"Exit code: {result.returncode}\n"
            f"STDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )

    stdout_tail = result.stdout[-3000:]
    print("Smoke test completed successfully.")
    print(stdout_tail)

    return {
        "artifact_path": artifact_path,
        "demo_model_repo": demo_model_repo,
        "demo_model_file": demo_model_file,
        "stdout_tail": stdout_tail,
    }


@app.local_entrypoint()
def main(
    hf_repo: str = "",
    artifact_path: str = "",
    demo_model_repo: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    demo_model_file: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
):
    if not hf_repo:
        raise ValueError(
            "Usage: modal run test_llama_cpp_build_modal.py --hf-repo username/llama-cpp-builds"
        )

    output = download_and_test.remote(
        hf_repo=hf_repo,
        artifact_path=artifact_path,
        demo_model_repo=demo_model_repo,
        demo_model_file=demo_model_file,
    )
    print("Build test completed.")
    print(json.dumps(output, indent=2))
