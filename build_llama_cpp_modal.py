import datetime
import json
import os
import pathlib
import shutil
import subprocess

import modal


app = modal.App("llama-cpp-build-upload-t4")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "cmake", "build-essential", "zip")
    .pip_install("huggingface-hub")
)


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
)
def build_and_upload(
    hf_repo: str,
    llama_ref: str = "master",
    artifact_prefix: str = "llama.cpp-cuda-t4",
):
    from huggingface_hub import HfApi

    if not hf_repo:
        raise ValueError("hf_repo is required (example: username/llama-cpp-builds)")

    llama_src = pathlib.Path("/tmp/llama.cpp")
    build_dir = llama_src / "build"
    package_root = pathlib.Path("/tmp/llama-cpp-build")
    binaries_dir = package_root / "bin"

    print(f"Cloning llama.cpp at ref '{llama_ref}'...")
    subprocess.run(
        ["git", "clone", "https://github.com/ggml-org/llama.cpp.git", str(llama_src)],
        check=True,
    )
    subprocess.run(["git", "checkout", llama_ref], check=True, cwd=llama_src)
    short_sha = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=llama_src)
        .decode()
        .strip()
    )
    print(f"Building commit {short_sha}")

    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(llama_src),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON",
            "-DLLAMA_CURL=ON",
            "-DBUILD_SHARED_LIBS=OFF",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release", "-j", "16"],
        check=True,
    )

    if package_root.exists():
        shutil.rmtree(package_root)
    binaries_dir.mkdir(parents=True, exist_ok=True)

    bin_root = build_dir / "bin"
    required_bins = ["llama-cli", "llama-quantize", "llama-server"]
    missing_bins = [name for name in required_bins if not (bin_root / name).exists()]
    if missing_bins:
        raise FileNotFoundError(
            f"Missing expected binaries after build: {', '.join(missing_bins)}"
        )

    print("Copying binaries into package...")
    for path in bin_root.iterdir():
        if path.is_file():
            shutil.copy2(path, binaries_dir / path.name)

    shutil.copy2(
        llama_src / "convert_hf_to_gguf.py",
        package_root / "convert_hf_to_gguf.py",
    )
    shutil.copy2(llama_src / "LICENSE", package_root / "LICENSE")

    metadata = {
        "llama_cpp_ref": llama_ref,
        "llama_cpp_commit": short_sha,
        "built_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "gpu_type": "T4",
    }
    with open(package_root / "build-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    artifact_name = f"{artifact_prefix}-{timestamp}-{short_sha}.zip"
    artifact_base = f"/tmp/{artifact_name.removesuffix('.zip')}"
    artifact_zip = f"{artifact_base}.zip"
    shutil.make_archive(artifact_base, "zip", root_dir=package_root)

    api = HfApi()
    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)

    path_in_repo = f"builds/{artifact_name}"
    print(f"Uploading {artifact_name} to https://huggingface.co/{hf_repo} ...")
    api.upload_file(
        path_or_fileobj=artifact_zip,
        path_in_repo=path_in_repo,
        repo_id=hf_repo,
        repo_type="model",
    )

    latest_file = "/tmp/latest.txt"
    with open(latest_file, "w", encoding="utf-8") as f:
        f.write(f"{path_in_repo}\n")
        f.write(json.dumps(metadata) + "\n")

    api.upload_file(
        path_or_fileobj=latest_file,
        path_in_repo="builds/latest.txt",
        repo_id=hf_repo,
        repo_type="model",
    )

    result = {
        "artifact_name": artifact_name,
        "path_in_repo": path_in_repo,
        "repo_url": f"https://huggingface.co/{hf_repo}",
        "commit": short_sha,
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    hf_repo: str = "",
    llama_ref: str = "master",
    artifact_prefix: str = "llama.cpp-cuda-t4",
):
    if not hf_repo:
        raise ValueError(
            "Usage: modal run build_llama_cpp_modal.py --hf-repo username/llama-cpp-builds"
        )

    result = build_and_upload.remote(
        hf_repo=hf_repo,
        llama_ref=llama_ref,
        artifact_prefix=artifact_prefix,
    )
    print("Build + upload completed.")
    print(json.dumps(result, indent=2))
