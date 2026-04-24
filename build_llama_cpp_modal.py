import datetime
import json
import os
import pathlib
import shutil
import subprocess
import zipfile

import modal


app = modal.App("buun-llama-cpp-build-upload-t4")

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
    llama_repo_url: str = "https://github.com/spiritbuun/buun-llama-cpp.git",
    artifact_prefix: str = "buun-llama-cpp-dflash-cuda-t4",
    cuda_architectures: str = "75",
):
    from huggingface_hub import HfApi

    if not hf_repo:
        raise ValueError("hf_repo is required (example: username/llama-cpp-builds)")

    api = HfApi()
    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)

    # Fail fast before spending build time if the secret token can't write to the repo.
    probe_path = f"builds/.write-check-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.txt"
    probe_file = pathlib.Path("/tmp/hf-write-check.txt")
    probe_file.write_text("write-check\n", encoding="utf-8")
    print(f"Running Hugging Face write-access preflight check for {hf_repo} ...")
    try:
        api.upload_file(
            path_or_fileobj=str(probe_file),
            path_in_repo=probe_path,
            repo_id=hf_repo,
            repo_type="model",
        )
        api.delete_file(
            path_in_repo=probe_path,
            repo_id=hf_repo,
            repo_type="model",
            commit_message="cleanup write-access preflight probe",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Write-access preflight failed for repo '{hf_repo}'. "
            "Ensure the Modal secret token has write permission to this repository."
        ) from exc
    print("Write-access preflight check passed.")

    llama_src = pathlib.Path("/tmp/llama.cpp")
    build_dir = llama_src / "build"
    package_root = pathlib.Path("/tmp/llama-cpp-build")
    binaries_dir = package_root / "bin"

    print(f"Cloning llama.cpp fork from {llama_repo_url} at ref '{llama_ref}'...")
    subprocess.run(
        ["git", "clone", llama_repo_url, str(llama_src)],
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
            "-DGGML_NATIVE=ON",
            "-DGGML_CUDA_FA=ON",
            "-DGGML_CUDA_FA_ALL_QUANTS=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}",
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
    required_bins = [
        "llama-cli",
        "llama-quantize",
        "llama-server",
        "llama-speculative-simple",
    ]
    missing_bins = [name for name in required_bins if not (bin_root / name).exists()]
    if missing_bins:
        raise FileNotFoundError(
            f"Missing expected binaries after build: {', '.join(missing_bins)}"
        )

    print("Copying selected binaries into package...")
    copied_files: list[pathlib.Path] = []
    for binary_name in required_bins:
        src = bin_root / binary_name
        dst = binaries_dir / binary_name
        print(f"  - copy {binary_name}")
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
        copied_files.append(dst)

    convert_script = package_root / "convert_hf_to_gguf.py"
    license_file = package_root / "LICENSE"
    shutil.copy2(llama_src / "convert_hf_to_gguf.py", convert_script)
    shutil.copy2(llama_src / "LICENSE", license_file)
    copied_files.extend([convert_script, license_file])

    metadata = {
        "llama_cpp_repo": llama_repo_url,
        "llama_cpp_ref": llama_ref,
        "llama_cpp_commit": short_sha,
        "built_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "gpu_type": "T4",
        "cmake_flags": [
            "-DGGML_CUDA=ON",
            "-DGGML_NATIVE=ON",
            "-DGGML_CUDA_FA=ON",
            "-DGGML_CUDA_FA_ALL_QUANTS=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}",
            "-DBUILD_SHARED_LIBS=OFF",
        ],
    }
    with open(package_root / "build-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    artifact_name = f"{artifact_prefix}-{timestamp}-{short_sha}.zip"
    artifact_zip = pathlib.Path(f"/tmp/{artifact_name}")
    print(f"Creating zip package at {artifact_zip} ...")
    with zipfile.ZipFile(artifact_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for file_path in copied_files:
            arcname = file_path.relative_to(package_root)
            zf.write(file_path, arcname=arcname)
        zf.write(
            package_root / "build-metadata.json",
            arcname="build-metadata.json",
        )
    print(f"Zip creation complete ({artifact_zip.stat().st_size / (1024 ** 2):.2f} MiB).")

    path_in_repo = f"builds/{artifact_name}"
    print(f"Uploading {artifact_name} to https://huggingface.co/{hf_repo} ...")
    print(f"Starting upload of zip artifact to {path_in_repo} ...")
    api.upload_file(
        path_or_fileobj=str(artifact_zip),
        path_in_repo=path_in_repo,
        repo_id=hf_repo,
        repo_type="model",
    )
    print("Zip upload complete.")

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
    llama_repo_url: str = "https://github.com/spiritbuun/buun-llama-cpp.git",
    artifact_prefix: str = "buun-llama-cpp-dflash-cuda-t4",
    cuda_architectures: str = "75",
):
    if not hf_repo:
        raise ValueError(
            "Usage: modal run build_llama_cpp_modal.py --hf-repo username/llama-cpp-builds"
        )

    result = build_and_upload.remote(
        hf_repo=hf_repo,
        llama_ref=llama_ref,
        llama_repo_url=llama_repo_url,
        artifact_prefix=artifact_prefix,
        cuda_architectures=cuda_architectures,
    )
    print("Build + upload completed.")
    print(json.dumps(result, indent=2))
