import datetime
import json
import os
import pathlib
import shutil
import subprocess
import zipfile

import modal


app = modal.App("ik-llama-cpp-cuda-build-upload")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "build-essential",
        "cmake",
        "curl",
        "git",
        "libcurl4-openssl-dev",
        "libgomp1",
        "zip",
    )
    .pip_install("cmake", "huggingface-hub")
)


CORE_BINARIES = [
    "llama-cli",
    "llama-server",
    "llama-quantize",
    "llama-imatrix",
    "llama-bench",
]

CONVERT_SCRIPTS = [
    "convert_hf_to_gguf.py",
    "convert_hf_to_gguf_update.py",
    "convert_imatrix_gguf_to_dat.py",
    "convert_lora_to_gguf.py",
]


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=14400,
)
def build_smoke_test_and_upload(
    hf_repo: str,
    llama_ref: str = "main",
    llama_repo_url: str = "https://github.com/ikawrakow/ik_llama.cpp.git",
    artifact_prefix: str = "ik-llama-cpp-cuda-default-colab",
    run_smoke_test: bool = True,
    demo_model_repo: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    demo_model_file: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
):
    from huggingface_hub import HfApi, hf_hub_download

    if not hf_repo:
        raise ValueError("hf_repo is required (example: username/llama-cpp-builds)")

    api = HfApi()
    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)

    # Fail fast before spending build time if the token cannot write to the repo.
    probe_path = (
        "builds/.write-check-"
        f"{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.txt"
    )
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

    source_dir = pathlib.Path("/tmp/ik_llama.cpp")
    build_dir = source_dir / "build"
    package_root = pathlib.Path("/tmp/ik-llama-cpp-build")
    binaries_dir = package_root / "bin"

    print(f"Cloning {llama_repo_url} at ref '{llama_ref}'...")
    if source_dir.exists():
        shutil.rmtree(source_dir)
    subprocess.run(["git", "clone", llama_repo_url, str(source_dir)], check=True)
    subprocess.run(["git", "checkout", llama_ref], check=True, cwd=source_dir)

    short_sha = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=source_dir)
        .decode()
        .strip()
    )
    full_sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_dir)
        .decode()
        .strip()
    )
    print(f"Building ik_llama.cpp commit {short_sha}")
    cmake_version = subprocess.check_output(["cmake", "--version"]).decode().strip()
    print(cmake_version)

    configure_cmd = [
        "cmake",
        "-B",
        "build",
        "-DGGML_NATIVE=ON",
        "-DGGML_CUDA=ON",
    ]
    build_cmd = [
        "cmake",
        "--build",
        "build",
        "--config",
        "Release",
        "-j",
        str(os.cpu_count() or 1),
    ]
    subprocess.run(configure_cmd, cwd=source_dir, check=True)
    subprocess.run(build_cmd, cwd=source_dir, check=True)

    bin_root = build_dir / "bin"
    missing_bins = [name for name in CORE_BINARIES if not (bin_root / name).exists()]
    if missing_bins:
        raise FileNotFoundError(
            f"Missing expected binaries after build: {', '.join(missing_bins)}"
        )

    if package_root.exists():
        shutil.rmtree(package_root)
    binaries_dir.mkdir(parents=True, exist_ok=True)

    print("Copying selected binaries into package...")
    copied_files: list[pathlib.Path] = []
    for binary_name in CORE_BINARIES:
        src = bin_root / binary_name
        dst = binaries_dir / binary_name
        print(f"  - {binary_name}")
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
        copied_files.append(dst)

    shared_libraries: list[str] = []
    for src in sorted(build_dir.rglob("*.so*")):
        if not src.is_file():
            continue
        dst = binaries_dir / src.name
        if dst.exists():
            continue
        print(f"  - {src.name}")
        shutil.copy2(src, dst)
        copied_files.append(dst)
        shared_libraries.append(src.name)

    for script_name in CONVERT_SCRIPTS:
        src = source_dir / script_name
        if src.exists():
            dst = package_root / script_name
            shutil.copy2(src, dst)
            copied_files.append(dst)

    for doc_name in ["LICENSE", "README.md"]:
        src = source_dir / doc_name
        if src.exists():
            dst = package_root / doc_name
            shutil.copy2(src, dst)
            copied_files.append(dst)

    run_env = os.environ.copy()
    run_env["LD_LIBRARY_PATH"] = (
        f"{binaries_dir}:{run_env.get('LD_LIBRARY_PATH', '')}"
    ).rstrip(":")

    version_output = subprocess.run(
        [str(binaries_dir / "llama-cli"), "--version"],
        capture_output=True,
        check=True,
        env=run_env,
        text=True,
    )
    print(version_output.stderr)

    smoke_test = None
    if run_smoke_test:
        print(f"Downloading demo model: {demo_model_repo}/{demo_model_file}")
        model_path = hf_hub_download(
            repo_id=demo_model_repo,
            filename=demo_model_file,
            repo_type="model",
        )
        smoke_cmd = [
            str(binaries_dir / "llama-cli"),
            "-m",
            model_path,
            "-ngl",
            "99",
            "-n",
            "32",
            "-p",
            "Write one short sentence confirming this CUDA build works.",
        ]
        print(f"Running CUDA smoke test: {' '.join(smoke_cmd)}")
        smoke_result = subprocess.run(
            smoke_cmd,
            capture_output=True,
            env=run_env,
            text=True,
        )
        smoke_test = {
            "command": smoke_cmd,
            "returncode": smoke_result.returncode,
            "stdout_tail": smoke_result.stdout[-3000:],
            "stderr_tail": smoke_result.stderr[-3000:],
        }
        if smoke_result.returncode != 0:
            raise RuntimeError(
                "CUDA smoke test failed.\n"
                f"Exit code: {smoke_result.returncode}\n"
                f"STDERR:\n{smoke_result.stderr}\n"
                f"STDOUT:\n{smoke_result.stdout}"
            )
        print("CUDA smoke test completed successfully.")
        print(smoke_result.stdout[-3000:])

    metadata = {
        "source_repo": llama_repo_url,
        "source_ref": llama_ref,
        "source_commit": full_sha,
        "source_commit_short": short_sha,
        "built_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "builder_gpu_type": "T4",
        "configure_command": configure_cmd,
        "build_command": build_cmd,
        "cmake_version": cmake_version,
        "binaries": CORE_BINARIES,
        "shared_libraries": shared_libraries,
        "version_stderr": version_output.stderr,
        "smoke_test": smoke_test,
    }
    metadata_path = package_root / "build-metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    artifact_name = f"{artifact_prefix}-{timestamp}-{short_sha}.zip"
    artifact_zip = pathlib.Path(f"/tmp/{artifact_name}")
    print(f"Creating zip package at {artifact_zip} ...")
    with zipfile.ZipFile(artifact_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for file_path in copied_files:
            zf.write(file_path, arcname=file_path.relative_to(package_root))
        zf.write(metadata_path, arcname="build-metadata.json")
    print(f"Zip creation complete ({artifact_zip.stat().st_size / (1024 ** 2):.2f} MiB).")

    path_in_repo = f"builds/{artifact_name}"
    print(f"Uploading {artifact_name} to https://huggingface.co/{hf_repo} ...")
    api.upload_file(
        path_or_fileobj=str(artifact_zip),
        path_in_repo=path_in_repo,
        repo_id=hf_repo,
        repo_type="model",
    )
    print("Zip upload complete.")

    latest_file = pathlib.Path("/tmp/latest-ik-llama-cuda.txt")
    latest_file.write_text(
        f"{path_in_repo}\n{json.dumps(metadata, sort_keys=True)}\n",
        encoding="utf-8",
    )
    latest_path = "builds/latest-ik-llama-cuda.txt"
    api.upload_file(
        path_or_fileobj=str(latest_file),
        path_in_repo=latest_path,
        repo_id=hf_repo,
        repo_type="model",
    )
    print(f"Latest pointer uploaded to {latest_path}.")

    result = {
        "artifact_name": artifact_name,
        "path_in_repo": path_in_repo,
        "latest_path": latest_path,
        "repo_url": f"https://huggingface.co/{hf_repo}",
        "commit": short_sha,
        "smoke_test_passed": run_smoke_test,
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    hf_repo: str = "sandeshrajx/llama_cpp_colab_builds",
    llama_ref: str = "main",
    llama_repo_url: str = "https://github.com/ikawrakow/ik_llama.cpp.git",
    artifact_prefix: str = "ik-llama-cpp-cuda-default-colab",
    run_smoke_test: bool = True,
    demo_model_repo: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    demo_model_file: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
):
    result = build_smoke_test_and_upload.remote(
        hf_repo=hf_repo,
        llama_ref=llama_ref,
        llama_repo_url=llama_repo_url,
        artifact_prefix=artifact_prefix,
        run_smoke_test=run_smoke_test,
        demo_model_repo=demo_model_repo,
        demo_model_file=demo_model_file,
    )
    print("ik_llama.cpp build + upload completed.")
    print(json.dumps(result, indent=2))
