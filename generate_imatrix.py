import os
import pathlib

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
app = modal.App("imatrix-generator")

# Use a CUDA-enabled base image for building and running
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "cmake", "build-essential", "libcurl4-openssl-dev")
    .pip_install(
        "torch",
        "accelerate",
        "huggingface-hub",
        "numpy",
        "sentencepiece",
        "gguf",
        "git+https://github.com/huggingface/transformers.git",  # For Qwen3.5 support in conversion
    )
)

# Volumes
results_vol = modal.Volume.from_name("reap-results")
RESULTS_DIR = "/results"
LLAMA_CPP_BIN_DIR = os.path.join(RESULTS_DIR, "tools/llama.cpp")

# Paths
MODEL_SUBPATH = "Qwen3.5-35B-A3B/evol-codealpaca-v1/pruned_models/reap-seed_42-0.32"
MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_SUBPATH)
F16_GGUF_PATH = os.path.join(MODEL_PATH, "model-f16.gguf")
IMATRIX_PATH = os.path.join(MODEL_PATH, "imatrix.dat")

# CALIBRATION_URL = "https://gist.githubusercontent.com/ubergarm/edfeb3ff9c6ec8b49e88cdf627b0711a/raw/ba5b01b6960a86874592f5913e283746ff734483/ubergarm-imatrix-calibration-corpus-v02.txt"
CALIBRATION_URL = "https://huggingface.co/spaces/Novaciano/Train-With-Erotiquant3/raw/1ed03c8d5eb4f359942ee3a8c209a68bcee5cdb4/calibration_data_v5_rc.txt"


# ---------------------------------------------------------------------------
# Step 1: Build llama.cpp with CUDA and store in Volume
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",  # GPU helps cmake detect CUDA arch correctly
    volumes={RESULTS_DIR: results_vol},
    timeout=1800,  # 30 mins
)
def build_llama_cpp():
    import shutil
    import subprocess

    print("🛠️  Cloning and building llama.cpp with CUDA...")
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/ggml-org/llama.cpp.git",
            "/root/llama.cpp",
        ],
        check=True,
    )

    llama_dir = "/root/llama.cpp"
    build_dir = "/root/llama.cpp/build"
    os.makedirs(build_dir, exist_ok=True)

    # Configure with CUDA
    subprocess.run(
        [
            "cmake",
            "-S",
            llama_dir,
            "-B",
            build_dir,
            "-DGGML_CUDA=ON",
            "-DLLAMA_CURL=ON",
            "-DBUILD_SHARED_LIBS=OFF",
        ],
        check=True,
    )

    # Build
    subprocess.run(
        ["cmake", "--build", build_dir, "--config", "Release", "-j", "16"], check=True
    )

    # Store in volume
    os.makedirs(LLAMA_CPP_BIN_DIR, exist_ok=True)
    binaries = ["llama-cli", "llama-imatrix", "llama-quantize"]
    for b in binaries:
        src = os.path.join(build_dir, "bin", b)
        dst = os.path.join(LLAMA_CPP_BIN_DIR, b)
        shutil.copy2(src, dst)
        print(f"✅ Saved {b} to {dst}")

    # Also copy the conversion script
    shutil.copy2(
        "/root/llama.cpp/convert_hf_to_gguf.py",
        os.path.join(LLAMA_CPP_BIN_DIR, "convert_hf_to_gguf.py"),
    )

    results_vol.commit()
    print("🎉 llama.cpp build complete and stored in volume.")


# ---------------------------------------------------------------------------
# Step 2: Generate imatrix
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RESULTS_DIR: results_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,  # 2 hours
)
def generate_imatrix():
    import json
    import os
    import subprocess

    # 1. Convert to F16 GGUF if not exists
    if not os.path.exists(F16_GGUF_PATH):
        print("🔄 Converting HF model to F16 GGUF...")

        # Patch config.json for Qwen3.5 support in llama.cpp
        config_path = os.path.join(MODEL_PATH, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        orig_arch = config["architectures"]
        config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        try:
            convert_script = os.path.join(LLAMA_CPP_BIN_DIR, "convert_hf_to_gguf.py")
            subprocess.run(
                [
                    "python3",
                    convert_script,
                    MODEL_PATH,
                    "--outfile",
                    F16_GGUF_PATH,
                    "--outtype",
                    "f16",
                ],
                check=True,
            )
            print(f"✅ Created F16 GGUF at {F16_GGUF_PATH}")
        finally:
            # Restore
            config["architectures"] = orig_arch
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
    else:
        print(f"✨ F16 GGUF already exists at {F16_GGUF_PATH}")

    # 2. Download calibration data
    cal_file = "/tmp/calibration.txt"
    print(f"⬇️  Downloading calibration data from {CALIBRATION_URL}...")
    subprocess.run(["wget", CALIBRATION_URL, "-O", cal_file], check=True)

    # 3. Run imatrix
    print("𓌳  Generating imatrix (this will take time)...")
    imatrix_bin = os.path.join(LLAMA_CPP_BIN_DIR, "llama-imatrix")

    # We use -ngl to offload to GPU
    cmd = [
        imatrix_bin,
        "--verbosity",
        "1",
        "-m",
        F16_GGUF_PATH,
        "-f",
        cal_file,
        "-o",
        IMATRIX_PATH,
        "-ngl",
        "99",  # Offload all layers to A100
        "--ctx-size",
        "512",
        "--threads",
        "16",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    results_vol.commit()
    print(f"✅ imatrix generated at {IMATRIX_PATH}")


@app.local_entrypoint()
def main(rebuild: bool = False):
    # Check if tools exist
    tools_exist = False
    try:
        # Check if volume has the binaries
        # We can't check directly from local, so we just run a small function or assume
        pass
    except:
        pass

    if rebuild:
        build_llama_cpp.remote()

    # Always try to build if not sure, or just run
    # For the first time, the user should run with --rebuild

    generate_imatrix.remote()
