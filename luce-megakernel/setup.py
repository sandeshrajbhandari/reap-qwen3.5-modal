import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Modal L4 = Ada sm_89; RTX 3090 = Ampere sm_86. Set MEGAKERNEL_CUDA_SM=89 on L4.
_sm = os.environ.get("MEGAKERNEL_CUDA_SM", "86")
_nvcc_arch = f"-arch=sm_{_sm}"

setup(
    name="qwen35_megakernel_bf16",
    install_requires=["torch", "safetensors", "huggingface_hub"],
    extras_require={"autoround": ["safetensors", "huggingface_hub", "transformers"]},
    ext_modules=[
        CUDAExtension(
            name="qwen35_megakernel_bf16_C",
            sources=[
                "torch_bindings.cpp",
                "kernel.cu",
                "prefill.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    _nvcc_arch,
                    "--use_fast_math",
                    "-std=c++17",
                    "-DNUM_BLOCKS=82",
                    "-DBLOCK_SIZE=512",
                    "-DLM_NUM_BLOCKS=512",
                    "-DLM_BLOCK_SIZE=256",
                ],
            },
            libraries=["cublas"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
