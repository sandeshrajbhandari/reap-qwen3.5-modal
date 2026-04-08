from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qwen35_megakernel_bf16",
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
                    "-arch=sm_86",
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
