thbe pruned model is in a modal volume storage.
reap-results -> Qwen3.5-35B-A3B/evol-codealpaca-v1/pruned_models/reap-seed_42-0.32

git clone https://github.com/ggml-org/llama.cpp.git

# from Hugginface, obtain the official meta-llama/Llama-3.1-8B model weights and place them in ./models
ls ./models
config.json             model-00001-of-00004.safetensors  model-00004-of-00004.safetensors  README.md                tokenizer.json
generation_config.json  model-00002-of-00004.safetensors  model.safetensors.index.json      special_tokens_map.json  USE_POLICY.md
LICENSE                 model-00003-of-00004.safetensors  original                          tokenizer_config.json

# [Optional] for PyTorch .bin models like Mistral-7B
ls ./models
<folder containing weights and tokenizer json>

# install Python dependencies
python3 -m pip install -r requirements.txt

# convert the model to ggml FP16 format
python3 convert_hf_to_gguf.py ./models/mymodel/

# quantize the model to 4-bits (using Q4_K_M method)
./llama-quantize ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_M.gguf Q4_K_M

you can find llama-quantize in https://github.com/ggml-org/llama.cpp/releases/download/b8192/llama-b8192-bin-ubuntu-vulkan-x64.tar.gz 

run it in modal.
