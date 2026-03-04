some commands from colab notebook for reference

git clone -b fix/gpt-oss-smoke-test-and-observer https://github.com/sandeshrajbhandari/reap.git

install uv if not !curl -LsSf https://astral.sh/uv/install.sh | sh

pip install git+https://github.com/huggingface/transformers.git
pip install kernels
!pip install triton==3.5.1 kernels
!pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels

need to run 
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda",
)
to download the model first cause of this error when running prune.py
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

using only observer state
!python ./reap/src/reap/prune.py \
    --model-name "openai/gpt-oss-20b" \
    --run_observer_only true \
    --samples_per_category 1
// using samples per category 1 for fast sampling, usually uses 128 
    
python ./reap/src/reap/prune.py \
        --model-name "openai/gpt-oss-20b" \
        --compression-ratio 0.32 \
        --prune-method reap
