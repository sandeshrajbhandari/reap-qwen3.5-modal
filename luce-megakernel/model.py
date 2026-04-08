"""Weight loading and decode API for Qwen3.5-9B bf16 megakernel (text stack)."""

import struct
import torch

# Must match kernel.cu / prefill.cu
NUM_LAYERS = 32
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12288
VOCAB_SIZE = 248320
MAX_SEQ_LEN = 2048

FA_NUM_Q_HEADS = 16
FA_NUM_KV_HEADS = 4
FA_HEAD_DIM = 256
FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM
FA_QPROJ_SIZE = FA_Q_SIZE * 2
FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM

DN_NUM_K_HEADS = 16
DN_NUM_V_HEADS = 32
DN_KEY_DIM = 128
DN_VALUE_DIM = 128
DN_QK_SIZE = DN_NUM_K_HEADS * DN_KEY_DIM
DN_V_SIZE = DN_NUM_V_HEADS * DN_VALUE_DIM
DN_CONV_CHANNELS = DN_QK_SIZE * 2 + DN_V_SIZE
DN_CONV_KERNEL = 4
DN_QK_BROADCAST_FLOATS = DN_NUM_K_HEADS * 2 * DN_KEY_DIM

LAYER_TYPE = [
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
]

_decode = None
_decode_w4 = None


def _load_ops():
    global _decode, _decode_w4
    if _decode is None:
        import qwen35_megakernel_bf16_C
        _decode = torch.ops.qwen35_megakernel_bf16_C.decode
        _decode_w4 = torch.ops.qwen35_megakernel_bf16_C.decode_w4


def load_weights(model_name="Qwen/Qwen3.5-9B", verbose=True):
    """Load Qwen3.5-9B text weights as bf16 (no quantization)."""
    if not verbose:
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoConfig, AutoTokenizer
    from hf_weights import merge_safetensors_shards, strip_to_language_state

    if verbose:
        print(f"Loading {model_name} (bf16)...")
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    lt_cfg = text_cfg.layer_types
    lt = [0 if x == "linear_attention" else 1 for x in lt_cfg]
    if len(lt) != NUM_LAYERS or lt != LAYER_TYPE:
        raise ValueError(
            f"Config layer_types ({len(lt)} layers) does not match megakernel LAYER_TYPE; "
            "update kernel.cu / model.py if the checkpoint pattern changed."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Load weights from disk (CPU) — avoids loading full Qwen3.5-VL into 22GB VRAM before stripping keys.
    merged = merge_safetensors_shards(model_name)
    raw_state = merged
    state_cpu = strip_to_language_state(merged)
    if "layers.0.input_layernorm.weight" not in state_cpu:
        raise KeyError(
            "Could not find layers.* in merged checkpoint; expected model.layers.* or model.language_model.layers.*"
        )

    def _get(k):
        if k not in state_cpu:
            raise KeyError(k)
        return state_cpu[k].to(device="cuda", dtype=torch.bfloat16).contiguous()

    layer_data = []
    for i in range(NUM_LAYERS):
        p = f"layers.{i}."
        layer_lt = lt[i]

        if layer_lt == 1:
            layer_data.append({
                "type": 1,
                "ptrs": [
                    _get(p + "input_layernorm.weight"),
                    _get(p + "self_attn.q_proj.weight"),
                    _get(p + "self_attn.k_proj.weight"),
                    _get(p + "self_attn.v_proj.weight"),
                    _get(p + "self_attn.q_norm.weight"),
                    _get(p + "self_attn.k_norm.weight"),
                    _get(p + "self_attn.o_proj.weight"),
                    _get(p + "post_attention_layernorm.weight"),
                    _get(p + "mlp.gate_proj.weight"),
                    _get(p + "mlp.up_proj.weight"),
                    _get(p + "mlp.down_proj.weight"),
                ]
            })
        else:
            layer_data.append({
                "type": 0,
                "ptrs": [
                    _get(p + "input_layernorm.weight"),
                    _get(p + "linear_attn.in_proj_qkv.weight"),
                    _get(p + "linear_attn.in_proj_z.weight"),
                    _get(p + "linear_attn.in_proj_b.weight"),
                    _get(p + "linear_attn.in_proj_a.weight"),
                    _get(p + "linear_attn.conv1d.weight"),
                    _get(p + "linear_attn.A_log"),
                    _get(p + "linear_attn.dt_bias"),
                    _get(p + "linear_attn.norm.weight"),
                    _get(p + "linear_attn.out_proj.weight"),
                    _get(p + "post_attention_layernorm.weight"),
                    _get(p + "mlp.gate_proj.weight"),
                    _get(p + "mlp.up_proj.weight"),
                    _get(p + "mlp.down_proj.weight"),
                ]
            })

    embed_weight = _get("embed_tokens.weight")
    final_norm_weight = _get("norm.weight")
    lm = _get("lm_head.weight") if "lm_head.weight" in raw_state else embed_weight

    weights = {
        "embed_weight": embed_weight,
        "final_norm_weight": final_norm_weight,
        "lm_head_weight": lm,
        "layer_data": layer_data,
    }

    del model
    torch.cuda.empty_cache()

    if verbose:
        total = sum(sum(t.numel() for t in ld["ptrs"]) for ld in layer_data) + lm.numel()
        print(f"BF16 weights: {total/1e6:.1f}M params ({total*2/1e6:.0f} MB)")

    return weights, tokenizer


def load_weights_intel_autoround(
    model_name: str = "Intel/Qwen3.5-9B-int4-AutoRound",
    verbose: bool = True,
):
    """Intel AutoRound int4 (`packing_format: auto_round:auto_gptq`). See `autoround_load.py`."""
    from autoround_load import load_weights_autoround

    return load_weights_autoround(model_name, verbose=verbose)


def _pack_layer_weights(layer_data):
    """Pack layer weights into device blob matching LayerWeights struct."""
    ptr_size = 8
    max_ptrs = 14
    header_size = 16
    struct_size = header_size + max_ptrs * ptr_size  # 128

    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = layer_data[i]
        offset = i * struct_size
        struct.pack_into("iiii", buf, offset, ld["type"], 0, 0, 0)
        for j, tensor in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, tensor.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class Decoder:
    """Stateful decoder for Qwen3.5-9B megakernel (bf16 or Intel AutoRound int4)."""

    def __init__(self, weights=None, tokenizer=None,
                 model_name=None, verbose=True,
                 weight_mode: str = "bf16"):
        _load_ops()

        if weights is None:
            if weight_mode == "bf16":
                mn = model_name or "Qwen/Qwen3.5-9B"
                weights, tokenizer = load_weights(mn, verbose=verbose)
            elif weight_mode in ("autoround_int4", "intel_autoround"):
                mn = model_name or "Intel/Qwen3.5-9B-int4-AutoRound"
                weights, tokenizer = load_weights_intel_autoround(mn, verbose=verbose)
            else:
                raise ValueError(f"Unknown weight_mode: {weight_mode}")
        self.tokenizer = tokenizer
        self._position = 0
        self._weights = weights
        self._weight_mode = weights.get("weight_mode", weight_mode)
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        if self._weight_mode == "bf16":
            self._layer_weights_packed = _pack_layer_weights(weights["layer_data"])
            self._run_decode = _decode
            self._w_dequant_scratch = None
        else:
            self._layer_weights_packed = weights["layer_weights_packed"]
            self._run_decode = _decode_w4
            from autoround_load import max_dequant_bf16_elems

            self._w_dequant_scratch = torch.empty(
                max_dequant_bf16_elems(), dtype=torch.bfloat16, device="cuda"
            )

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        n_fa = sum(1 for t in LAYER_TYPE if t == 1)
        self._fa_k_cache = torch.zeros(n_fa, FA_NUM_KV_HEADS, MAX_SEQ_LEN, FA_HEAD_DIM, **bf16)
        self._fa_v_cache = torch.zeros_like(self._fa_k_cache)

        n_dn = sum(1 for t in LAYER_TYPE if t == 0)
        self._dn_states = torch.zeros(n_dn, DN_NUM_V_HEADS, DN_KEY_DIM, DN_VALUE_DIM, **f32)
        self._conv_bufs = torch.zeros(n_dn, DN_CONV_CHANNELS, DN_CONV_KERNEL, **f32)

        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        max_scratch = max(
            FA_QPROJ_SIZE,
            DN_CONV_CHANNELS,
            HIDDEN_SIZE * 8 + INTERMEDIATE_SIZE,
            DN_QK_BROADCAST_FLOATS,
        )
        self._activations = torch.empty(max_scratch, **f32)
        self._residual = torch.empty(HIDDEN_SIZE, **bf16)
        self._qkv_scratch = torch.empty(max(FA_QPROJ_SIZE, DN_CONV_CHANNELS), **f32)
        self._kv_scratch = torch.empty(FA_KV_SIZE * 2, **f32)
        self._attn_out = torch.empty(max(FA_Q_SIZE, DN_V_SIZE), **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._z_scratch = torch.empty(DN_V_SIZE, **f32)
        self._beta_scratch = torch.empty(DN_NUM_V_HEADS, **f32)
        self._alpha_scratch = torch.empty(DN_NUM_V_HEADS, **f32)
        self._normalized = torch.empty(HIDDEN_SIZE, **f32)

        self._barrier_counter = torch.zeros(1, **u32)
        self._barrier_generation = torch.zeros(1, **u32)
        self._block_max_vals = torch.empty(1024, **f32)
        self._block_max_idxs = torch.empty(1024, **i32)
        self._lm_sync_counter = torch.zeros(1, **u32)
        self._out_token = torch.empty(1, **i32)

    def step(self, token_id: int) -> int:
        """Decode one token. Returns next token id."""
        self._run_decode(
            self._out_token, token_id,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight,
            self._fa_k_cache, self._fa_v_cache,
            self._dn_states, self._conv_bufs,
            self._hidden, self._activations, self._residual,
            self._qkv_scratch, self._kv_scratch, self._attn_out,
            self._mlp_inter, self._z_scratch, self._beta_scratch,
            self._alpha_scratch, self._normalized,
            self._barrier_counter, self._barrier_generation,
            self._block_max_vals, self._block_max_idxs,
            self._lm_sync_counter,
            self._position, MAX_SEQ_LEN,
        )
        self._position += 1
        return self._out_token.item()

    def reset(self):
        self._position = 0
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()
        self._dn_states.zero_()
        self._conv_bufs.zero_()

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)
        out = []
        next_id = ids[-1]
        eos = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            next_id = self.step(next_id)
            if next_id == eos:
                break
            out.append(next_id)
        return self.tokenizer.decode(out, skip_special_tokens=True)

    def prefill(self, token_ids_cuda: torch.Tensor, bufs: dict) -> int:
        """Run prefill on GPU int32 token ids; `bufs` layout matches bench_pp_tg."""
        if self._weight_mode == "bf16":
            _pf = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16
            _pf(
                self._out_token,
                token_ids_cuda,
                self._embed_weight,
                self._layer_weights_packed,
                self._final_norm_weight,
                self._lm_head_weight,
                self._fa_k_cache,
                self._fa_v_cache,
                self._dn_states,
                self._conv_bufs,
                bufs["hidden"],
                bufs["residual"],
                bufs["normalized"],
                bufs["proj_buf"],
                bufs["proj_buf2"],
                bufs["attn_buf"],
                bufs["mlp_buf"],
                bufs["dn_out_buf"],
                bufs["beta_buf"],
                bufs["alpha_buf"],
                bufs["final_normed"],
                bufs["hidden_bf16_out"],
                bufs["lm_bmv"],
                bufs["lm_bmi"],
            )
        else:
            _pf = torch.ops.qwen35_megakernel_bf16_C.prefill_w4
            _pf(
                self._out_token,
                token_ids_cuda,
                self._embed_weight,
                self._layer_weights_packed,
                self._final_norm_weight,
                self._lm_head_weight,
                self._fa_k_cache,
                self._fa_v_cache,
                self._dn_states,
                self._conv_bufs,
                bufs["hidden"],
                bufs["residual"],
                bufs["normalized"],
                bufs["proj_buf"],
                bufs["proj_buf2"],
                bufs["attn_buf"],
                bufs["mlp_buf"],
                bufs["dn_out_buf"],
                bufs["beta_buf"],
                bufs["alpha_buf"],
                bufs["final_normed"],
                bufs["hidden_bf16_out"],
                bufs["lm_bmv"],
                bufs["lm_bmi"],
                self._w_dequant_scratch,
            )
        self._hidden.copy_(bufs["hidden_bf16_out"])
        self._position = int(token_ids_cuda.numel())
        return self._out_token.item()
