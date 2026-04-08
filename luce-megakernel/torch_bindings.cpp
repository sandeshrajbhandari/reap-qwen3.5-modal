/**
 * PyTorch bindings for Qwen3.5-9B bf16 megakernel — decode.
 */

#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }

struct LayerWeights {
    int layer_type;
    int _pad[3];
    void *ptrs[14];  // max(11 FA, 14 DN) pointers — all bf16, no scales
};

extern "C" void launch_decode(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LayerWeights *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    int position, int max_seq_len, cudaStream_t stream);

void decode(
    torch::Tensor output_token, int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor qkv_scratch, torch::Tensor kv_scratch, torch::Tensor attn_out,
    torch::Tensor mlp_inter, torch::Tensor z_scratch, torch::Tensor beta_scratch,
    torch::Tensor alpha_scratch, torch::Tensor normalized,
    torch::Tensor barrier_counter, torch::Tensor barrier_generation,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    torch::Tensor lm_sync_counter, int64_t position, int64_t max_seq_len)
{
    launch_decode(
        (int)input_token_id, (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        qkv_scratch.data_ptr(), kv_scratch.data_ptr(), attn_out.data_ptr(),
        mlp_inter.data_ptr(), z_scratch.data_ptr(), beta_scratch.data_ptr(),
        alpha_scratch.data_ptr(), normalized.data_ptr(),
        (unsigned int*)barrier_counter.data_ptr(), (unsigned int*)barrier_generation.data_ptr(),
        (float*)block_max_vals.data_ptr(), (int*)block_max_idxs.data_ptr(),
        (unsigned int*)lm_sync_counter.data_ptr(),
        (int)position, (int)max_seq_len,
        c10::cuda::getCurrentCUDAStream().stream());
}

// ===== Prefill BF16 =====

extern "C" void launch_prefill_bf16(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const LayerWeights *layers,
    const void *final_norm_w, const void *lm_head_w,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2, void *attn_buf, void *mlp_buf,
    void *dn_out_buf, void *beta_buf, void *alpha_buf,
    void *final_normed, void *hidden_bf16_out,
    void *lm_bmv, void *lm_bmi,
    cudaStream_t stream);

void prefill_bf16(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2,
    torch::Tensor attn_buf, torch::Tensor mlp_buf,
    torch::Tensor dn_out_buf, torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor final_normed, torch::Tensor hidden_bf16_out,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi)
{
    launch_prefill_bf16(
        (const int*)token_ids.data_ptr(), token_ids.size(0),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(),
        dn_out_buf.data_ptr(), beta_buf.data_ptr(), alpha_buf.data_ptr(),
        final_normed.data_ptr(), hidden_bf16_out.data_ptr(),
        lm_bmv.data_ptr(), lm_bmi.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("decode(Tensor output_token, int input_token_id, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "int position, int max_seq_len) -> ()");
    ops.impl("decode", torch::kCUDA, &decode);

    ops.def("prefill_bf16(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, Tensor attn_buf, Tensor mlp_buf, "
            "Tensor dn_out_buf, Tensor beta_buf, Tensor alpha_buf, "
            "Tensor final_normed, Tensor hidden_bf16_out, "
            "Tensor lm_bmv, Tensor lm_bmi) -> ()");
    ops.impl("prefill_bf16", torch::kCUDA, &prefill_bf16);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
