/**
 * BF16 Prefill: cuBLAS bf16 GEMM + standalone recurrence kernel.
 * Weights bf16, activations bf16, state f32. No quantization, no conversion.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

constexpr int HIDDEN = 4096;
constexpr int INTER = 12288;
constexpr int VOCAB = 248320;
constexpr float RMS_EPS = 1e-6f;

constexpr int FA_Q_HEADS = 16;
constexpr int FA_KV_HEADS = 4;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA = FA_Q_HEADS / FA_KV_HEADS;
constexpr int FA_Q_SIZE = FA_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE * 2;
constexpr int FA_KV_SIZE = FA_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROT_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

constexpr int DN_K_HEADS = 16;
constexpr int DN_V_HEADS = 32;
constexpr int DN_V_PER_K = DN_V_HEADS / DN_K_HEADS;
constexpr int DN_KEY = 128;
constexpr int DN_VAL = 128;
constexpr int DN_CONV_K = 4;
constexpr int DN_QK_SIZE = DN_K_HEADS * DN_KEY;
constexpr int DN_V_SIZE = DN_V_HEADS * DN_VAL;
constexpr int DN_CONV_CH = DN_QK_SIZE * 2 + DN_V_SIZE;

constexpr int NUM_LAYERS = 32;
constexpr int LAYER_TYPE[32] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

struct PFLayerWeights { int layer_type; int _pad[3]; void *ptrs[14]; };

constexpr int Q4_GS = 128;
constexpr int Q4_PACK = 8;

#pragma pack(push, 8)
struct PFQ4 { void *qweight; void *scales; void *qzeros; };
struct PFFD_W4 {
    void *input_ln;
    PFQ4 qkv, z, beta, alpha;
    void *conv_w, *a_log, *dt_bias, *norm_w;
    PFQ4 out_proj;
    void *post_ln;
    PFQ4 gate, up, down;
};
struct PFFA_W4 {
    void *input_ln;
    PFQ4 q_proj, k_proj, v_proj;
    void *q_norm, *k_norm;
    PFQ4 o_proj;
    void *post_ln;
    PFQ4 gate, up, down;
    char _fa_pad[40];
};
struct PFLayerWeightsW4 {
    int layer_type;
    int _pad[3];
    union { PFFD_W4 dn; PFFA_W4 fa; };
};
#pragma pack(pop)

__device__ __forceinline__ float pf_w4_at(
    const int *qw, int in_dim, int out_dim, const __half *sc, const int *qz, int m, int k)
{
    int in_p = k / Q4_PACK, sub = k % Q4_PACK;
    int packed = __ldg(qw + (size_t)in_p * out_dim + m);
    int quint = (packed >> (4 * sub)) & 0xF;
    int g = k / Q4_GS;
    int zp_word = __ldg(qz + (size_t)g * (out_dim / Q4_PACK) + (m / Q4_PACK));
    int zp_nib = (zp_word >> (4 * (m % Q4_PACK))) & 0xF;
    float s = __half2float(__ldg(sc + (size_t)g * out_dim + m));
    return ((float)quint - (float)zp_nib) * s;
}

__global__ void pf_dequant_linear(
    const int *qw, const __half *sc, const int *qz,
    __nv_bfloat16 *out_rowmajor, int in_dim, int out_dim)
{
    int m = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= out_dim || k >= in_dim) return;
    float v = pf_w4_at(qw, in_dim, out_dim, sc, qz, m, k);
    out_rowmajor[m * in_dim + k] = __float2bfloat16(v);
}

static void launch_dequant_linear(
    const int *qw, const __half *sc, const int *qz,
    __nv_bfloat16 *out, int in_dim, int out_dim, cudaStream_t stream)
{
    dim3 blk(256, 1, 1);
    dim3 grd((in_dim + 255) / 256, out_dim, 1);
    pf_dequant_linear<<<grd, blk, 0, stream>>>(qw, sc, qz, out, in_dim, out_dim);
}

__device__ __forceinline__ float pf_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o); return v;
}
__device__ __forceinline__ float pf_silu(float x) { return x / (1.0f + expf(-x)); }

// Embedding
__global__ void pf_embed(const int *ids, const __nv_bfloat16 *embed, __nv_bfloat16 *out, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * HIDDEN) return;
    out[idx] = embed[ids[idx / HIDDEN] * HIDDEN + idx % HIDDEN];
}

// Batched RMSNorm: bf16 in → bf16 out, saves bf16 residual
__global__ void pf_rmsnorm(const __nv_bfloat16 *in, const __nv_bfloat16 *w,
    __nv_bfloat16 *out, __nv_bfloat16 *res, int S, int D) {
    int s = blockIdx.x; if (s >= S) return;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    __shared__ float smem[32];
    const __nv_bfloat16 *ri = in + s*D;
    __nv_bfloat16 *ro = out + s*D, *rr = res + s*D;
    float sq = 0;
    for (int i = tid; i < D; i += blockDim.x) { float v = __bfloat162float(ri[i]); rr[i] = ri[i]; sq += v*v; }
    sq = pf_warp_sum(sq); if(lid==0) smem[wid]=sq; __syncthreads();
    if(wid==0){float v=(lid<blockDim.x/32)?smem[lid]:0;v=pf_warp_sum(v);if(lid==0)smem[0]=rsqrtf(v/D+RMS_EPS);}
    __syncthreads(); float rstd = smem[0];
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __bfloat162float(ri[i]) * rstd * (1.0f + __bfloat162float(w[i]));
        ro[i] = __float2bfloat16(v);
    }
}

// bf16 matvec for tiny projections (beta/alpha)
__global__ void pf_bf16_matvec(const __nv_bfloat16 *in, const __nv_bfloat16 *w, float *out, int S, int K, int N) {
    int idx = blockIdx.x; if (idx >= S * N) return;
    int s = idx / N, n = idx % N, lid = threadIdx.x;
    const __nv_bfloat16 *ir = in + s*K, *wr = w + n*K;
    float sum = 0;
    for (int k = lid; k < K; k += 32) sum += __bfloat162float(ir[k]) * __bfloat162float(wr[k]);
    sum = pf_warp_sum(sum);
    if (lid == 0) out[idx] = sum;
}

// bf16 result + bf16 residual → bf16 output
__global__ void pf_add_residual_bf16(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

// SiLU(gate) * up — bf16 inputs → bf16 output
__global__ void pf_silu_mul_bf16(const __nv_bfloat16 *gate, const __nv_bfloat16 *up, __nv_bfloat16 *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { float g = __bfloat162float(gate[i]); out[i] = __float2bfloat16(pf_silu(g) * __bfloat162float(up[i])); }
}

// ===== Standalone DeltaNet recurrence (state-in-registers, bf16 I/O, f32 state) =====
__global__ void __launch_bounds__(512, 1)
pf_deltanet_recurrence(
    const __nv_bfloat16 *qkv_proj, const __nv_bfloat16 *z_proj,
    const float *beta_proj, const float *alpha_proj,
    const __nv_bfloat16 *conv_w, const __nv_bfloat16 *a_log,
    const __nv_bfloat16 *dt_bias, const __nv_bfloat16 *norm_w,
    float *state, float *conv_buf, __nv_bfloat16 *output, int S)
{
    int k_head = blockIdx.x;
    if (k_head >= DN_K_HEADS) return;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    constexpr int NWARPS = 16;
    constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
    constexpr int CPW = DN_VAL / NWARPS;
    constexpr int RPL = DN_KEY / 32;

    __shared__ float s_q[DN_KEY], s_k[DN_KEY], s_v[DN_VAL];
    __shared__ float s_beta, s_decay;
    __shared__ float s_gnorm[NWARPS];

    float sreg[DN_V_PER_K][CPW * RPL];

    for (int vs = 0; vs < DN_V_PER_K; vs++) {
        int vh = k_head * DN_V_PER_K + vs;
        float *my_state = state + vh * DN_KEY * DN_VAL;
        for (int jj = 0; jj < CPW; jj++) {
            int j = wid * CPW + jj;
            for (int ii = 0; ii < RPL; ii++)
                sreg[vs][jj*RPL+ii] = my_state[j*DN_KEY + lid+ii*32];
        }
    }

    for (int t = 0; t < S; t++) {
        for (int c = tid; c < DN_KEY; c += 512) {
            int ch = k_head*DN_KEY + c;
            float h0=conv_buf[ch*DN_CONV_K+1],h1=conv_buf[ch*DN_CONV_K+2],h2=conv_buf[ch*DN_CONV_K+3];
            conv_buf[ch*DN_CONV_K]=h0;conv_buf[ch*DN_CONV_K+1]=h1;conv_buf[ch*DN_CONV_K+2]=h2;
            conv_buf[ch*DN_CONV_K+3]=__bfloat162float(qkv_proj[t*DN_CONV_CH+ch]);
            float co=0;for(int k=0;k<DN_CONV_K;k++)co+=conv_buf[ch*DN_CONV_K+k]*__bfloat162float(conv_w[ch*DN_CONV_K+k]);
            s_q[c]=pf_silu(co);
        }
        for (int c = tid; c < DN_KEY; c += 512) {
            int ch = DN_QK_SIZE + k_head*DN_KEY + c;
            float h0=conv_buf[ch*DN_CONV_K+1],h1=conv_buf[ch*DN_CONV_K+2],h2=conv_buf[ch*DN_CONV_K+3];
            conv_buf[ch*DN_CONV_K]=h0;conv_buf[ch*DN_CONV_K+1]=h1;conv_buf[ch*DN_CONV_K+2]=h2;
            conv_buf[ch*DN_CONV_K+3]=__bfloat162float(qkv_proj[t*DN_CONV_CH+ch]);
            float co=0;for(int k=0;k<DN_CONV_K;k++)co+=conv_buf[ch*DN_CONV_K+k]*__bfloat162float(conv_w[ch*DN_CONV_K+k]);
            s_k[c]=pf_silu(co);
        }
        __syncthreads();

        if(wid==0){float sq=0;for(int i=lid;i<DN_KEY;i+=32)sq+=s_q[i]*s_q[i];sq=pf_warp_sum(sq);float n=rsqrtf(sq+1e-6f)*Q_SCALE;n=__shfl_sync(0xffffffff,n,0);for(int i=lid;i<DN_KEY;i+=32)s_q[i]*=n;}
        if(wid==1){float sq=0;for(int i=lid;i<DN_KEY;i+=32)sq+=s_k[i]*s_k[i];sq=pf_warp_sum(sq);float n=rsqrtf(sq+1e-6f);n=__shfl_sync(0xffffffff,n,0);for(int i=lid;i<DN_KEY;i+=32)s_k[i]*=n;}
        __syncthreads();

        for (int vs = 0; vs < DN_V_PER_K; vs++) {
            int vh = k_head * DN_V_PER_K + vs;
            for (int c = tid; c < DN_VAL; c += 512) {
                int ch = 2*DN_QK_SIZE + vh*DN_VAL + c;
                float h0=conv_buf[ch*DN_CONV_K+1],h1=conv_buf[ch*DN_CONV_K+2],h2=conv_buf[ch*DN_CONV_K+3];
                conv_buf[ch*DN_CONV_K]=h0;conv_buf[ch*DN_CONV_K+1]=h1;conv_buf[ch*DN_CONV_K+2]=h2;
                conv_buf[ch*DN_CONV_K+3]=__bfloat162float(qkv_proj[t*DN_CONV_CH+ch]);
                float co=0;for(int k=0;k<DN_CONV_K;k++)co+=conv_buf[ch*DN_CONV_K+k]*__bfloat162float(conv_w[ch*DN_CONV_K+k]);
                s_v[c]=pf_silu(co);
            }
            __syncthreads();

            float a_log_val = __bfloat162float(a_log[vh]);
            float dt_b = __bfloat162float(dt_bias[vh]);
            if(tid==0){
                s_beta=1.f/(1.f+expf(-beta_proj[t*DN_V_HEADS+vh]));
                float x=alpha_proj[t*DN_V_HEADS+vh]+dt_b;
                float sp=(x>20.f)?x:logf(1.f+expf(x));
                s_decay=expf(-expf(a_log_val)*sp);
            }
            __syncthreads();
            float beta = s_beta, decay = s_decay;
            __nv_bfloat16 *out_h = output + t * DN_V_SIZE + vh * DN_VAL;

            for (int jj = 0; jj < CPW; jj++) {
                int j = wid * CPW + jj;
                float kv = 0;
                for (int ii = 0; ii < RPL; ii++) kv += sreg[vs][jj*RPL+ii] * s_k[lid+ii*32];
                kv = pf_warp_sum(kv); kv = __shfl_sync(0xffffffff, kv, 0);
                float delta = (s_v[j] - decay * kv) * beta;
                float attn = 0;
                for (int ii = 0; ii < RPL; ii++) {
                    sreg[vs][jj*RPL+ii] = decay * sreg[vs][jj*RPL+ii] + s_k[lid+ii*32] * delta;
                    attn += sreg[vs][jj*RPL+ii] * s_q[lid+ii*32];
                }
                attn = pf_warp_sum(attn);
                if (lid == 0) out_h[j] = __float2bfloat16(attn);
            }
            __syncthreads();

            const __nv_bfloat16 *z_h = z_proj + t*DN_V_SIZE + vh*DN_VAL;
            float sq2=0;for(int i=tid;i<DN_VAL;i+=512){float v=__bfloat162float(out_h[i]);sq2+=v*v;}
            sq2=pf_warp_sum(sq2);if(lid==0)s_gnorm[wid]=sq2;__syncthreads();
            if(wid==0){float v=(lid<NWARPS)?s_gnorm[lid]:0;v=pf_warp_sum(v);if(lid==0)s_gnorm[0]=rsqrtf(v/DN_VAL+RMS_EPS);}
            __syncthreads();float rstd=s_gnorm[0];
            for(int i=tid;i<DN_VAL;i+=512){
                float n=__bfloat162float(out_h[i])*rstd*__bfloat162float(norm_w[vh*DN_VAL+i]);
                out_h[i]=__float2bfloat16(n*pf_silu(__bfloat162float(z_h[i])));
            }
            __syncthreads();
        }
    }

    for (int vs = 0; vs < DN_V_PER_K; vs++) {
        int vh = k_head * DN_V_PER_K + vs;
        float *my_state = state + vh * DN_KEY * DN_VAL;
        for (int jj = 0; jj < CPW; jj++) {
            int j = wid * CPW + jj;
            for (int ii = 0; ii < RPL; ii++)
                my_state[j*DN_KEY + lid+ii*32] = sreg[vs][jj*RPL+ii];
        }
    }
}

// ===== QK norm + RoPE + KV cache =====
__global__ void pf_qk_norm_rope(
    __nv_bfloat16 *q, __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const __nv_bfloat16 *qnw, const __nv_bfloat16 *knw,
    __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, int S, int max_seq)
{
    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    int total_q = S * FA_Q_HEADS, total_k = S * FA_KV_HEADS;
    if (idx < total_q) {
        int pos = idx / FA_Q_HEADS, head = idx % FA_Q_HEADS;
        __nv_bfloat16 *qh = q + pos * FA_QPROJ_SIZE + head * FA_HEAD_DIM * 2;
        float ss = 0; for (int i = lid; i < FA_HEAD_DIM; i += 32) { float v = __bfloat162float(qh[i]); ss += v*v; }
        ss = pf_warp_sum(ss); float sc = rsqrtf(ss/FA_HEAD_DIM+RMS_EPS); sc = __shfl_sync(0xffffffff,sc,0);
        for (int i = lid; i < FA_HEAD_DIM; i += 32) {
            float normed = __bfloat162float(qh[i])*sc*(1.f+__bfloat162float(qnw[i]));
            if (i < FA_ROT_DIM) {
                float fe=float(2*(i%(FA_ROT_DIM/2)))/FA_ROT_DIM; float freq=float(pos)/powf(FA_ROPE_THETA,fe);
                float cv=cosf(freq),sv=sinf(freq); int p=(i<FA_ROT_DIM/2)?i+FA_ROT_DIM/2:i-FA_ROT_DIM/2;
                float pv=__bfloat162float(qh[p])*sc*(1.f+__bfloat162float(qnw[p]));
                qh[i]=__float2bfloat16((i<FA_ROT_DIM/2)?(normed*cv-pv*sv):(pv*sv+normed*cv));
            } else qh[i]=__float2bfloat16(normed);
        }
    }
    int kidx = idx - total_q;
    if (idx >= total_q && kidx < total_k) {
        int pos = kidx / FA_KV_HEADS, head = kidx % FA_KV_HEADS;
        __nv_bfloat16 *kh = k + pos*FA_KV_SIZE + head*FA_HEAD_DIM;
        const __nv_bfloat16 *vh = v + pos*FA_KV_SIZE + head*FA_HEAD_DIM;
        __nv_bfloat16 *kc = k_cache + head*max_seq*FA_HEAD_DIM + pos*FA_HEAD_DIM;
        __nv_bfloat16 *vc = v_cache + head*max_seq*FA_HEAD_DIM + pos*FA_HEAD_DIM;
        float ss = 0; for (int i = lid; i < FA_HEAD_DIM; i += 32) { float v = __bfloat162float(kh[i]); ss += v*v; }
        ss = pf_warp_sum(ss); float sc = rsqrtf(ss/FA_HEAD_DIM+RMS_EPS); sc = __shfl_sync(0xffffffff,sc,0);
        for (int i = lid; i < FA_HEAD_DIM; i += 32) {
            float normed = __bfloat162float(kh[i])*sc*(1.f+__bfloat162float(knw[i])); float fk;
            if (i < FA_ROT_DIM) {
                float fe=float(2*(i%(FA_ROT_DIM/2)))/FA_ROT_DIM; float freq=float(pos)/powf(FA_ROPE_THETA,fe);
                float cv=cosf(freq),sv=sinf(freq); int p=(i<FA_ROT_DIM/2)?i+FA_ROT_DIM/2:i-FA_ROT_DIM/2;
                float pv=__bfloat162float(kh[p])*sc*(1.f+__bfloat162float(knw[p]));
                fk=(i<FA_ROT_DIM/2)?(normed*cv-pv*sv):(pv*sv+normed*cv);
            } else fk=normed;
            kh[i]=__float2bfloat16(fk); kc[i]=__float2bfloat16(fk); vc[i]=vh[i];
        }
    }
}

// ===== Causal attention (bf16 Q/K/V, f32 accumulation, bf16 output) =====
__global__ void pf_causal_attn(const __nv_bfloat16 *q, const __nv_bfloat16 *k,
    const __nv_bfloat16 *v, __nv_bfloat16 *out, int S)
{
    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    if (idx >= S * FA_Q_HEADS) return;
    int pos = idx / FA_Q_HEADS, qh = idx % FA_Q_HEADS, kvh = qh / FA_GQA;
    float scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
    constexpr int EPL = FA_HEAD_DIM / 32;
    const __nv_bfloat16 *qv = q + pos*FA_QPROJ_SIZE + qh*FA_HEAD_DIM*2;
    const __nv_bfloat16 *gv = qv + FA_HEAD_DIM;
    __nv_bfloat16 *ov = out + pos*FA_Q_SIZE + qh*FA_HEAD_DIM;
    float ql[EPL]; for(int e=0;e<EPL;e++) ql[e]=__bfloat162float(qv[lid*EPL+e]);
    float oa[EPL]={}; float mx=-1e30f, se=0;
    for (int kp = 0; kp <= pos; kp++) {
        const __nv_bfloat16 *kv=k+kp*FA_KV_SIZE+kvh*FA_HEAD_DIM;
        const __nv_bfloat16 *vv=v+kp*FA_KV_SIZE+kvh*FA_HEAD_DIM;
        float sc=0; for(int e=0;e<EPL;e++) sc+=ql[e]*__bfloat162float(kv[lid*EPL+e]);
        sc=pf_warp_sum(sc)*scale; sc=__shfl_sync(0xffffffff,sc,0);
        float om=mx; mx=fmaxf(mx,sc); float ed=expf(om-mx); se=se*ed+expf(sc-mx);
        float wt=expf(sc-mx); for(int e=0;e<EPL;e++) oa[e]=oa[e]*ed+wt*__bfloat162float(vv[lid*EPL+e]);
    }
    float rs=1.f/se;
    for(int e=0;e<EPL;e++){int i=lid*EPL+e;float g=1.f/(1.f+expf(-__bfloat162float(gv[i])));ov[i]=__float2bfloat16(oa[e]*rs*g);}
}

// Final norm
__global__ void pf_final_norm(const __nv_bfloat16 *hidden, const __nv_bfloat16 *w,
    __nv_bfloat16 *normed, __nv_bfloat16 *hidden_out, int S) {
    int tid=threadIdx.x, wid=tid/32, lid=tid%32;
    __shared__ float smem[16];
    const __nv_bfloat16 *row = hidden + (S-1)*HIDDEN;
    float sq=0; for(int i=tid;i<HIDDEN;i+=blockDim.x){float v=__bfloat162float(row[i]);sq+=v*v;}
    sq=pf_warp_sum(sq);if(lid==0)smem[wid]=sq;__syncthreads();
    if(wid==0){float v=(lid<blockDim.x/32)?smem[lid]:0;v=pf_warp_sum(v);if(lid==0)smem[0]=rsqrtf(v/HIDDEN+RMS_EPS);}
    __syncthreads();float rstd=smem[0];
    for(int i=tid;i<HIDDEN;i+=blockDim.x){
        float v=__bfloat162float(row[i]);
        normed[i]=__float2bfloat16(v*rstd*(1.f+__bfloat162float(w[i])));
        hidden_out[i]=row[i];
    }
}

// LM head: bf16 weight × bf16 hidden
__global__ void pf_lm_head(const __nv_bfloat16 *hidden, const __nv_bfloat16 *w,
    float *bmv, int *bmi, int N) {
    __shared__ __nv_bfloat16 s_h[HIDDEN];
    for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x) s_h[i]=hidden[i];
    __syncthreads();
    int wid=threadIdx.x/32, lid=threadIdx.x%32, nw=blockDim.x/32;
    int rpb=(N+gridDim.x-1)/gridDim.x, rs=blockIdx.x*rpb, re=min(rs+rpb,N);
    float lm=-1e30f; int li=-1;
    for(int m=rs+wid;m<re;m+=nw){const __nv_bfloat16 *wr=w+m*HIDDEN;float s=0;
        for(int k=lid*8;k<HIDDEN;k+=32*8){for(int i=0;i<8;i++)s+=__bfloat162float(wr[k+i])*__bfloat162float(s_h[k+i]);}
        s=pf_warp_sum(s);if(lid==0&&s>lm){lm=s;li=m;}}
    lm=__shfl_sync(0xffffffff,lm,0);li=__shfl_sync(0xffffffff,li,0);
    __shared__ float wm[32]; __shared__ int wi[32];
    if(lid==0){wm[wid]=lm;wi[wid]=li;}__syncthreads();
    if(wid==0){float mv=(lid<nw)?wm[lid]:-1e30f;int mi=(lid<nw)?wi[lid]:-1;
        for(int o=16;o>0;o>>=1){float ov=__shfl_down_sync(0xffffffff,mv,o);int oi=__shfl_down_sync(0xffffffff,mi,o);if(ov>mv){mv=ov;mi=oi;}}
        if(lid==0){bmv[blockIdx.x]=mv;bmi[blockIdx.x]=mi;}}
}
__global__ void pf_bf16_to_float(const __nv_bfloat16 *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

__global__ void pf_lm_reduce(const float *bmv, const int *bmi, int *out, int nb) {
    int tid=threadIdx.x; float best=-1e30f; int bi=-1;
    for(int i=tid;i<nb;i+=blockDim.x){float v=bmv[i];if(v>best){best=v;bi=bmi[i];}}
    __shared__ float sv[256]; __shared__ int si[256];
    sv[tid]=best;si[tid]=bi;__syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){if(tid<s&&sv[tid+s]>sv[tid]){sv[tid]=sv[tid+s];si[tid]=si[tid+s];}__syncthreads();}
    if(tid==0)*out=si[0];
}

// ===== cuBLAS bf16 GEMM =====
static void cublas_bf16_gemm(cublasHandle_t h,
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
    int S, int N, int K) {
    float alpha = 1.0f, beta_val = 0.0f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, S, K,
        &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K,
        &beta_val, C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ===== Main orchestrator =====
extern "C" void launch_prefill_bf16(
    const int *token_ids, int seq_len, int *output_token,
    const __nv_bfloat16 *embed_weight, const PFLayerWeights *layers,
    const __nv_bfloat16 *final_norm_w, const __nv_bfloat16 *lm_head_w,
    __nv_bfloat16 *fa_k_cache, __nv_bfloat16 *fa_v_cache,
    float *dn_states, float *conv_bufs,
    // Scratch (ALL bf16 except state/conv which are f32)
    __nv_bfloat16 *hidden, __nv_bfloat16 *residual, __nv_bfloat16 *normalized,
    __nv_bfloat16 *proj_buf, __nv_bfloat16 *proj_buf2,
    __nv_bfloat16 *attn_buf, __nv_bfloat16 *mlp_buf,
    __nv_bfloat16 *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    __nv_bfloat16 *final_normed, __nv_bfloat16 *hidden_bf16_out,
    float *lm_bmv, int *lm_bmi,
    cudaStream_t stream)
{
    static cublasHandle_t cublas = nullptr;
    if (!cublas) cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    PFLayerWeights hl[NUM_LAYERS];
    cudaMemcpy(hl, layers, NUM_LAYERS*sizeof(PFLayerWeights), cudaMemcpyDeviceToHost);

    int S = seq_len;
    int bk = (S*HIDDEN+255)/256;

    pf_embed<<<bk, 256, 0, stream>>>(token_ids, embed_weight, hidden, S);

    int fa_stride = FA_KV_HEADS * 2048 * FA_HEAD_DIM;
    int dn_stride = DN_V_HEADS * DN_KEY * DN_VAL;
    int fa_idx = 0, dn_idx = 0;

    for (int li = 0; li < NUM_LAYERS; li++) {
        const PFLayerWeights &lw = hl[li];
        int lt = LAYER_TYPE[li];

        const __nv_bfloat16 *norm_w = (const __nv_bfloat16 *)lw.ptrs[0];
        pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, norm_w, normalized, residual, S, HIDDEN);

        if (lt == 0) {
            // DeltaNet
            const __nv_bfloat16 *qkv_w=(const __nv_bfloat16*)lw.ptrs[1];
            const __nv_bfloat16 *z_w=(const __nv_bfloat16*)lw.ptrs[2];
            const __nv_bfloat16 *beta_w=(const __nv_bfloat16*)lw.ptrs[3];
            const __nv_bfloat16 *alpha_w=(const __nv_bfloat16*)lw.ptrs[4];
            const __nv_bfloat16 *conv_w=(const __nv_bfloat16*)lw.ptrs[5];
            const __nv_bfloat16 *a_log=(const __nv_bfloat16*)lw.ptrs[6];
            const __nv_bfloat16 *dt_bias=(const __nv_bfloat16*)lw.ptrs[7];
            const __nv_bfloat16 *dn_norm=(const __nv_bfloat16*)lw.ptrs[8];
            const __nv_bfloat16 *out_w=(const __nv_bfloat16*)lw.ptrs[9];
            const __nv_bfloat16 *post_norm=(const __nv_bfloat16*)lw.ptrs[10];
            const __nv_bfloat16 *gate_w=(const __nv_bfloat16*)lw.ptrs[11];
            const __nv_bfloat16 *up_w=(const __nv_bfloat16*)lw.ptrs[12];
            const __nv_bfloat16 *down_w=(const __nv_bfloat16*)lw.ptrs[13];

            // cuBLAS projections — direct bf16, no conversion!
            cublas_bf16_gemm(cublas, normalized, qkv_w, proj_buf, S, DN_CONV_CH, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, z_w, proj_buf2, S, DN_V_SIZE, HIDDEN);
            pf_bf16_matvec<<<S*DN_V_HEADS, 32, 0, stream>>>(normalized, beta_w, beta_buf, S, HIDDEN, DN_V_HEADS);
            pf_bf16_matvec<<<S*DN_V_HEADS, 32, 0, stream>>>(normalized, alpha_w, alpha_buf, S, HIDDEN, DN_V_HEADS);

            // Standalone recurrence
            pf_deltanet_recurrence<<<DN_K_HEADS, 512, 0, stream>>>(
                proj_buf, proj_buf2, beta_buf, alpha_buf,
                conv_w, a_log, dt_bias, dn_norm,
                dn_states + dn_idx*dn_stride,
                conv_bufs + dn_idx*DN_CONV_CH*DN_CONV_K,
                dn_out_buf, S);

            // Out projection + residual
            cublas_bf16_gemm(cublas, dn_out_buf, out_w, proj_buf, S, HIDDEN, DN_V_SIZE);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            // MLP
            pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, post_norm, normalized, residual, S, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, gate_w, proj_buf, S, INTER, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, up_w, proj_buf2, S, INTER, HIDDEN);
            int mlp_bk = (S*INTER+255)/256;
            pf_silu_mul_bf16<<<mlp_bk, 256, 0, stream>>>(proj_buf, proj_buf2, mlp_buf, S*INTER);
            cublas_bf16_gemm(cublas, mlp_buf, down_w, proj_buf, S, HIDDEN, INTER);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            dn_idx++;
        } else {
            // Full Attention
            const __nv_bfloat16 *q_w=(const __nv_bfloat16*)lw.ptrs[1];
            const __nv_bfloat16 *k_w=(const __nv_bfloat16*)lw.ptrs[2];
            const __nv_bfloat16 *v_w=(const __nv_bfloat16*)lw.ptrs[3];
            const __nv_bfloat16 *q_nw=(const __nv_bfloat16*)lw.ptrs[4];
            const __nv_bfloat16 *k_nw=(const __nv_bfloat16*)lw.ptrs[5];
            const __nv_bfloat16 *o_w=(const __nv_bfloat16*)lw.ptrs[6];
            const __nv_bfloat16 *post_norm=(const __nv_bfloat16*)lw.ptrs[7];
            const __nv_bfloat16 *gate_w=(const __nv_bfloat16*)lw.ptrs[8];
            const __nv_bfloat16 *up_w=(const __nv_bfloat16*)lw.ptrs[9];
            const __nv_bfloat16 *down_w=(const __nv_bfloat16*)lw.ptrs[10];

            cublas_bf16_gemm(cublas, normalized, q_w, proj_buf, S, FA_QPROJ_SIZE, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, k_w, proj_buf2, S, FA_KV_SIZE, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, v_w, attn_buf, S, FA_KV_SIZE, HIDDEN);

            int total_heads = S*(FA_Q_HEADS+FA_KV_HEADS);
            pf_qk_norm_rope<<<(total_heads+15)/16, 512, 0, stream>>>(
                proj_buf, proj_buf2, attn_buf, q_nw, k_nw,
                fa_k_cache + fa_idx*fa_stride, fa_v_cache + fa_idx*fa_stride, S, 2048);

            pf_causal_attn<<<(S*FA_Q_HEADS+15)/16, 512, 0, stream>>>(
                proj_buf, proj_buf2, attn_buf, dn_out_buf, S);

            cublas_bf16_gemm(cublas, dn_out_buf, o_w, proj_buf, S, HIDDEN, FA_Q_SIZE);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            // MLP
            pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, post_norm, normalized, residual, S, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, gate_w, proj_buf, S, INTER, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, up_w, proj_buf2, S, INTER, HIDDEN);
            int mlp_bk = (S*INTER+255)/256;
            pf_silu_mul_bf16<<<mlp_bk, 256, 0, stream>>>(proj_buf, proj_buf2, mlp_buf, S*INTER);
            cublas_bf16_gemm(cublas, mlp_buf, down_w, proj_buf, S, HIDDEN, INTER);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            fa_idx++;
        }
    }

    pf_final_norm<<<1, 512, 0, stream>>>(hidden, final_norm_w, final_normed, hidden_bf16_out, S);

    int lm_blocks = 512;
    pf_lm_head<<<lm_blocks, 256, 0, stream>>>(final_normed, lm_head_w, lm_bmv, lm_bmi, VOCAB);
    pf_lm_reduce<<<1, 256, 0, stream>>>(lm_bmv, lm_bmi, output_token, lm_blocks);
}

// Scratch: bf16 row-major [out_dim, in_dim] for one linear (max ~50M elems for this model).
extern "C" void launch_prefill_w4(
    const int *token_ids, int seq_len, int *output_token,
    const __nv_bfloat16 *embed_weight, const PFLayerWeightsW4 *layers,
    const __nv_bfloat16 *final_norm_w, const __nv_bfloat16 *lm_head_w,
    __nv_bfloat16 *fa_k_cache, __nv_bfloat16 *fa_v_cache,
    float *dn_states, float *conv_bufs,
    __nv_bfloat16 *hidden, __nv_bfloat16 *residual, __nv_bfloat16 *normalized,
    __nv_bfloat16 *proj_buf, __nv_bfloat16 *proj_buf2,
    __nv_bfloat16 *attn_buf, __nv_bfloat16 *mlp_buf,
    __nv_bfloat16 *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    __nv_bfloat16 *final_normed, __nv_bfloat16 *hidden_bf16_out,
    float *lm_bmv, int *lm_bmi,
    __nv_bfloat16 *w_dequant_scratch,
    cudaStream_t stream)
{
    static cublasHandle_t cublas = nullptr;
    if (!cublas) cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    PFLayerWeightsW4 hl[NUM_LAYERS];
    cudaMemcpy(hl, layers, NUM_LAYERS * sizeof(PFLayerWeightsW4), cudaMemcpyDeviceToHost);

    int S = seq_len;
    int bk = (S*HIDDEN+255)/256;
    pf_embed<<<bk, 256, 0, stream>>>(token_ids, embed_weight, hidden, S);

    int fa_stride = FA_KV_HEADS * 2048 * FA_HEAD_DIM;
    int dn_stride = DN_V_HEADS * DN_KEY * DN_VAL;
    int fa_idx = 0, dn_idx = 0;

    auto deq_gemm = [&](const PFQ4 &q, int in_dim, int out_dim,
                        const __nv_bfloat16 *A, __nv_bfloat16 *C) {
        launch_dequant_linear(
            (const int *)q.qweight, (const __half *)q.scales, (const int *)q.qzeros,
            w_dequant_scratch, in_dim, out_dim, stream);
        cublas_bf16_gemm(cublas, A, w_dequant_scratch, C, S, out_dim, in_dim);
    };

    for (int li = 0; li < NUM_LAYERS; li++) {
        const PFLayerWeightsW4 &lw = hl[li];
        int lt = LAYER_TYPE[li];

        if (lt == 0) {
            const PFFD_W4 &d = lw.dn;
            const __nv_bfloat16 *norm_w = (const __nv_bfloat16 *)d.input_ln;
            pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, norm_w, normalized, residual, S, HIDDEN);

            deq_gemm(d.qkv, HIDDEN, DN_CONV_CH, normalized, proj_buf);
            deq_gemm(d.z, HIDDEN, DN_V_SIZE, normalized, proj_buf2);
            launch_dequant_linear(
                (const int *)d.beta.qweight, (const __half *)d.beta.scales, (const int *)d.beta.qzeros,
                w_dequant_scratch, HIDDEN, DN_V_HEADS, stream);
            cublas_bf16_gemm(cublas, normalized, w_dequant_scratch, attn_buf, S, DN_V_HEADS, HIDDEN);
            pf_bf16_to_float<<<(S * DN_V_HEADS + 255) / 256, 256, 0, stream>>>(
                attn_buf, beta_buf, S * DN_V_HEADS);
            launch_dequant_linear(
                (const int *)d.alpha.qweight, (const __half *)d.alpha.scales, (const int *)d.alpha.qzeros,
                w_dequant_scratch, HIDDEN, DN_V_HEADS, stream);
            cublas_bf16_gemm(cublas, normalized, w_dequant_scratch, attn_buf, S, DN_V_HEADS, HIDDEN);
            pf_bf16_to_float<<<(S * DN_V_HEADS + 255) / 256, 256, 0, stream>>>(
                attn_buf, alpha_buf, S * DN_V_HEADS);

            const __nv_bfloat16 *conv_w = (const __nv_bfloat16 *)d.conv_w;
            const __nv_bfloat16 *a_log = (const __nv_bfloat16 *)d.a_log;
            const __nv_bfloat16 *dt_bias = (const __nv_bfloat16 *)d.dt_bias;
            const __nv_bfloat16 *dn_norm = (const __nv_bfloat16 *)d.norm_w;

            pf_deltanet_recurrence<<<DN_K_HEADS, 512, 0, stream>>>(
                proj_buf, proj_buf2, beta_buf, alpha_buf,
                conv_w, a_log, dt_bias, dn_norm,
                dn_states + dn_idx * dn_stride,
                conv_bufs + dn_idx * DN_CONV_CH * DN_CONV_K,
                dn_out_buf, S);

            deq_gemm(d.out_proj, DN_V_SIZE, HIDDEN, dn_out_buf, proj_buf);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            const __nv_bfloat16 *post_norm = (const __nv_bfloat16 *)d.post_ln;
            pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, post_norm, normalized, residual, S, HIDDEN);
            deq_gemm(d.gate, HIDDEN, INTER, normalized, proj_buf);
            deq_gemm(d.up, HIDDEN, INTER, normalized, proj_buf2);
            int mlp_bk = (S*INTER+255)/256;
            pf_silu_mul_bf16<<<mlp_bk, 256, 0, stream>>>(proj_buf, proj_buf2, mlp_buf, S*INTER);
            deq_gemm(d.down, INTER, HIDDEN, mlp_buf, proj_buf);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            dn_idx++;
        } else {
            const PFFA_W4 &fa = lw.fa;
            const __nv_bfloat16 *norm_w = (const __nv_bfloat16 *)fa.input_ln;
            pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, norm_w, normalized, residual, S, HIDDEN);

            deq_gemm(fa.q_proj, HIDDEN, FA_QPROJ_SIZE, normalized, proj_buf);
            deq_gemm(fa.k_proj, HIDDEN, FA_KV_SIZE, normalized, proj_buf2);
            deq_gemm(fa.v_proj, HIDDEN, FA_KV_SIZE, normalized, attn_buf);

            const __nv_bfloat16 *q_nw = (const __nv_bfloat16 *)fa.q_norm;
            const __nv_bfloat16 *k_nw = (const __nv_bfloat16 *)fa.k_norm;

            int total_heads = S*(FA_Q_HEADS+FA_KV_HEADS);
            pf_qk_norm_rope<<<(total_heads+15)/16, 512, 0, stream>>>(
                proj_buf, proj_buf2, attn_buf, q_nw, k_nw,
                fa_k_cache + fa_idx*fa_stride, fa_v_cache + fa_idx*fa_stride, S, 2048);

            pf_causal_attn<<<(S*FA_Q_HEADS+15)/16, 512, 0, stream>>>(
                proj_buf, proj_buf2, attn_buf, dn_out_buf, S);

            deq_gemm(fa.o_proj, FA_Q_SIZE, HIDDEN, dn_out_buf, proj_buf);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            const __nv_bfloat16 *post_norm = (const __nv_bfloat16 *)fa.post_ln;
            pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, post_norm, normalized, residual, S, HIDDEN);
            deq_gemm(fa.gate, HIDDEN, INTER, normalized, proj_buf);
            deq_gemm(fa.up, HIDDEN, INTER, normalized, proj_buf2);
            int mlp_bk = (S*INTER+255)/256;
            pf_silu_mul_bf16<<<mlp_bk, 256, 0, stream>>>(proj_buf, proj_buf2, mlp_buf, S*INTER);
            deq_gemm(fa.down, INTER, HIDDEN, mlp_buf, proj_buf);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            fa_idx++;
        }
    }

    pf_final_norm<<<1, 512, 0, stream>>>(hidden, final_norm_w, final_normed, hidden_bf16_out, S);

    int lm_blocks = 512;
    pf_lm_head<<<lm_blocks, 256, 0, stream>>>(final_normed, lm_head_w, lm_bmv, lm_bmi, VOCAB);
    pf_lm_reduce<<<1, 256, 0, stream>>>(lm_bmv, lm_bmi, output_token, lm_blocks);
}
