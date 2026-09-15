#include "attention.cuh"
#include "tiny_llm/cuda_utils.h"
#include "warp_utils.cuh"
#include <cfloat>

namespace tiny_llm {
namespace kernels {

// ============================================================================
// Online-softmax attention kernels
//
// Shared-memory usage is O(tile) instead of O(sequence length).  This fixes a
// correctness bug for long sequences (dynamic shared memory used to overflow
// 48 KB when visible_len exceeded ~12k) and is numerically identical to the
// naive two-pass softmax up to fp32 rounding.
//
// Decode: one block per query head, Q held in shared memory, K/V streamed in
//         tiles of ATTEND_TILE positions.  Running max / running sum / partial
//         output are updated tile-by-tile with flash-attention rescaling.
// Prefill: one block per (query_pos, query_head), same online-softmax scheme
//          with causal masking applied per tile.
//
// Layout (token-major):
//   Q:       [S, Hq,  D]   q(s,h,d)     = (s*Hq  + h )*D + d
//   K_cache: [T, Hkv, D]   k(t,kh,d)    = (t*Hkv + kh)*D + d
//   V_cache: [T, Hkv, D]   v(t,kh,d)    = (t*Hkv + kh)*D + d
//   O:       [S, Hq,  D]   o(s,h,d)     = (s*Hq  + h )*D + d
//
// GQA: kv_head = q_head / group_size,  group_size = Hq / Hkv
// ============================================================================

// Score tile size.  128 positions x 4 bytes = 512 B shared memory per block;
// the same block also keeps the query head (fp16) and partial output (fp32).
constexpr int ATTEND_TILE = 128;

// Dynamic shared memory layout for the tiled attention kernels:
//   scores[ATTEND_TILE]       fp32 attention scores of the current tile
//   red[8]                    block-reduction scratch (1 float per warp)
//   out_acc[head_dim]         partial attention output (un-normalized)
//   q_smem[head_dim]          query head cache (fp16)
struct AttentionSmemLayout {
    int                               head_dim;
    __host__ __device__ constexpr int scores_offset() const { return 0; }
    __host__ __device__ constexpr int red_offset() const { return ATTEND_TILE; }
    __host__ __device__ constexpr int out_acc_offset() const { return ATTEND_TILE + 8; }
    __host__ __device__ constexpr int q_offset_bytes() const {
        return (ATTEND_TILE + 8 + head_dim) * sizeof(float);
    }
    __host__ __device__ constexpr int total_bytes() const {
        return q_offset_bytes() + head_dim * sizeof(half);
    }
};

__device__ __forceinline__ float block_reduce_max_dyn(float val, float *red, int nthreads) {
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) red[wid] = val;
    __syncthreads();
    int nwarps = (nthreads + 31) >> 5;
    if (wid == 0) {
        val = (lane < nwarps) ? red[lane] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane == 0) red[0] = val;
    }
    __syncthreads();
    val = red[0];
    return val;
}

__device__ __forceinline__ float block_reduce_sum_dyn(float val, float *red, int nthreads) {
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) red[wid] = val;
    __syncthreads();
    int nwarps = (nthreads + 31) >> 5;
    if (wid == 0) {
        val = (lane < nwarps) ? red[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) red[0] = val;
    }
    __syncthreads();
    val = red[0];
    return val;
}

// ── Decode tile loop 的取址策略（TLLM-P0-004）──────────────────────────────
//
// decode 的 online softmax 与 K/V 的存放方式无关，只差"逻辑位置 → 行指针"这一步。
// 把它抽成策略类型后，连续 KV 与分页 KV 共用**同一份**归约循环：
//   - 避免在线 softmax 出现两份实现而数值漂移；
//   - 使 direct 与 legacy 的差分退化为纯寻址测试。
//
// 共享 vs 复制的取舍见 issue #8 与设计包 §10.1：抽取会给 attention_decode 带来已测量
// 的 +1.4~2.3% kernel 回归；但复制方案唯一的风险（两份实现漂移）已由
// tests/test_paged_direct.cpp 的"逐位相同"门禁自动覆盖，故选择共享。
//
// 关键设计：**无效行返回零行，而不是 nullptr**。共享循环里因此没有任何有效性分支，
// 累加表达式与抽取前逐字相同；且语义与既有 paged_gather_blocks 的"非法块 id 写 0"
// 完全一致（零 K 行点积为 0 → score = 0；零 V 行贡献为 0），所以 direct 与 legacy 的
// 逐元素相等是**构造性**成立的，而不是靠容差。
//
// 策略契约：
//   static constexpr bool kNeedsTilePrep
//       true  → begin_tile 会协作填充行缓存，共享循环随后插入一次 __syncthreads()
//               （编译期决定，连续路径不会多出屏障）；
//       false → begin_tile 为空操作。
//   void begin_tile(int tile_start, int tile_size, int tid, int nthreads)
//   const half *k_row(int pos)   // 逻辑位置 → K 行首；无效行返回零行（非空）
//   const half *v_row(int pos)   // 逻辑位置 → V 行首；无效行返回零行（非空）
namespace {

// 连续 KV：K/V 为 [T, Hkv, D]，按 token 步长线性寻址。行永远有效。
struct ContiguousRows {
    static constexpr bool kNeedsTilePrep = false;

    const half *k;
    const half *v;
    int         stride;

    __device__ __forceinline__ void        begin_tile(int, int, int, int) {}
    __device__ __forceinline__ const half *k_row(int pos) { return k + pos * stride; }
    __device__ __forceinline__ const half *v_row(int pos) { return v + pos * stride; }
};

// 共享的 decode online-softmax 循环。累加顺序、__expf 与 final normalize 与抽取前
// 逐字相同。
template <typename Rows>
__device__ __forceinline__ void decode_online_softmax(const half *q_smem, half *output,
                                                      float *scores, float *red, float *out_acc,
                                                      Rows rows, int visible_len, int head_dim,
                                                      float scale, int tid, int nthreads) {
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int tile_start = 0; tile_start < visible_len; tile_start += ATTEND_TILE) {
        const int tile_size = min(ATTEND_TILE, visible_len - tile_start);
        rows.begin_tile(tile_start, tile_size, tid, nthreads);
        if (Rows::kNeedsTilePrep) __syncthreads();

        // Step 1: scores s_i = scale * (Q dot K_i) for this tile.
        for (int i = tid; i < tile_size; i += nthreads) {
            const half *k_pos = rows.k_row(tile_start + i);
            float       score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += __half2float(q_smem[d]) * __half2float(k_pos[d]);
            }
            scores[i] = score * scale;
        }
        __syncthreads();

        // Step 2: tile max.
        float m_tile = -FLT_MAX;
        for (int i = tid; i < tile_size; i += nthreads) {
            m_tile = fmaxf(m_tile, scores[i]);
        }
        m_tile = block_reduce_max_dyn(m_tile, red, nthreads);

        // Step 3: online rescale.
        const float m_new = fmaxf(running_max, m_tile);
        const float old_rescale = __expf(running_max - m_new);

        float sum_tile = 0.0f;
        for (int i = tid; i < tile_size; i += nthreads) {
            sum_tile += __expf(scores[i] - m_new);
        }
        sum_tile = block_reduce_sum_dyn(sum_tile, red, nthreads);

        running_sum = running_sum * old_rescale + sum_tile;

        // Step 4: update partial output.  Each thread owns a subset of head_dim
        // dimensions and scans all positions in the tile.
        for (int d = tid; d < head_dim; d += nthreads) {
            float partial = 0.0f;
            for (int i = 0; i < tile_size; ++i) {
                const half *v_pos = rows.v_row(tile_start + i);
                partial += __expf(scores[i] - m_new) * __half2float(v_pos[d]);
            }
            out_acc[d] = out_acc[d] * old_rescale + partial;
        }

        running_max = m_new;
        __syncthreads();
    }

    // Final normalize.
    const float inv_sum = 1.0f / (running_sum + 1e-9f);
    for (int d = tid; d < head_dim; d += nthreads) {
        output[d] = __float2half(out_acc[d] * inv_sum);
    }
}

} // namespace

// Decode attention: single query token against cached K/V.
// Q: [1, Hq, D].  One block per q_head.
// visible_len is read from global memory so the kernel is CUDA-Graph
// replayable (task 3.1): the host updates one device int then replays.
__global__ void attention_decode_kernel(const half *__restrict__ query,
                                        const half *__restrict__ k_cache,
                                        const half *__restrict__ v_cache, half *__restrict__ output,
                                        float scale, int num_q_heads, int num_kv_heads,
                                        const int *device_visible_len, int head_dim) {
    const int visible_len = *device_visible_len;
    const int q_head = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int group_size = num_q_heads / num_kv_heads;
    const int kv_head = q_head / group_size;
    const int kv_stride = num_kv_heads * head_dim;

    const half *k = k_cache + kv_head * head_dim;
    const half *v = v_cache + kv_head * head_dim;
    half       *o = output + q_head * head_dim;

    extern __shared__ float smem[];
    AttentionSmemLayout     layout{head_dim};
    float                  *scores = smem + layout.scores_offset();
    float                  *red = smem + layout.red_offset();
    float                  *out_acc = smem + layout.out_acc_offset();
    half                   *q_smem =
        reinterpret_cast<half *>(reinterpret_cast<char *>(smem) + layout.q_offset_bytes());

    // Cache Q for this head and zero the output accumulator.
    for (int d = tid; d < head_dim; d += nthreads) {
        q_smem[d] = query[q_head * head_dim + d];
        out_acc[d] = 0.0f;
    }
    __syncthreads();

    ContiguousRows rows{k, v, kv_stride};
    decode_online_softmax(q_smem, o, scores, red, out_acc, rows, visible_len, head_dim, scale, tid,
                          nthreads);
}

void attention_decode(const half *query, const half *k_cache, const half *v_cache, half *output,
                      float scale, int num_q_heads, int num_kv_heads, const int *device_visible_len,
                      int head_dim, cudaStream_t stream) {
    if (num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || device_visible_len == nullptr) {
        return;
    }

    const int           num_blocks = num_q_heads;
    const int           block_size = 128;
    AttentionSmemLayout layout{head_dim};
    const size_t        shared_size = layout.total_bytes();

    attention_decode_kernel<<<num_blocks, block_size, shared_size, stream>>>(
        query, k_cache, v_cache, output, scale, num_q_heads, num_kv_heads, device_visible_len,
        head_dim);
}

// 旧签名薄封装：把 host 端 int 复制到 device 后转发。仅测试/兼容用，
// 每次调用复用同一块 device 缓冲（不适用于并发多线程，生产路径应直接用
// device 指针版本）。
void attention_decode(const half *query, const half *k_cache, const half *v_cache, half *output,
                      float scale, int num_q_heads, int num_kv_heads, int visible_len, int head_dim,
                      cudaStream_t stream) {
    // 函数级 thread_local：避免测试场景每次调用都 cudaMalloc。CUDA 上下文在
    // 首次调用前已初始化。用 thread_local 而非 plain static：多线程并发调用
    // 旧签名封装时会竞争同一 device 缓冲。
    static thread_local DeviceBuffer<int> device_len(1);
    device_len.copyFromHost(&visible_len, 1, stream);
    attention_decode(query, k_cache, v_cache, output, scale, num_q_heads, num_kv_heads,
                     device_len.data(), head_dim, stream);
}

// ============================================================================
// Direct paged decode attention（TLLM-P0-004）
//
// 与 attention_decode_kernel 的唯一区别是 K/V 取址：直接从物理 pool + block table
// 寻址，不再把可见窗口 gather 成连续 scratch。归约循环与连续版**共用**同一份
// decode_online_softmax（取舍见 issue #8 与设计包 §10.1）。
//
// 无效行（越界块 id、b >= table_len）返回共享内存里的零行，因此循环内无分支，且语义
// 与 paged_gather_blocks 写 0 完全一致。
// ============================================================================

// 分页 KV 取址策略。每 tile 一次性把 tile 内所有逻辑位置映射到 pool 行内元素偏移，
// 避免在 step 4 的 (head_dim × tile_size) 内层循环里反复做整数除法与块表读取。
struct PagedRows {
    static constexpr bool kNeedsTilePrep = true;

    const half *k_base;
    const half *v_base;
    const half *zero_row; // 共享内存零行，长度 >= head_dim
    const int  *block_table;
    int         block_size;
    int         max_num_blocks;
    int         table_len;
    int         kv_dim;
    int        *row_elem; // 共享内存 [ATTEND_TILE]：行内元素偏移；-1 = 无效
    int         tile_start;

    __device__ __forceinline__ void begin_tile(int start, int tile_size, int tid, int nthreads) {
        tile_start = start;
        for (int i = tid; i < tile_size; i += nthreads) {
            const int pos = start + i;
            const int b = pos / block_size;
            const int r = pos - b * block_size;
            const int p = (b < table_len) ? block_table[b] : -1;
            row_elem[i] =
                (p < 0 || p >= max_num_blocks)
                    ? -1
                    : static_cast<int>(
                          (static_cast<size_t>(p) * block_size + static_cast<size_t>(r)) * kv_dim);
        }
    }

    __device__ __forceinline__ const half *k_row(int pos) {
        const int e = row_elem[pos - tile_start];
        return (e < 0) ? zero_row : k_base + e;
    }
    __device__ __forceinline__ const half *v_row(int pos) {
        const int e = row_elem[pos - tile_start];
        return (e < 0) ? zero_row : v_base + e;
    }
};

__global__ void attention_decode_paged_kernel(
    const half *__restrict__ query, const half *__restrict__ k_pool_layer,
    const half *__restrict__ v_pool_layer, const int *__restrict__ block_table,
    half *__restrict__ output, float scale, int num_q_heads, int num_kv_heads, int head_dim,
    const int *device_visible_tokens, int block_size, int max_num_blocks, int table_len) {
    const int visible_tokens = *device_visible_tokens;
    const int q_head = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int group_size = num_q_heads / num_kv_heads;
    const int kv_head = q_head / group_size;
    const int kv_dim = num_kv_heads * head_dim;

    // 本 head 在 pool 内的基址（layer 偏移已由 caller 加在指针上）
    const half *k_base = k_pool_layer + kv_head * head_dim;
    const half *v_base = v_pool_layer + kv_head * head_dim;
    half       *o = output + q_head * head_dim;

    extern __shared__ float smem[];
    AttentionSmemLayout     layout{head_dim};
    float                  *scores = smem + layout.scores_offset();
    float                  *red = smem + layout.red_offset();
    float                  *out_acc = smem + layout.out_acc_offset();
    half                   *q_smem =
        reinterpret_cast<half *>(reinterpret_cast<char *>(smem) + layout.q_offset_bytes());

    // paged 专用共享内存（layout 之后）：
    //   row_elem[ATTEND_TILE]  本 tile 每个逻辑 token 的行内元素偏移；-1 表示无效
    //   zero_row[head_dim]     无效行使用的零行
    char *tail = reinterpret_cast<char *>(smem) + layout.total_bytes();
    int  *row_elem = reinterpret_cast<int *>(tail);
    half *zero_row = reinterpret_cast<half *>(tail + ATTEND_TILE * sizeof(int));

    // Cache Q / zero the accumulator / 初始化零行
    for (int d = tid; d < head_dim; d += nthreads) {
        q_smem[d] = query[q_head * head_dim + d];
        out_acc[d] = 0.0f;
        zero_row[d] = __float2half(0.0f);
    }
    __syncthreads();

    PagedRows rows;
    rows.k_base = k_base;
    rows.v_base = v_base;
    rows.zero_row = zero_row;
    rows.block_table = block_table;
    rows.block_size = block_size;
    rows.max_num_blocks = max_num_blocks;
    rows.table_len = table_len;
    rows.kv_dim = kv_dim;
    rows.row_elem = row_elem;
    rows.tile_start = 0;

    decode_online_softmax(q_smem, o, scores, red, out_acc, rows, visible_tokens, head_dim, scale,
                          tid, nthreads);
}

void attention_decode_paged(const half *query, const half *k_pool_layer, const half *v_pool_layer,
                            const int *block_table, half *output, float scale, int num_q_heads,
                            int num_kv_heads, int head_dim, const int *device_visible_tokens,
                            int block_size, int max_num_blocks, int table_len,
                            cudaStream_t stream) {
    // 防御性检查（与文件内其他 kernel 一致）；真正的错误契约在 host 侧
    // TransformerLayer::forwardPaged，参数非法不应依赖这里的静默 no-op。
    if (query == nullptr || k_pool_layer == nullptr || v_pool_layer == nullptr ||
        block_table == nullptr || output == nullptr || device_visible_tokens == nullptr) {
        return;
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || block_size <= 0 ||
        max_num_blocks <= 0) {
        return;
    }

    const int           num_blocks = num_q_heads;
    const int           threads = 128;
    AttentionSmemLayout layout{head_dim};
    const size_t        shared_size = layout.total_bytes() + ATTEND_TILE * sizeof(int) +
                               static_cast<size_t>(head_dim) * sizeof(half);

    attention_decode_paged_kernel<<<num_blocks, threads, shared_size, stream>>>(
        query, k_pool_layer, v_pool_layer, block_table, output, scale, num_q_heads, num_kv_heads,
        head_dim, device_visible_tokens, block_size, max_num_blocks, table_len);
}

// Prefill attention: full sequence with causal masking.
// Q: [S, Hq, D]; K/V: [S, Hkv, D].  One block per (query_pos, q_head).
__global__ void attention_prefill_kernel(const half *__restrict__ query,
                                         const half *__restrict__ key,
                                         const half *__restrict__ value, half *__restrict__ output,
                                         float scale, int num_q_heads, int num_kv_heads,
                                         int seq_len, int head_dim) {
    const int query_pos = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int group_size = num_q_heads / num_kv_heads;
    const int kv_head = q_head / group_size;
    const int kv_stride = num_kv_heads * head_dim;

    const half *q = query + (query_pos * num_q_heads + q_head) * head_dim;
    const half *k = key + kv_head * head_dim;
    const half *v = value + kv_head * head_dim;
    half       *o = output + (query_pos * num_q_heads + q_head) * head_dim;

    extern __shared__ float smem[];
    AttentionSmemLayout     layout{head_dim};
    float                  *scores = smem + layout.scores_offset();
    float                  *red = smem + layout.red_offset();
    float                  *out_acc = smem + layout.out_acc_offset();
    half                   *q_smem =
        reinterpret_cast<half *>(reinterpret_cast<char *>(smem) + layout.q_offset_bytes());

    for (int d = tid; d < head_dim; d += nthreads) {
        q_smem[d] = q[d];
        out_acc[d] = 0.0f;
    }
    __syncthreads();

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    // Causal attention: key positions [0, query_pos] only.
    for (int tile_start = 0; tile_start <= query_pos; tile_start += ATTEND_TILE) {
        const int tile_end = min(tile_start + ATTEND_TILE, min(query_pos + 1, seq_len));
        const int tile_size = tile_end - tile_start;
        if (tile_size <= 0) break;

        for (int i = tid; i < tile_size; i += nthreads) {
            const int   pos = tile_start + i;
            const half *k_pos = k + pos * kv_stride;
            float       score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += __half2float(q_smem[d]) * __half2float(k_pos[d]);
            }
            scores[i] = score * scale;
        }
        __syncthreads();

        float m_tile = -FLT_MAX;
        for (int i = tid; i < tile_size; i += nthreads) {
            m_tile = fmaxf(m_tile, scores[i]);
        }
        m_tile = block_reduce_max_dyn(m_tile, red, nthreads);

        const float m_new = fmaxf(running_max, m_tile);
        const float old_rescale = __expf(running_max - m_new);

        float sum_tile = 0.0f;
        for (int i = tid; i < tile_size; i += nthreads) {
            sum_tile += __expf(scores[i] - m_new);
        }
        sum_tile = block_reduce_sum_dyn(sum_tile, red, nthreads);

        running_sum = running_sum * old_rescale + sum_tile;

        for (int d = tid; d < head_dim; d += nthreads) {
            float partial = 0.0f;
            for (int i = 0; i < tile_size; ++i) {
                const int pos = tile_start + i;
                partial += __expf(scores[i] - m_new) * __half2float(v[pos * kv_stride + d]);
            }
            out_acc[d] = out_acc[d] * old_rescale + partial;
        }

        running_max = m_new;
        __syncthreads();
    }

    const float inv_sum = 1.0f / (running_sum + 1e-9f);
    for (int d = tid; d < head_dim; d += nthreads) {
        o[d] = __float2half(out_acc[d] * inv_sum);
    }
}

void attention_prefill(const half *query, const half *key, const half *value, half *output,
                       float scale, int num_q_heads, int num_kv_heads, int seq_len, int head_dim,
                       cudaStream_t stream) {
    if (num_q_heads <= 0 || num_kv_heads <= 0 || seq_len <= 0 || head_dim <= 0) {
        return;
    }

    dim3                grid(seq_len, num_q_heads);
    const int           block_size = 128;
    AttentionSmemLayout layout{head_dim};
    const size_t        shared_size = layout.total_bytes();

    attention_prefill_kernel<<<grid, block_size, shared_size, stream>>>(
        query, key, value, output, scale, num_q_heads, num_kv_heads, seq_len, head_dim);
}

// Get attention weights (for testing causal mask)
// Q: [query_len, Hq,  D]
// K: [key_len,   Hkv, D]
// weights: [query_len, Hq, key_len]  (token-major)
__global__ void get_attention_weights_kernel(const half *__restrict__ query,
                                             const half *__restrict__ key,
                                             half *__restrict__ weights, float scale,
                                             int num_q_heads, int num_kv_heads, int query_len,
                                             int key_len, int head_dim, bool apply_causal_mask) {
    int query_pos = blockIdx.x;
    int q_head = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;

    int kv_stride = num_kv_heads * head_dim;

    const half *q = query + (query_pos * num_q_heads + q_head) * head_dim;
    const half *k = key + kv_head * head_dim;
    half       *w = weights + (query_pos * num_q_heads + q_head) * key_len;

    for (int key_pos = tid; key_pos < key_len; key_pos += block_size) {
        if (apply_causal_mask && key_pos > query_pos) {
            w[key_pos] = __float2half(0.0f);
        } else {
            float       score = 0.0f;
            const half *k_pos = k + key_pos * kv_stride;
            for (int d = 0; d < head_dim; ++d) {
                score += __half2float(q[d]) * __half2float(k_pos[d]);
            }
            w[key_pos] = __float2half(score * scale);
        }
    }
}

void get_attention_weights(const half *query, const half *key, half *__restrict__ weights,
                           float scale, int num_q_heads, int num_kv_heads, int query_len,
                           int key_len, int head_dim, bool apply_causal_mask, cudaStream_t stream) {
    if (num_q_heads <= 0 || num_kv_heads <= 0 || query_len <= 0 || key_len <= 0 || head_dim <= 0) {
        return;
    }

    dim3 grid(query_len, num_q_heads);
    int  block_size = 256;

    get_attention_weights_kernel<<<grid, block_size, 0, stream>>>(
        query, key, weights, scale, num_q_heads, num_kv_heads, query_len, key_len, head_dim,
        apply_causal_mask);
}

// Simple softmax kernel
// 修复：共享内存 O(1)。旧实现把每个元素的 exp 值缓存在动态共享内存里
// （(seq_len+32)*4B），seq_len 超过 ~12K 就超过 48KB 默认上限导致 launch
// 失败。现改为三遍法：求 max -> 求 sum（不缓存，第三遍重算 exp）-> 写出，
// 共享内存只占 warp 归一槽位（32 float）与两个标量。
__global__ void softmax_kernel(const half *__restrict__ input, half *__restrict__ output,
                               int seq_len) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const half *x = input + row * seq_len;
    half       *y = output + row * seq_len;

    __shared__ float warp_buf[32];
    __shared__ float global_max;
    __shared__ float global_sum;

    // Pass 1: 行最大值
    float max_val = -FLT_MAX;
    for (int i = tid; i < seq_len; i += block_size) {
        max_val = fmaxf(max_val, __half2float(x[i]));
    }
    max_val = warp_reduce_max(max_val);

    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) warp_buf[warp_id] = max_val;
    __syncthreads();

    int num_warps = (block_size + 31) / 32;
    if (warp_id == 0) {
        max_val = (lane < num_warps) ? warp_buf[lane] : -FLT_MAX;
        max_val = warp_reduce_max(max_val);
        if (lane == 0) global_max = max_val;
    }
    __syncthreads();

    // Pass 2: exp 和（exp 值不缓存）
    float sum = 0.0f;
    for (int i = tid; i < seq_len; i += block_size) {
        sum += expf(__half2float(x[i]) - global_max);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) warp_buf[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < num_warps) ? warp_buf[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) global_sum = sum;
    }
    __syncthreads();

    // Pass 3: 重算 exp 并写出
    float inv_sum = 1.0f / (global_sum + 1e-6f);
    for (int i = tid; i < seq_len; i += block_size) {
        y[i] = __float2half(expf(__half2float(x[i]) - global_max) * inv_sum);
    }
}

void softmax(const half *input, half *output, int batch_size, int seq_len, cudaStream_t stream) {
    if (batch_size <= 0 || seq_len <= 0) return;

    int block_size = 256;
    softmax_kernel<<<batch_size, block_size, 0, stream>>>(input, output, seq_len);
}

} // namespace kernels
} // namespace tiny_llm
