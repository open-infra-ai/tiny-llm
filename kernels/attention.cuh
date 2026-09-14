#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// ============================================================================
// Token-major layout contract (TLLM-001/002)
//
// All attention tensors use token-major physical layout:
//   Q       [S, Hq,  D]   q(s,h,d)     = ((s * Hq  + h)  * D + d)
//   K       [S, Hkv, D]   k(s,kh,d)    = ((s * Hkv + kh) * D + d)
//   V       [S, Hkv, D]   v(s,kh,d)    = ((s * Hkv + kh) * D + d)
//   K_cache [T, Hkv, D]   cache(t,kh,d)= ((t * Hkv + kh) * D + d)
//   V_cache [T, Hkv, D]
//   O       [S, Hq,  D]
//
// GQA mapping (TLLM-002):
//   group_size = Hq / Hkv
//   kv_head(qh) = qh / group_size
//
// Pre-requisites (validated by caller):
//   Hq > 0, Hkv > 0, Hq % Hkv == 0, D > 0, D even
// ============================================================================

// Decode attention: single query token against cached K/V
// Q:       [1, Hq,  D]
// K_cache: [T, Hkv, D]   (T = visible_len, includes the just-appended token)
// V_cache: [T, Hkv, D]
// O:       [1, Hq,  D]
//
// CUDA Graph 重放前置条件（任务 3.1）：visible_len 不再作为 kernel 参数，
// 而是由 device 端 int 变量提供。graph 捕获后 host 只需更新该 device 值
// （同一 stream 上的 cudaMemcpyAsync）再重放，无需重新 capture。
// 调用方必须保证 *device_visible_len 已在同一 stream 上被写入。
void attention_decode(const half *__restrict__ query, const half *__restrict__ k_cache,
                      const half *__restrict__ v_cache, half *__restrict__ output, float scale,
                      int num_q_heads, int num_kv_heads, const int *device_visible_len,
                      int head_dim, cudaStream_t stream = 0);

// 旧签名薄封装：host int 版本，内部分配/复用 device int 后转发给上面版本。
// 供测试与不需要 graph 重放的调用方使用。
void attention_decode(const half *__restrict__ query, const half *__restrict__ k_cache,
                      const half *__restrict__ v_cache, half *__restrict__ output, float scale,
                      int num_q_heads, int num_kv_heads, int visible_len, int head_dim,
                      cudaStream_t stream = 0);

// Direct paged decode attention（TLLM-P0-004）：直接从物理 K/V pool + block table 寻址，
// 不再把可见窗口 gather 到连续 scratch。语义与 "gather 到连续缓冲后调用
// attention_decode" 完全等价（包括非法块 id 视为零行的既有语义），因此两条路径的输出
// 必须逐位相同——由 tests/test_paged_direct.cpp 守护。
//
// 几何与寻址（与 kernels/paged_kv.cu 的 scatter/gather 一致）：
//   kv_dim       = num_kv_heads * head_dim
//   b = t / block_size;  r = t % block_size;  p = block_table[b]
//   K[ℓ,t,kh,d] 偏移 = (p * block_size + r) * kv_dim + kh * head_dim + d
// k_pool_layer / v_pool_layer 是**已含 layer 偏移**的本层指针（由 caller 计算）。
//
// 前置条件（由 TransformerLayer::forwardPaged 的 host 校验保证，kernel 侧另做
// 廉价防御性检查，不作为错误契约）：
//   - 指针非空；num_q_heads > 0、num_kv_heads > 0、num_q_heads % num_kv_heads == 0；
//   - head_dim > 0 且动态共享内存需求 (ATTEND_TILE+8+head_dim)*4 + head_dim*2
//     + ATTEND_TILE*4 + head_dim*2 在设备上限内；
//   - block_size > 0、max_num_blocks > 0；
//   - table_len >= ceil(visible_tokens / block_size)（table_len 不足时按零行处理，
//     不会越界读块表）；
//   - *device_visible_tokens == 本步可见 KV 长度（decode 时为 position + 1）。
//
// visible_tokens 走 device int，launch 路径无 D2H、无分配，因此可被 CUDA Graph 捕获。
void attention_decode_paged(const half *__restrict__ query, const half *__restrict__ k_pool_layer,
                            const half *__restrict__ v_pool_layer,
                            const int *__restrict__ block_table, half *__restrict__ output,
                            float scale, int num_q_heads, int num_kv_heads, int head_dim,
                            const int *device_visible_tokens, int block_size, int max_num_blocks,
                            int table_len, cudaStream_t stream = 0);

// ============================================================================
// Split-KV decode attention（TLLM-ATTN-SPLITKV）
//
// 把可见 KV **逻辑窗口**切成 num_splits 段，每段一个 block 独立归约（grid = (Hq, num_splits)），
// 再由一个 combine kernel（grid = (Hq, 1)）按 online-softmax 合并式折成输出。动机是并行度：
// 单 query decode 原本只有 Hq 这一个并行轴，实测 occupancy 8.33%、无任何资源饱和
// （见 docs/architecture/decode-attention-splitkv-design.md §2）。
//
// 与 attention_decode / attention_decode_paged 的关系：
//   - **语义等价**：两者是同一次归约的两种切分方式，输出差异仅来自 fp32 求和顺序；
//   - **num_splits == 1 时逐位相同**（见 kernels/attention.cu 的 combine 注释），这条是
//     重构正确性的回归锚点；
//   - **CUDA Graph 可捕获**：段范围在 device 端由 *device_visible_* 派生，num_splits 是 host
//     参数（grid 捕获时固定），partial 缓冲由调用方预分配、捕获期零分配零 D2H；
//   - 两个既有入口**保持不变**（等价于单遍路径），因此既有调用点无需改动。
//
// partial_workspace 布局（device fp32，连续，无 padding）：
//   offset(head, split) = (head * num_splits + split) * (2 + head_dim)
//     [+0]         = m      该段最大 score（空段为 -FLT_MAX，combine 视作中性）
//     [+1]         = l      该段 Σ exp
//     [+2 .. +1+D] = acc[d] 该段 Σ exp * v
//   total floats = num_q_heads * num_splits * (2 + head_dim)
// 调用方负责分配（kernel 零分配、零状态）。每个 block 都会写自己的槽位，含空段。
//
// 前置条件：指针非空；num_q_heads > 0、num_kv_heads > 0、num_q_heads % num_kv_heads == 0；
//   head_dim > 0；1 <= num_splits <= 65535（grid.y 上界）；num_splits == 1 为恒等切分。
void attention_decode_splitkv(const half *__restrict__ query, const half *__restrict__ k_cache,
                              const half *__restrict__ v_cache, half *__restrict__ output,
                              float scale, int num_q_heads, int num_kv_heads,
                              const int *__restrict__ device_visible_len, int head_dim,
                              float *__restrict__ partial_workspace, int      num_splits,
                              cudaStream_t stream = 0);

void attention_decode_paged_splitkv(
    const half *__restrict__ query, const half *__restrict__ k_pool_layer,
    const half *__restrict__ v_pool_layer, const int *__restrict__ block_table,
    half *__restrict__ output, float scale, int num_q_heads, int num_kv_heads, int head_dim,
    const int *__restrict__ device_visible_tokens, int block_size, int max_num_blocks,
    int table_len, float *__restrict__ partial_workspace, int num_splits, cudaStream_t stream = 0);

// Prefill attention: full sequence with causal masking
// Q: [S, Hq,  D]
// K: [S, Hkv, D]
// V: [S, Hkv, D]
// O: [S, Hq,  D]
void attention_prefill(const half *__restrict__ query, const half *__restrict__ key,
                       const half *__restrict__ value, half *__restrict__ output, float scale,
                       int num_q_heads, int num_kv_heads, int seq_len, int head_dim,
                       cudaStream_t stream = 0);

// Softmax kernel (for testing)
void softmax(const half *__restrict__ input, half *__restrict__ output, int batch_size, int seq_len,
             cudaStream_t stream = 0);

// Get attention weights for testing causal mask
// Q: [query_len, Hq,  D]
// K: [key_len,   Hkv, D]
// weights: [query_len, Hq, key_len]  (token-major)
void get_attention_weights(const half *__restrict__ query, const half *__restrict__ key,
                           half *__restrict__ weights, float scale, int num_q_heads,
                           int num_kv_heads, int query_len, int key_len, int head_dim,
                           bool apply_causal_mask, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
