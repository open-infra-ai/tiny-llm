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
