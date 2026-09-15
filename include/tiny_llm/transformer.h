#pragma once

#include "rope.cuh" // TLLM-003: RoPE cache
#include "tiny_llm/kv_cache.h"
#include "tiny_llm/result.h"
#include "tiny_llm/types.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {

// split-KV decode attention（TLLM-ATTN-SPLITKV）partial 工作区的 split 数上界。
//
// 运行时开关 TLLM_ATTN_SPLITKV 允许 1..kAttnMaxSplits，而缓冲按**上界**一次性分配：
// 这样"每次调用重新解析环境变量"（不缓存、因而测试里 setenv 即时生效、且不需要为测试
// 暴露 reset seam）与"分配大小固定"可以同时成立，不会因为运行中改小/改大取值而越界。
inline constexpr int kAttnMaxSplits = 32;

// 共享中间激活工作区：所有 TransformerLayer 复用同一组 GPU 缓冲。
// 推理逐层串行，同一时刻只有一个层在使用这些缓冲；每层独立分配会按
// 层数线性放大显存（24 层 × ~0.9GB 必然 OOM）。
struct LayerWorkspace {
    half *norm_output = nullptr; // [max_batch, hidden_dim]
    half *q_buf = nullptr;       // [max_batch, num_heads * head_dim]
    half *k_buf = nullptr;       // [max_batch, num_kv_heads * head_dim]
    half *v_buf = nullptr;       // [max_batch, num_kv_heads * head_dim]
    half *attn_output = nullptr; // [max_batch, hidden_dim]（O 投影输出，非就地）
    half *attn_buf = nullptr;    // [max_batch, hidden_dim]（注意力输出，O 投影输入）
    half *ffn_gate = nullptr;    // [max_batch, intermediate_dim]
    half *ffn_up = nullptr;      // [max_batch, intermediate_dim]
    half *ffn_output = nullptr;  // [max_batch, hidden_dim]
    // split-KV partial：[num_heads * kAttnMaxSplits * (2 + head_dim)] fp32，布局见
    // kernels/attention.cuh。kernel 零分配，缓冲由这里持有（层间串行，可复用）。
    float *attn_partial = nullptr;
    int    max_batch_tokens = 0;
    bool   allocated = false;

    void allocate(const ModelConfig &config);
    void free();
    ~LayerWorkspace() { free(); }
    LayerWorkspace() = default;
    LayerWorkspace(const LayerWorkspace &) = delete;
    LayerWorkspace &operator=(const LayerWorkspace &) = delete;
};

// 分页 KV（策略 1）视图：一次 forwardPaged 需要的全部 paged KV 资源与几何。
// pool 是"全局"（含全部层）指针，attentionPaged 内部按 layer_idx_ 计算本层偏移。
struct PagedKVCacheView {
    half      *k_pool = nullptr; // [L * max_num_blocks * block_size * kv_dim]
    half      *v_pool = nullptr;
    const int *block_table = nullptr; // device int[visible_blocks]
    half      *k_scratch = nullptr;   // [max_visible_tokens * kv_dim]
    half      *v_scratch = nullptr;
    int        visible_blocks = 0;
    int        block_size = 0;
    int        max_num_blocks = 0;
    int        max_visible_tokens = 0;
    int        position = 0;         // 本步首 token 的绝对位置
    const int *decode_len = nullptr; // decode 专用 device int；prefill 传 nullptr
};

// TransformerLayer implements a single transformer decoder layer
// with W8A16 quantized weights and KV cache support
class TransformerLayer {
  public:
    TransformerLayer(int layer_idx, const TransformerWeights &weights, const ModelConfig &config,
                     LayerWorkspace *workspace);
    ~TransformerLayer();

    // Non-copyable
    TransformerLayer(const TransformerLayer &) = delete;
    TransformerLayer &operator=(const TransformerLayer &) = delete;

    // 不可移动：weights_/config_ 为 const 引用成员（见 transformer.cpp 说明），
    // 调用方一律用 unique_ptr 持有。

    // Forward pass for single token (decode phase)
    // hidden_states: [batch_size, hidden_dim] - input and output
    // decode_len: device int = 当前可见 KV 长度（appendKV 后 = getSeqLen+1），
    //             只在 decode 分支使用；CUDA Graph 重放前置。
    // rope_pos: device int = 本 token 的绝对位置（RoPE 起始位置，graph 重放前置）。
    // TLLM-005: Returns Result<void> for proper error propagation
    Result<void> forward(half *hidden_states, KVCacheManager &kv_cache, int seq_id, int position,
                         const int *decode_len, const int *rope_pos, const float *rope_cos,
                         const float *rope_sin, cudaStream_t stream = 0);

    // Forward pass for multiple tokens (prefill phase)
    // hidden_states: [batch_size, seq_len, hidden_dim] - input and output
    // rope_pos: device int = 本 batch 起始绝对位置（prefill 为 0）。
    // TLLM-005: Returns Result<void> for proper error propagation
    Result<void> forwardPrefill(half *hidden_states, KVCacheManager &kv_cache, int seq_id,
                                int seq_len, const int *rope_pos, const float *rope_cos,
                                const float *rope_sin, cudaStream_t stream = 0);

    // 分页 KV（策略 1）前向：与 runLayer 的差异只在 attention —— 走
    // attentionPaged（scatter 写入 pool、gather 读回 scratch 后做 attention）。
    // num_tokens == 1 且 kv.decode_len != nullptr 时为 decode（可见长度取
    // kv.decode_len / kv.position+1）；否则为 prefill（可见长度 = num_tokens）。
    Result<void> forwardPaged(half *hidden_states, const PagedKVCacheView &kv, int num_tokens,
                              const int *rope_pos, const float *rope_cos, const float *rope_sin,
                              cudaStream_t stream = 0);

    // Get layer index
    int getLayerIdx() const noexcept { return layer_idx_; }

  private:
    // Attention sublayer
    // decode_len: 见 forward；仅在 num_tokens==1 且 decode_len!=nullptr 时使用。
    // rope_pos: device int RoPE 起始位置。
    Result<void> attention(const half *x, half *output, KVCacheManager &kv_cache, int seq_id,
                           int position, int num_tokens, const int *decode_len, const int *rope_pos,
                           const float *rope_cos, const float *rope_sin, cudaStream_t stream);

    // 分页 KV 版 attention：Q/K/V 投影与 RoPE 同 attention()，之后把 K/V 经
    // block_table scatter 进 pool，再 gather 出可见区间到 scratch 做 attention。
    Result<void> attentionPaged(const half *x, half *output, const PagedKVCacheView &kv,
                                int num_tokens, const int *rope_pos, const float *rope_cos,
                                const float *rope_sin, cudaStream_t stream);

    // Feed-forward network sublayer (SwiGLU)
    void feedForward(const half *x, half *output, int num_tokens, cudaStream_t stream);

    // RMSNorm helper
    void rmsNorm(const half *x, const half *weight, half *output, int num_tokens,
                 cudaStream_t stream);

    // Shared attention + FFN residual block for forward/forwardPrefill
    Result<void> runLayer(half *hidden_states, KVCacheManager &kv_cache, int seq_id, int position,
                          int num_tokens, const int *decode_len, const int *rope_pos,
                          const float *rope_cos, const float *rope_sin, cudaStream_t stream);

    int                       layer_idx_;
    const TransformerWeights &weights_;
    const ModelConfig        &config_;

    // 共享中间激活缓冲（非拥有，由引擎统一管理生命周期）
    LayerWorkspace *ws_ = nullptr;
};

} // namespace tiny_llm
