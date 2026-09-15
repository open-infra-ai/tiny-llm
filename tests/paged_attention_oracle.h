#pragma once

// TLLM-P0-002：paged / contiguous synthetic correctness oracle 的独立参考部分。
//
// 约束（P0_P1_AGENT_BACKLOG.md · TLLM-P0-002）：
//   - 纯 host 实现，不调用任何生产 kernel（尤其不得调用 paged_gather_blocks /
//     paged_scatter_blocks / attention_decode）；
//   - 不依赖外部 GGUF 或真实模型几何；
//   - 只冻结参考语义与地址公式，不改动生产实现。
//
// 冻结地址公式（与 kernels/paged_kv.cu 的语义一致，但此处独立写出）：
//
//   logical_token   t   in [0, visible_tokens)
//   logical_block   b   = t / block_size
//   block_offset    r   = t % block_size
//   physical_block  p   = block_table[b]
//   pool element    idx(layer, t, kh, d)
//                       = ((layer * max_num_blocks + p) * block_size + r) * kv_dim
//                         + kh * head_dim + d
//
//   kv_dim = num_kv_heads * head_dim；pool 为 half，按元素线性寻址。
//   越界 physical_block（p < 0 或 p >= max_num_blocks）按 gather 的 guard 语义
//   整行视为 0（生产 gather 写 0，不寻址 pool）。
//
// GQA 映射（与 kernels/attention.cuh 的 contract 一致）：group = Hq / Hkv，
// kv_head(qh) = qh / group。二者必须整除，否则几何非法。
//
// 已知生产缺口（本 oracle 用 requiredBlocks() 把它写成显式 contract，供
// TLLM-P0-004 的 direct paged kernel 设计处理）：
//   kernels/paged_kv.cu 的 scatter/gather 没有"块表长度"参数；块表长度不足时
//   读 block_table 越界是未定义行为。长度由 caller 保证
//   >= requiredBlocks(visible_tokens, block_size)。src/ffi.cpp 与
//   src/transformer.cpp::forwardPaged 在入口校验该 contract。

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace tiny_llm {
namespace test {

// 描述一次 paged attention 所需的全部几何（不含数据）。
struct PagedGeometry {
    int num_q_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int block_size = 0;
    int max_num_blocks = 0;
    int num_layers = 1;

    int    kvDim() const { return num_kv_heads * head_dim; }
    int    groupSize() const { return num_q_heads / num_kv_heads; }
    size_t layerStride() const {
        return static_cast<size_t>(max_num_blocks) * static_cast<size_t>(block_size) *
               static_cast<size_t>(kvDim());
    }
    size_t poolElements() const { return static_cast<size_t>(num_layers) * layerStride(); }
};

// 块表长度 contract：容纳 visible_tokens 个逻辑 token 所需的最小物理块数。
inline int requiredBlocks(int visible_tokens, int block_size) {
    return (visible_tokens + block_size - 1) / block_size;
}

// 冻结地址公式：逻辑 token t 在本层 pool 内的 half 元素偏移。
// 调用方须先保证 block_table 长度 >= requiredBlocks(visible_tokens, block_size)。
inline size_t pagedRowOffset(const PagedGeometry &g, const std::vector<int> &block_table,
                             int token) {
    const int b = token / g.block_size;
    const int r = token - b * g.block_size;
    const int p = block_table[static_cast<size_t>(b)];
    return (static_cast<size_t>(p) * static_cast<size_t>(g.block_size) + static_cast<size_t>(r)) *
           static_cast<size_t>(g.kvDim());
}

// 独立重建 [visible_tokens, kv_dim] 连续逻辑行（fp32），从单层 pool 读取。
// 不调用生产 kernel。越界 / 负 physical_block 的整行保持 0（gather guard 语义）。
inline std::vector<float> oracleGatherRows(const std::vector<float> &pool_layer,
                                           const PagedGeometry      &g,
                                           const std::vector<int>   &block_table,
                                           int                       visible_tokens) {
    const int          kv_dim = g.kvDim();
    std::vector<float> rows(static_cast<size_t>(visible_tokens) * static_cast<size_t>(kv_dim),
                            0.0f);
    for (int t = 0; t < visible_tokens; ++t) {
        const int b = t / g.block_size;
        if (b >= static_cast<int>(block_table.size())) continue;
        const int p = block_table[static_cast<size_t>(b)];
        if (p < 0 || p >= g.max_num_blocks) continue;
        const int    r = t - b * g.block_size;
        const size_t src =
            (static_cast<size_t>(p) * static_cast<size_t>(g.block_size) + static_cast<size_t>(r)) *
            static_cast<size_t>(kv_dim);
        for (int c = 0; c < kv_dim; ++c) {
            rows[static_cast<size_t>(t) * static_cast<size_t>(kv_dim) + static_cast<size_t>(c)] =
                pool_layer[src + static_cast<size_t>(c)];
        }
    }
    return rows;
}

// fp32 decode attention：q [1, Hq, D] × k/v [T, Hkv, D] -> out [1, Hq, D]。
// 两遍 softmax（max -> exp -> 归一化），与生产的 online-softmax 数值等价。
inline std::vector<float> oracleAttentionDecode(const std::vector<float> &q,
                                                const std::vector<float> &k_rows,
                                                const std::vector<float> &v_rows,
                                                int visible_tokens, const PagedGeometry &g) {
    const int          D = g.head_dim;
    const int          Hq = g.num_q_heads;
    const int          Hkv = g.num_kv_heads;
    const int          group = Hq / Hkv;
    const float        scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> out(static_cast<size_t>(Hq) * static_cast<size_t>(D), 0.0f);
    std::vector<float> scores(static_cast<size_t>(visible_tokens), 0.0f);

    for (int qh = 0; qh < Hq; ++qh) {
        const int    kh = qh / group;
        const size_t q_base = static_cast<size_t>(qh) * static_cast<size_t>(D);

        float max_score = -std::numeric_limits<float>::infinity();
        for (int t = 0; t < visible_tokens; ++t) {
            const size_t k_base =
                (static_cast<size_t>(t) * static_cast<size_t>(Hkv) + static_cast<size_t>(kh)) *
                static_cast<size_t>(D);
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += q[q_base + static_cast<size_t>(d)] * k_rows[k_base + static_cast<size_t>(d)];
            }
            scores[static_cast<size_t>(t)] = dot * scale;
            max_score = std::max(max_score, scores[static_cast<size_t>(t)]);
        }

        float sum = 0.0f;
        for (int t = 0; t < visible_tokens; ++t) {
            const float e = std::exp(scores[static_cast<size_t>(t)] - max_score);
            scores[static_cast<size_t>(t)] = e;
            sum += e;
        }
        const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int t = 0; t < visible_tokens; ++t) {
                const size_t v_base =
                    (static_cast<size_t>(t) * static_cast<size_t>(Hkv) + static_cast<size_t>(kh)) *
                    static_cast<size_t>(D);
                acc += scores[static_cast<size_t>(t)] * v_rows[v_base + static_cast<size_t>(d)];
            }
            out[q_base + static_cast<size_t>(d)] = acc * inv;
        }
    }
    return out;
}

// 比较口径：最大绝对误差与参考尺度（用于给出可读的失败信息）。
inline float maxAbsDiff(const std::vector<float> &a, const std::vector<float> &b) {
    const size_t n = std::min(a.size(), b.size());
    float        m = 0.0f;
    for (size_t i = 0; i < n; ++i)
        m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

inline float maxAbsValue(const std::vector<float> &a) {
    float m = 0.0f;
    for (float v : a)
        m = std::max(m, std::fabs(v));
    return m;
}

// 几何合法性：GQA 必须整除、head_dim 必须为偶数（RoPE half-split）、正尺寸。
inline bool geometryIsValid(const PagedGeometry &g) {
    return g.num_q_heads > 0 && g.num_kv_heads > 0 && g.head_dim > 0 && g.block_size > 0 &&
           g.max_num_blocks > 0 && g.num_layers > 0 && g.num_q_heads % g.num_kv_heads == 0 &&
           g.head_dim % 2 == 0;
}

} // namespace test
} // namespace tiny_llm
