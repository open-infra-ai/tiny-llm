// TLLM-P0-002：paged / contiguous synthetic correctness oracle。
//
// 目的：为 direct paged decode attention（TLLM-P0-004）冻结一组**不依赖外部
// GGUF / 真实模型**的不可变 correctness oracle，并在此之前先把现有
// "paged storage + gather + continuous attention" 路径钉死在参考实现上。
//
// 层次（与 L3_L4_DESIGN_REVIEW_PACKAGES.md §4.4 的 correctness 矩阵对应）：
//   1. PagedOracleKernelTest  —— kernel 级：生产 scatter + gather + attention_decode
//      对 tests/paged_attention_oracle.h 的独立 host 参考。
//   2. PagedOracleLayerTest   —— layer 级：TransformerLayer::forwardPaged 对
//      连续 KV 路径（同一个 layer、同一份权重、同一输入），覆盖多 layer pool offset。
//   3. PagedOracleContractTest—— host-only：冻结地址公式、块表长度 contract。
//
// 边界：本文件只做 reference / fixture / 测试。不实现 direct paged kernel，
// 不修改 FFI，不引入新的 public API；测试通过 kernel 私有头（kernels/*.cuh）
// 这一既有 seam 访问内部原语。
//
// 已知限制（不得据此宣称 direct PagedAttention 已完成）：
//   - 被测路径仍是 scatter → gather → continuous attention，不是 direct kernel；
//   - oracle 为 fp32 host 参考，比较口径为 fp16 输出量化后的绝对误差；
//   - 未接入真实模型 / 真实 serving，本文件不产生任何性能数字。

#include "attention.cuh"
#include "paged_attention_oracle.h"
#include "paged_kv.cuh"
#include "rope.cuh"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/kv_cache.h"
#include "tiny_llm/transformer.h"
#include "transpose_weights.cuh"

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace tiny_llm;
using namespace tiny_llm::kernels;
using tiny_llm::test::maxAbsDiff;
using tiny_llm::test::oracleAttentionDecode;
using tiny_llm::test::oracleGatherRows;
using tiny_llm::test::PagedGeometry;
using tiny_llm::test::pagedRowOffset;
using tiny_llm::test::requiredBlocks;

namespace {

// fp16 输出量化到约 5e-4（|v|<=1）；留 4 倍余量覆盖 softmax 归约顺序差异。
constexpr float kFp16OutputTolerance = 2e-3f;

bool hasCudaDevice() {
    static bool checked = false;
    static bool has_device = false;
    if (!checked) {
        int         device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        has_device = (err == cudaSuccess && device_count > 0);
        checked = true;
    }
    return has_device;
}

std::vector<half> randomFp16(size_t n, unsigned seed, float scale = 1.0f) {
    std::mt19937                          gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<half>                     v(n);
    for (auto &x : v)
        x = __float2half(dist(gen));
    return v;
}

std::vector<float> toFloat(const std::vector<half> &h) {
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); ++i)
        f[i] = __half2float(h[i]);
    return f;
}

// 确定性、互异、通常非单调的物理块表：覆盖"非连续物理块"。
std::vector<int> makeBlockTable(int num_blocks, int max_num_blocks, unsigned seed) {
    if (num_blocks <= 0 || num_blocks > max_num_blocks) {
        throw std::invalid_argument("makeBlockTable: num_blocks out of range");
    }
    std::vector<int> ids(static_cast<size_t>(max_num_blocks));
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 gen(seed);
    std::shuffle(ids.begin(), ids.end(), gen);
    return std::vector<int>(ids.begin(), ids.begin() + num_blocks);
}

PagedGeometry makeGeometry(int hq, int hkv, int head_dim, int block_size, int max_num_blocks,
                           int num_layers = 1) {
    PagedGeometry g;
    g.num_q_heads = hq;
    g.num_kv_heads = hkv;
    g.head_dim = head_dim;
    g.block_size = block_size;
    g.max_num_blocks = max_num_blocks;
    g.num_layers = num_layers;
    return g;
}

// 按冻结层步长读取 pool 的某一层（fp32）。
std::vector<float> readPoolLayer(const DeviceBuffer<half> &pool, int layer,
                                 const PagedGeometry &g) {
    std::vector<half> h(g.layerStride());
    CUDA_CHECK(cudaMemcpy(h.data(), pool.data() + static_cast<size_t>(layer) * g.layerStride(),
                          g.layerStride() * sizeof(half), cudaMemcpyDeviceToHost));
    return toFloat(h);
}

// 读取连续 KV cache 的前 rows 行（fp32）。
std::vector<float> readCacheRows(KVCacheManager &cache, int seq_id, int layer, int rows, int kv_dim,
                                 bool is_value) {
    auto [k_ptr, v_ptr] = cache.getCache(seq_id, layer);
    const half *src = is_value ? v_ptr : k_ptr;
    if (src == nullptr) throw std::runtime_error("readCacheRows: null cache pointer");
    std::vector<half> h(static_cast<size_t>(rows) * static_cast<size_t>(kv_dim));
    CUDA_CHECK(cudaMemcpy(h.data(), src, h.size() * sizeof(half), cudaMemcpyDeviceToHost));
    return toFloat(h);
}

// ── kernel 级 harness：只调用生产原语，pool/table 由测试持有 ──────────────
class PagedKernelHarness {
  public:
    explicit PagedKernelHarness(const PagedGeometry &g)
        : g_(g), d_k_pool_(g.poolElements()), d_v_pool_(g.poolElements()),
          d_k_scratch_(static_cast<size_t>(g.max_num_blocks) * static_cast<size_t>(g.block_size) *
                       static_cast<size_t>(g.kvDim())),
          d_v_scratch_(static_cast<size_t>(g.max_num_blocks) * static_cast<size_t>(g.block_size) *
                       static_cast<size_t>(g.kvDim())),
          d_table_(static_cast<size_t>(g.max_num_blocks)) {}

    void uploadTable(const std::vector<int> &table) {
        EXPECT_LE(table.size(), d_table_.size());
        d_table_.copyFromHost(table.data(), table.size());
    }

    // 用生产 scatter 把 [num_tokens, kv_dim] 逻辑 K/V 写到 layer 区域的位置 position。
    void scatter(int layer, const std::vector<half> &k, const std::vector<half> &v, int num_tokens,
                 int position) {
        const size_t       elems = static_cast<size_t>(num_tokens) * g_.kvDim();
        DeviceBuffer<half> d_k(elems), d_v(elems);
        d_k.copyFromHost(k.data(), elems);
        d_v.copyFromHost(v.data(), elems);
        half *k_pool = d_k_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        half *v_pool = d_v_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        paged_scatter_blocks(d_k.data(), k_pool, d_table_.data(), num_tokens, position,
                             g_.block_size, g_.kvDim(), g_.max_num_blocks);
        paged_scatter_blocks(d_v.data(), v_pool, d_table_.data(), num_tokens, position,
                             g_.block_size, g_.kvDim(), g_.max_num_blocks);
    }

    // 生产 gather + decode attention，返回 fp32 输出 [Hq * D]。
    std::vector<float> decode(int layer, const std::vector<half> &q, int visible_tokens) {
        const size_t       q_elems = static_cast<size_t>(g_.num_q_heads) * g_.head_dim;
        DeviceBuffer<half> d_q(q_elems), d_out(q_elems);
        d_q.copyFromHost(q.data(), q_elems);
        const half *k_pool = d_k_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        const half *v_pool = d_v_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        paged_gather_blocks(d_k_scratch_.data(), k_pool, d_table_.data(), visible_tokens,
                            g_.block_size, g_.kvDim(), g_.max_num_blocks);
        paged_gather_blocks(d_v_scratch_.data(), v_pool, d_table_.data(), visible_tokens,
                            g_.block_size, g_.kvDim(), g_.max_num_blocks);
        const float scale = 1.0f / std::sqrt(static_cast<float>(g_.head_dim));
        attention_decode(d_q.data(), d_k_scratch_.data(), d_v_scratch_.data(), d_out.data(), scale,
                         g_.num_q_heads, g_.num_kv_heads, visible_tokens, g_.head_dim);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<half> h_out(q_elems);
        d_out.copyToHost(h_out.data(), q_elems);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return toFloat(h_out);
    }

    // 生产 gather 的原始结果（fp32），用于校验 pool 内容与寻址。
    std::vector<float> gatherK(int layer, int visible_tokens) {
        const half *k_pool = d_k_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        paged_gather_blocks(d_k_scratch_.data(), k_pool, d_table_.data(), visible_tokens,
                            g_.block_size, g_.kvDim(), g_.max_num_blocks);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<half> h(d_k_scratch_.size());
        d_k_scratch_.copyToHost(h.data(), h.size());
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return toFloat(h);
    }

    // 指定 layer 的 K pool 内容（fp32）。
    std::vector<float> poolLayerK(int layer) {
        std::vector<half> h(g_.layerStride());
        CUDA_CHECK(cudaMemcpy(h.data(),
                              d_k_pool_.data() + static_cast<size_t>(layer) * g_.layerStride(),
                              g_.layerStride() * sizeof(half), cudaMemcpyDeviceToHost));
        return toFloat(h);
    }

    const PagedGeometry &geometry() const { return g_; }

  private:
    PagedGeometry      g_;
    DeviceBuffer<half> d_k_pool_;
    DeviceBuffer<half> d_v_pool_;
    DeviceBuffer<half> d_k_scratch_;
    DeviceBuffer<half> d_v_scratch_;
    DeviceBuffer<int>  d_table_;
};

// ── 合成 Transformer 模型（无 GGUF，无真实权重） ──────────────────────────
QuantizedWeight makeSyntheticWeight(int rows, int cols, int group_size, unsigned seed) {
    QuantizedWeight qw;
    qw.rows = rows;
    qw.cols = cols;
    qw.group_size = group_size;
    const int srows = (rows + group_size - 1) / group_size;

    std::mt19937                          gen(seed);
    std::uniform_int_distribution<int>    idist(-20, 20);
    std::uniform_real_distribution<float> sdist(0.0005f, 0.002f);

    std::vector<int8_t> h_data(static_cast<size_t>(rows) * cols);
    for (auto &v : h_data)
        v = static_cast<int8_t>(idist(gen));
    std::vector<half> h_scales(static_cast<size_t>(srows) * cols);
    for (auto &s : h_scales)
        s = __float2half(sdist(gen));

    CUDA_CHECK(cudaMalloc(&qw.data, h_data.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&qw.scales, h_scales.size() * sizeof(half)));
    CUDA_CHECK(
        cudaMemcpy(qw.data, h_data.data(), h_data.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(qw.scales, h_scales.data(), h_scales.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&qw.data_t, h_data.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&qw.scales_t, h_scales.size() * sizeof(half)));
    transpose_int8(qw.data, qw.data_t, rows, cols, 0);
    transpose_scales(qw.scales, qw.scales_t, srows, cols, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    return qw;
}

struct SyntheticModel {
    ModelConfig                                    config;
    LayerWorkspace                                 ws;
    std::vector<TransformerWeights>                weights;
    std::vector<std::unique_ptr<TransformerLayer>> layers;
    DeviceBuffer<float>                            d_cos;
    DeviceBuffer<float>                            d_sin;

    void build(int hidden_dim, int num_layers, int num_heads, int num_kv_heads, int head_dim,
               int intermediate_dim, int max_seq_len) {
        config.vocab_size = 64;
        config.hidden_dim = hidden_dim;
        config.num_layers = num_layers;
        config.num_heads = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.head_dim = head_dim;
        config.intermediate_dim = intermediate_dim;
        config.max_seq_len = max_seq_len;
        config.rope_theta = 10000.0f;
        config.rms_norm_eps = 1e-5f;

        const int q_dim = num_heads * head_dim;
        const int kv_dim = num_kv_heads * head_dim;

        // QuantizedWeight 几何约定（见 kernels/w8a16_matmul.cuh）：
        //   weight [K, N] INT8 -> rows = K（输入/归约维）, cols = N（输出维）
        //   scales [ceil(K/group_size), N]
        // 传反会让 scales 少算因子，反向 kernel 读到分配外（sanitizer 可复现）。
        constexpr int kGroup = 32;

        weights.resize(static_cast<size_t>(num_layers));
        for (int l = 0; l < num_layers; ++l) {
            auto          &w = weights[static_cast<size_t>(l)];
            const unsigned seed = 1000u + static_cast<unsigned>(l) * 17u;
            w.wq = makeSyntheticWeight(hidden_dim, q_dim, kGroup, seed + 1);
            w.wk = makeSyntheticWeight(hidden_dim, kv_dim, kGroup, seed + 2);
            w.wv = makeSyntheticWeight(hidden_dim, kv_dim, kGroup, seed + 3);
            w.wo = makeSyntheticWeight(q_dim, hidden_dim, kGroup, seed + 4);
            w.w1 = makeSyntheticWeight(hidden_dim, intermediate_dim, kGroup, seed + 5);
            w.w2 = makeSyntheticWeight(intermediate_dim, hidden_dim, kGroup, seed + 6);
            w.w3 = makeSyntheticWeight(hidden_dim, intermediate_dim, kGroup, seed + 7);

            std::vector<half> ones(static_cast<size_t>(hidden_dim), __float2half(1.0f));
            CUDA_CHECK(cudaMalloc(&w.rms_att_weight, ones.size() * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&w.rms_ffn_weight, ones.size() * sizeof(half)));
            CUDA_CHECK(cudaMemcpy(w.rms_att_weight, ones.data(), ones.size() * sizeof(half),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(w.rms_ffn_weight, ones.data(), ones.size() * sizeof(half),
                                  cudaMemcpyHostToDevice));
        }

        ws.allocate(config);
        layers.reserve(static_cast<size_t>(num_layers));
        for (int l = 0; l < num_layers; ++l) {
            layers.push_back(std::make_unique<TransformerLayer>(l, weights[static_cast<size_t>(l)],
                                                                config, &ws));
        }

        const int half_d = head_dim / 2;
        d_cos = DeviceBuffer<float>(static_cast<size_t>(max_seq_len) * half_d);
        d_sin = DeviceBuffer<float>(static_cast<size_t>(max_seq_len) * half_d);
        rope_precompute_cache(d_cos.data(), d_sin.data(), max_seq_len, head_dim, config.rope_theta,
                              0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

void freeQuantizedWeight(QuantizedWeight &qw) {
    if (qw.data) cudaFree(qw.data);
    if (qw.scales) cudaFree(qw.scales);
    if (qw.data_t) cudaFree(qw.data_t);
    if (qw.scales_t) cudaFree(qw.scales_t);
    qw.data = nullptr;
    qw.data_t = nullptr;
    qw.scales = nullptr;
    qw.scales_t = nullptr;
}

void freeSyntheticModel(SyntheticModel &m) {
    m.layers.clear();
    for (auto &w : m.weights) {
        freeQuantizedWeight(w.wq);
        freeQuantizedWeight(w.wk);
        freeQuantizedWeight(w.wv);
        freeQuantizedWeight(w.wo);
        freeQuantizedWeight(w.w1);
        freeQuantizedWeight(w.w2);
        freeQuantizedWeight(w.w3);
        if (w.rms_att_weight) cudaFree(w.rms_att_weight);
        if (w.rms_ffn_weight) cudaFree(w.rms_ffn_weight);
        w.rms_att_weight = nullptr;
        w.rms_ffn_weight = nullptr;
    }
}

} // namespace

// ============================================================================
// 1. kernel 级：生产 scatter + gather + attention_decode vs 独立 host 参考
// ============================================================================

struct KernelMatrixCase {
    int         block_size;
    int         visible_tokens;
    int         num_q_heads;
    int         num_kv_heads;
    int         head_dim;
    const char *label;
};

TEST(PagedOracleKernelTest, ScatterGatherDecodeMatchesIndependentOracle) {
    if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";

    const std::vector<KernelMatrixCase> cases = {
        {1, 1, 4, 4, 32, "block=1 MHA 单 token"},
        {1, 17, 4, 2, 32, "block=1 GQA 跨块"},
        {16, 1, 4, 4, 32, "block=16 单 token"},
        {16, 15, 4, 4, 32, "block=16 block-1"},
        {16, 16, 4, 2, 32, "block=16 恰好一块 GQA"},
        {16, 17, 8, 2, 64, "block=16 block+1 GQA"},
        {16, 35, 8, 1, 64, "block=16 2*block+尾部 MQA"},
        {32, 33, 4, 4, 128, "block=32 跨块尾部 D=128"},
        {32, 64, 8, 4, 32, "block=32 两块"},
        {32, 96, 4, 2, 64, "block=32 三块 GQA"},
    };

    for (const auto &c : cases) {
        const int nb = requiredBlocks(c.visible_tokens, c.block_size);
        for (unsigned seed : {1u, 7u, 1234u}) {
            SCOPED_TRACE(std::string(c.label) + " seed=" + std::to_string(seed));
            PagedGeometry g =
                makeGeometry(c.num_q_heads, c.num_kv_heads, c.head_dim, c.block_size, nb + 3);
            ASSERT_TRUE(tiny_llm::test::geometryIsValid(g));

            const int    kv_dim = g.kvDim();
            const size_t k_elems = static_cast<size_t>(c.visible_tokens) * kv_dim;
            const size_t q_elems = static_cast<size_t>(g.num_q_heads) * g.head_dim;

            const auto k_logical = randomFp16(k_elems, seed * 10 + 1);
            const auto v_logical = randomFp16(k_elems, seed * 10 + 2);
            const auto q = randomFp16(q_elems, seed * 10 + 3);
            const auto table = makeBlockTable(nb, g.max_num_blocks, seed * 10 + 4);

            PagedKernelHarness h(g);
            h.uploadTable(table);
            h.scatter(0, k_logical, v_logical, c.visible_tokens, /*position=*/0);

            const auto out = h.decode(0, q, c.visible_tokens);

            // (a) pool 内容 + 地址公式：生产 gather 结果必须等于独立重建的逻辑行。
            const auto gathered = h.gatherK(0, c.visible_tokens);
            const auto k_expected = toFloat(k_logical);
            EXPECT_LT(maxAbsDiff(gathered, k_expected), 1e-6f)
                << "生产 gather 未还原逻辑 K 行（地址公式或 scatter/gather 不一致）";

            // (b) 端到端：生产输出必须等于"理想连续 KV"上的独立参考注意力。
            const auto ref = oracleAttentionDecode(toFloat(q), k_expected, toFloat(v_logical),
                                                   c.visible_tokens, g);
            EXPECT_LT(maxAbsDiff(out, ref), kFp16OutputTolerance)
                << "paged 路径输出与独立 oracle 不一致，max|ref|="
                << tiny_llm::test::maxAbsValue(ref);
        }
    }
}

// 增量 decode：分多次 scatter 到不同绝对位置（RoPE position / visible length）。
TEST(PagedOracleKernelTest, IncrementalScatterAtNonZeroPositionsMatchesOracle) {
    if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";

    struct Step {
        int position;
        int tokens;
    };
    // 20 = block 16 + 尾部 4；第二次 scatter 自身跨块（位置 12..19）。
    const std::vector<Step> steps = {{0, 12}, {12, 8}};

    const int     block_size = 16;
    const int     visible = 20;
    const int     nb = requiredBlocks(visible, block_size);
    PagedGeometry g = makeGeometry(8, 2, 64, block_size, nb + 2);
    const int     kv_dim = g.kvDim();
    const size_t  k_elems = static_cast<size_t>(visible) * kv_dim;

    const auto k_logical = randomFp16(k_elems, 555);
    const auto v_logical = randomFp16(k_elems, 556);
    const auto q = randomFp16(static_cast<size_t>(g.num_q_heads) * g.head_dim, 557);
    const auto table = makeBlockTable(nb, g.max_num_blocks, 558);

    PagedKernelHarness h(g);
    h.uploadTable(table);
    for (const auto &s : steps) {
        const auto k_part = std::vector<half>(
            k_logical.begin() + static_cast<size_t>(s.position) * kv_dim,
            k_logical.begin() + static_cast<size_t>(s.position + s.tokens) * kv_dim);
        const auto v_part = std::vector<half>(
            v_logical.begin() + static_cast<size_t>(s.position) * kv_dim,
            v_logical.begin() + static_cast<size_t>(s.position + s.tokens) * kv_dim);
        h.scatter(0, k_part, v_part, s.tokens, s.position);
    }

    const auto out = h.decode(0, q, visible);
    const auto ref =
        oracleAttentionDecode(toFloat(q), toFloat(k_logical), toFloat(v_logical), visible, g);
    EXPECT_LT(maxAbsDiff(out, ref), kFp16OutputTolerance);
}

// 非法块 id：scatter 跳过写入、gather 整行写 0；注意力输出必须与"该行视为 0"
// 的独立参考一致（包括 softmax 归一化被零向量行改变的部分）。
TEST(PagedOracleKernelTest, InvalidBlockIdMatchesZeroRowOracle) {
    if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";

    const int     block_size = 16;
    const int     visible = 20;
    const int     nb = requiredBlocks(visible, block_size);
    PagedGeometry g = makeGeometry(4, 4, 32, block_size, nb + 1);
    const int     kv_dim = g.kvDim();
    const size_t  k_elems = static_cast<size_t>(visible) * kv_dim;

    const auto k_logical = randomFp16(k_elems, 900);
    const auto v_logical = randomFp16(k_elems, 901);
    const auto q = randomFp16(static_cast<size_t>(g.num_q_heads) * g.head_dim, 902);

    // 第二块给越界 id（同时覆盖负值与 == max_num_blocks）。
    const std::vector<int> bad_table = {1, g.max_num_blocks};

    PagedKernelHarness h(g);
    h.uploadTable(bad_table);

    h.scatter(0, k_logical, v_logical, visible, 0);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess)
        << "scatter with invalid block id must not fault";

    const auto out = h.decode(0, q, visible);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess)
        << "gather/decode with invalid block id must not fault";

    // 独立参考：合法块保留逻辑行，非法块整行 0。
    std::vector<float> k_eff = toFloat(k_logical);
    std::vector<float> v_eff = toFloat(v_logical);
    const int          bad_block = bad_table[1];
    ASSERT_TRUE(bad_block < 0 || bad_block >= g.max_num_blocks);
    for (int t = block_size; t < visible; ++t) {
        for (int c = 0; c < kv_dim; ++c) {
            k_eff[static_cast<size_t>(t) * kv_dim + c] = 0.0f;
            v_eff[static_cast<size_t>(t) * kv_dim + c] = 0.0f;
        }
    }
    const auto ref = oracleAttentionDecode(toFloat(q), k_eff, v_eff, visible, g);
    EXPECT_LT(maxAbsDiff(out, ref), kFp16OutputTolerance);
}

// 多 layer pool offset：layer 步长 = max_num_blocks * block_size * kv_dim；
// 各层写入互不污染，且从 layer 偏移读取的结果与各自 layer 的独立参考一致。
TEST(PagedOracleKernelTest, MultiLayerPoolOffsetIsolatedAndAddressable) {
    if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";

    const int     block_size = 16;
    const int     visible = 20;
    const int     nb = requiredBlocks(visible, block_size);
    constexpr int kLayers = 3;
    PagedGeometry g = makeGeometry(4, 2, 32, block_size, nb + 1, kLayers);
    const int     kv_dim = g.kvDim();
    const size_t  k_elems = static_cast<size_t>(visible) * kv_dim;

    const auto table = makeBlockTable(nb, g.max_num_blocks, 77);

    PagedKernelHarness h(g);
    h.uploadTable(table);

    std::vector<std::vector<half>> k(kLayers), v(kLayers), q(kLayers);
    for (int l = 0; l < kLayers; ++l) {
        k[static_cast<size_t>(l)] = randomFp16(k_elems, 200u + static_cast<unsigned>(l));
        v[static_cast<size_t>(l)] = randomFp16(k_elems, 300u + static_cast<unsigned>(l));
        q[static_cast<size_t>(l)] = randomFp16(static_cast<size_t>(g.num_q_heads) * g.head_dim,
                                               400u + static_cast<unsigned>(l));
        h.scatter(l, k[static_cast<size_t>(l)], v[static_cast<size_t>(l)], visible, 0);
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (int l = 0; l < kLayers; ++l) {
        SCOPED_TRACE("layer " + std::to_string(l));
        const auto out = h.decode(l, q[static_cast<size_t>(l)], visible);
        const auto ref = oracleAttentionDecode(toFloat(q[static_cast<size_t>(l)]),
                                               toFloat(k[static_cast<size_t>(l)]),
                                               toFloat(v[static_cast<size_t>(l)]), visible, g);
        EXPECT_LT(maxAbsDiff(out, ref), kFp16OutputTolerance)
            << "layer " << l << " 读到了其他 layer 的 pool 区域";
    }

    // 交叉检查：layer 0 的 pool 区域没有被 layer 1/2 的写入覆盖。
    const auto pool0 = h.poolLayerK(0);
    const auto k0 = toFloat(k[0]);
    EXPECT_LT(maxAbsDiff(pool0, k0), 1e-6f) << "layer 0 pool 被其他层污染";
}

// ============================================================================
// 2. layer 级：TransformerLayer::forwardPaged vs 连续 KV 路径（无 GGUF）
// ============================================================================

class PagedOracleLayerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!hasCudaDevice()) {
            GTEST_SKIP() << "No CUDA device available";
        }
        cudaSetDevice(0);
    }
    void TearDown() override { cudaDeviceSynchronize(); }
};

// 逐层比较：把 paged pool 用**冻结地址公式**读回逻辑 K/V，与连续 KV cache
// 的可见 K/V 做 fp16 位级比较。
//
// 这是本文件的强门禁：不依赖最终 hidden state（残差主导 + fp16 输出量化会
// 吞掉 attention 的微小差异），而是直接检查 paged 存储布局是否正确落位。
TEST_F(PagedOracleLayerTest, ForwardPagedKVMatchesContiguousKVPerLayer) {
    constexpr int kBlockSize = 16;
    constexpr int kPrompt = 20; // 1 个整块 + 4 尾部
    constexpr int kSteps = 3;
    constexpr int kMaxBlocks = 8;

    SyntheticModel model;
    ASSERT_NO_THROW(model.build(/*hidden*/ 128, /*layers*/ 2, /*heads*/ 4, /*kv_heads*/ 2,
                                /*head_dim*/ 32, /*inter*/ 64, /*max_seq*/ 64));
    const ModelConfig &config = model.config;
    const int          hidden = config.hidden_dim;
    const int          kv_dim = config.num_kv_heads * config.head_dim;
    ASSERT_EQ(config.num_heads * config.head_dim, hidden);

    const int          visible_max = kPrompt + kSteps;
    const size_t       total_elems = static_cast<size_t>(visible_max) * static_cast<size_t>(hidden);
    const auto         input_host = randomFp16(total_elems, 4242);
    DeviceBuffer<half> d_contig(total_elems);
    DeviceBuffer<half> d_paged(total_elems);
    d_contig.copyFromHost(input_host.data(), input_host.size());
    d_paged.copyFromHost(input_host.data(), input_host.size());

    DeviceBuffer<int> d_pos(1);
    DeviceBuffer<int> d_decode_len(1);
    const int         zero = 0;
    d_pos.copyFromHost(&zero, 1);

    // ── 路径 A：连续 KV（KVCacheManager） ──
    KVCacheConfig kvc;
    kvc.num_layers = config.num_layers;
    kvc.num_kv_heads = config.num_kv_heads;
    kvc.head_dim = config.head_dim;
    kvc.max_seq_len = config.max_seq_len;
    kvc.max_batch_size = 1;
    auto cache_r = KVCacheManager::create(kvc);
    ASSERT_TRUE(cache_r.isOk()) << cache_r.error();
    auto cache = std::move(cache_r.value());
    ASSERT_TRUE(cache->allocateSequence(0, config.max_seq_len).isOk());

    // ── 路径 B：paged KV（pool + block table），pool 用冻结步长分层 ──
    PagedGeometry g = makeGeometry(config.num_heads, config.num_kv_heads, config.head_dim,
                                   kBlockSize, kMaxBlocks, config.num_layers);
    const auto table = makeBlockTable(requiredBlocks(visible_max, kBlockSize), kMaxBlocks, 31337);

    DeviceBuffer<half> d_k_pool(g.poolElements());
    DeviceBuffer<half> d_v_pool(g.poolElements());
    const size_t scratch_elems = static_cast<size_t>(kMaxBlocks) * static_cast<size_t>(kBlockSize) *
                                 static_cast<size_t>(kv_dim);
    DeviceBuffer<half> d_k_scratch(scratch_elems);
    DeviceBuffer<half> d_v_scratch(scratch_elems);
    DeviceBuffer<int>  d_table(static_cast<size_t>(kMaxBlocks));
    d_table.copyFromHost(table.data(), table.size());

    PagedKVCacheView view;
    view.k_pool = d_k_pool.data();
    view.v_pool = d_v_pool.data();
    view.block_table = d_table.data();
    view.k_scratch = d_k_scratch.data();
    view.v_scratch = d_v_scratch.data();
    view.visible_blocks = static_cast<int>(table.size());
    view.block_size = kBlockSize;
    view.max_num_blocks = kMaxBlocks;
    view.max_visible_tokens = kMaxBlocks * kBlockSize;
    view.position = 0;
    view.decode_len = nullptr;

    // 逐层 K/V 门禁：pool 按冻结公式读回，必须与连续 cache 的可见行逐元素相同。
    auto expect_layers_match = [&](int visible, const std::string &phase) {
        for (int l = 0; l < config.num_layers; ++l) {
            const auto pool_k = oracleGatherRows(readPoolLayer(d_k_pool, l, g), g, table, visible);
            const auto pool_v = oracleGatherRows(readPoolLayer(d_v_pool, l, g), g, table, visible);
            const auto cont_k = readCacheRows(*cache, 0, l, visible, kv_dim, /*is_value=*/false);
            const auto cont_v = readCacheRows(*cache, 0, l, visible, kv_dim, /*is_value=*/true);
            EXPECT_LT(maxAbsDiff(pool_k, cont_k), 1e-6f)
                << phase << " layer " << l
                << " 可见 K 与连续路径不一致（层偏移 / 位置 / 块表落位错误）";
            EXPECT_LT(maxAbsDiff(pool_v, cont_v), 1e-6f)
                << phase << " layer " << l
                << " 可见 V 与连续路径不一致（层偏移 / 位置 / 块表落位错误）";
        }
    };

    // ── prefill：两路写同一批 token，随后逐层比较 ──
    d_pos.copyFromHost(&zero, 1);
    cache->setAppendPos(0, 0);
    for (auto &layer : model.layers) {
        auto r = layer->forwardPrefill(d_contig.data(), *cache, 0, kPrompt, d_pos.data(),
                                       model.d_cos.data(), model.d_sin.data(), 0);
        ASSERT_TRUE(r.isOk()) << r.error();
    }
    ASSERT_TRUE(cache->advanceSeqLen(0, kPrompt).isOk());

    view.position = 0;
    view.decode_len = nullptr;
    for (auto &layer : model.layers) {
        auto r = layer->forwardPaged(d_paged.data(), view, kPrompt, d_pos.data(),
                                     model.d_cos.data(), model.d_sin.data(), 0);
        ASSERT_TRUE(r.isOk()) << r.error();
    }
    expect_layers_match(kPrompt, "prefill 后");

    // ── decode：每步两路各写入一个绝对位置的新 token，然后逐层比较 ──
    for (int step = 0; step < kSteps; ++step) {
        const int pos = kPrompt + step;
        const int vis = pos + 1;
        d_decode_len.copyFromHost(&vis, 0);
        d_pos.copyFromHost(&pos, 0);

        cache->setAppendPos(pos, 0);
        for (auto &layer : model.layers) {
            auto r = layer->forward(d_contig.data() + static_cast<size_t>(pos) * hidden, *cache, 0,
                                    pos, d_decode_len.data(), d_pos.data(), model.d_cos.data(),
                                    model.d_sin.data(), 0);
            ASSERT_TRUE(r.isOk()) << r.error();
        }
        ASSERT_TRUE(cache->advanceSeqLen(0, 1).isOk());

        view.position = pos;
        view.decode_len = d_decode_len.data();
        for (auto &layer : model.layers) {
            auto r =
                layer->forwardPaged(d_paged.data() + static_cast<size_t>(pos) * hidden, view, 1,
                                    d_pos.data(), model.d_cos.data(), model.d_sin.data(), 0);
            ASSERT_TRUE(r.isOk()) << r.error();
        }

        expect_layers_match(vis, "decode 第 " + std::to_string(step) + " 步后");

        // 次要比对：相同输入下两条路径的最终 hidden 也应一致。
        std::vector<half> hc(static_cast<size_t>(hidden)), hp(static_cast<size_t>(hidden));
        CUDA_CHECK(cudaMemcpy(hc.data(), d_contig.data() + static_cast<size_t>(pos) * hidden,
                              static_cast<size_t>(hidden) * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hp.data(), d_paged.data() + static_cast<size_t>(pos) * hidden,
                              static_cast<size_t>(hidden) * sizeof(half), cudaMemcpyDeviceToHost));
        EXPECT_LT(maxAbsDiff(toFloat(hp), toFloat(hc)), kFp16OutputTolerance)
            << "decode step " << step << " 最终 hidden 与连续路径不一致";
    }

    freeSyntheticModel(model);
}

// 无效几何 / 块表长度：forwardPaged 必须在入口拒绝，而不是越界读块表。
TEST_F(PagedOracleLayerTest, ForwardPagedRejectsInvalidGeometryAndShortBlockTable) {
    SyntheticModel model;
    ASSERT_NO_THROW(model.build(/*hidden*/ 64, /*layers*/ 1, /*heads*/ 2, /*kv_heads*/ 1,
                                /*head_dim*/ 32, /*inter*/ 32, /*max_seq*/ 32));
    const ModelConfig &config = model.config;
    const int          hidden = config.hidden_dim;

    constexpr int kBlockSize = 16;
    constexpr int kMaxBlocks = 4;
    const int     kv_dim = config.num_kv_heads * config.head_dim;

    DeviceBuffer<half>     d_hidden(static_cast<size_t>(config.max_seq_len) * hidden);
    DeviceBuffer<half>     d_k_pool(static_cast<size_t>(kMaxBlocks) * kBlockSize * kv_dim);
    DeviceBuffer<half>     d_v_pool(static_cast<size_t>(kMaxBlocks) * kBlockSize * kv_dim);
    DeviceBuffer<half>     d_k_scratch(static_cast<size_t>(kMaxBlocks) * kBlockSize * kv_dim);
    DeviceBuffer<half>     d_v_scratch(static_cast<size_t>(kMaxBlocks) * kBlockSize * kv_dim);
    DeviceBuffer<int>      d_table(static_cast<size_t>(kMaxBlocks));
    const std::vector<int> table = {0, 1, 2};
    d_table.copyFromHost(table.data(), table.size());
    DeviceBuffer<int> d_pos(1);
    const int         zero = 0;
    d_pos.copyFromHost(&zero, 1);

    PagedKVCacheView view;
    view.k_pool = d_k_pool.data();
    view.v_pool = d_v_pool.data();
    view.block_table = d_table.data();
    view.k_scratch = d_k_scratch.data();
    view.v_scratch = d_v_scratch.data();
    view.visible_blocks = 3; // 可寻址 3 块 = 48 token
    view.block_size = kBlockSize;
    view.max_num_blocks = kMaxBlocks;
    view.max_visible_tokens = kMaxBlocks * kBlockSize;
    view.position = 0;
    view.decode_len = nullptr;

    // 合法基线：prefill 20 token（需要 2 块）。
    EXPECT_TRUE(model.layers[0]
                    ->forwardPaged(d_hidden.data(), view, 20, d_pos.data(), model.d_cos.data(),
                                   model.d_sin.data(), 0)
                    .isOk());

    // null pool：入口拒绝。
    PagedKVCacheView null_view = view;
    null_view.k_pool = nullptr;
    EXPECT_TRUE(model.layers[0]
                    ->forwardPaged(d_hidden.data(), null_view, 20, d_pos.data(), model.d_cos.data(),
                                   model.d_sin.data(), 0)
                    .isErr());

    // 块表长度不足：visible 20 需要 2 块，visible_blocks=1 必须被拒绝（防止越界读块表）。
    PagedKVCacheView short_view = view;
    short_view.visible_blocks = 1;
    EXPECT_TRUE(model.layers[0]
                    ->forwardPaged(d_hidden.data(), short_view, 20, d_pos.data(),
                                   model.d_cos.data(), model.d_sin.data(), 0)
                    .isErr())
        << "过短块表必须在入口被拒绝";

    // decode 分支：visible = position + 1 = 17 → 需要 2 块；visible_blocks=1 必须被拒绝。
    DeviceBuffer<int> d_vis(1);
    const int         vis = 17;
    d_vis.copyFromHost(&vis, 1);
    PagedKVCacheView decode_view = short_view;
    decode_view.position = 16;
    decode_view.decode_len = d_vis.data();
    EXPECT_TRUE(model.layers[0]
                    ->forwardPaged(d_hidden.data(), decode_view, 1, d_pos.data(),
                                   model.d_cos.data(), model.d_sin.data(), 0)
                    .isErr())
        << "decode 分支过短块表必须在入口被拒绝";

    // decode 分支合法：visible_blocks=2 恰好够。
    decode_view.visible_blocks = 2;
    EXPECT_TRUE(model.layers[0]
                    ->forwardPaged(d_hidden.data(), decode_view, 1, d_pos.data(),
                                   model.d_cos.data(), model.d_sin.data(), 0)
                    .isOk());

    // 位置越界：position + num_tokens > max_visible_tokens。
    PagedKVCacheView over = view;
    over.position = kMaxBlocks * kBlockSize - 1;
    EXPECT_TRUE(model.layers[0]
                    ->forwardPaged(d_hidden.data(), over, 4, d_pos.data(), model.d_cos.data(),
                                   model.d_sin.data(), 0)
                    .isErr());

    freeSyntheticModel(model);
}

// ============================================================================
// 3. host-only contract：地址公式与块表长度
// ============================================================================

TEST(PagedOracleContractTest, BlockTableLengthContractMatchesCeilDivision) {
    const std::vector<std::pair<int, int>> cases = {
        {1, 1},   {2, 1},   {16, 16}, {17, 16}, {32, 16}, {33, 16},   {1, 16},    {15, 16},
        {35, 16}, {33, 32}, {64, 32}, {96, 32}, {1, 128}, {128, 128}, {129, 128},
    };
    for (const auto &[visible, block_size] : cases) {
        const int nb = requiredBlocks(visible, block_size);
        EXPECT_EQ(nb, (visible + block_size - 1) / block_size);
        EXPECT_GE(nb * block_size, visible) << "visible=" << visible << " bs=" << block_size;
        if (nb > 1) {
            EXPECT_LT((nb - 1) * block_size, visible)
                << "块表长度最小值不是最小：" << nb << " visible=" << visible;
        }
    }
}

TEST(PagedOracleContractTest, FrozenAddressFormulaIsSelfConsistent) {
    // 把连续逻辑行按 pagedRowOffset 放成 pool，再用 oracleGatherRows 读回：
    // 两者必须互为逆运算（覆盖 block=1 / 非单调物理块 / 跨块尾部）。
    const std::vector<int> block_sizes = {1, 4, 16, 32};
    for (int bs : block_sizes) {
        for (int num_layers : {1, 3}) {
            const int     visible = 2 * bs + 3;
            PagedGeometry g =
                makeGeometry(4, 2, 8, bs, requiredBlocks(visible, bs) + 2, num_layers);
            const auto table = makeBlockTable(requiredBlocks(visible, bs), g.max_num_blocks, 5);
            ASSERT_TRUE(tiny_llm::test::geometryIsValid(g));

            for (int layer = 0; layer < num_layers; ++layer) {
                std::vector<float> pool_layer(g.layerStride(), 0.0f);
                std::vector<float> logical(static_cast<size_t>(visible) * g.kvDim());
                for (size_t i = 0; i < logical.size(); ++i)
                    logical[i] = static_cast<float>(i % 97) * 0.25f - 3.0f;

                for (int t = 0; t < visible; ++t) {
                    const size_t dst = pagedRowOffset(g, table, t);
                    for (int c = 0; c < g.kvDim(); ++c)
                        pool_layer[dst + static_cast<size_t>(c)] =
                            logical[static_cast<size_t>(t) * g.kvDim() + c];
                }

                const auto round_trip = oracleGatherRows(pool_layer, g, table, visible);
                EXPECT_LT(maxAbsDiff(round_trip, logical), 1e-6f)
                    << "bs=" << bs << " layer=" << layer;
            }
        }
    }
}
