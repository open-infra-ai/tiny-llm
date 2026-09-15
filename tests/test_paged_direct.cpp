// TLLM-P0-004 PR-2：direct paged decode attention 的 kernel 级差分门禁。
//
// 三层验证（与设计包 §8 对应）：
//   1. direct vs legacy **逐元素相等**：legacy = 生产 scatter + gather + attention_decode；
//      direct = attention_decode_paged。两者消费同一份物理 pool 与块表，语义等价，
//      因此输出必须**逐位相同**——这是定位寻址错误的主门禁，比容差比较强得多。
//   2. direct vs 独立 oracle（容差）：oracle 来自 TLLM-P0-002
//      （tests/paged_attention_oracle.h），不调用任何生产 kernel，因此能独立发现
//      "两条生产路径一起错"的情况。
//   3. 边界：非法块 id、visible=0、块表长度不足、多 layer pool offset。
//
// 边界说明：被测的是 kernel 级接口；Transformer 的 dispatch/fallback 属于 PR-3，
// 不在本文件范围内。本文件不产生任何性能数字。

#include "attention.cuh"
#include "paged_attention_oracle.h"
#include "paged_kv.cuh"
#include "tiny_llm/cuda_utils.h"

#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace tiny_llm;
using namespace tiny_llm::kernels;
using tiny_llm::test::maxAbsDiff;
using tiny_llm::test::oracleAttentionDecode;
using tiny_llm::test::PagedGeometry;
using tiny_llm::test::requiredBlocks;

namespace {

constexpr float kOracleTolerance = 2e-3f;

bool hasCudaDevice() {
    static bool checked = false;
    static bool has_device = false;
    if (!checked) {
        int         n = 0;
        cudaError_t e = cudaGetDeviceCount(&n);
        has_device = (e == cudaSuccess && n > 0);
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

// 逐位比较：fp16 输出的原始位模式必须完全一致（比数值比较严格）。
bool bitwiseEqual(const std::vector<half> &a, const std::vector<half> &b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(half)) == 0;
}

PagedGeometry makeGeometry(int hq, int hkv, int hd, int bs, int max_blocks, int layers = 1) {
    PagedGeometry g;
    g.num_q_heads = hq;
    g.num_kv_heads = hkv;
    g.head_dim = hd;
    g.block_size = bs;
    g.max_num_blocks = max_blocks;
    g.num_layers = layers;
    return g;
}

std::vector<int> makeTable(int n, int max_blocks, unsigned seed) {
    std::vector<int> ids(static_cast<size_t>(max_blocks));
    for (int i = 0; i < max_blocks; ++i)
        ids[static_cast<size_t>(i)] = i;
    std::mt19937 gen(seed);
    std::shuffle(ids.begin(), ids.end(), gen);
    return std::vector<int>(ids.begin(), ids.begin() + n);
}

// 同一份 pool 上跑两条生产路径，返回 (legacy, direct) 的 fp16 输出。
struct BothPaths {
    std::vector<half> legacy;
    std::vector<half> direct;
};

class PagedDirectFixture {
  public:
    PagedDirectFixture(const PagedGeometry &g, const std::vector<int> &table, int table_len)
        : g_(g), table_(table), table_len_(table_len), d_k_pool_(g.poolElements()),
          d_v_pool_(g.poolElements()),
          d_k_scratch_(static_cast<size_t>(g.max_num_blocks) * static_cast<size_t>(g.block_size) *
                       static_cast<size_t>(g.kvDim())),
          d_v_scratch_(static_cast<size_t>(g.max_num_blocks) * static_cast<size_t>(g.block_size) *
                       static_cast<size_t>(g.kvDim())),
          d_table_(static_cast<size_t>(g.max_num_blocks)), d_len_(1) {
        d_table_.copyFromHost(table_.data(), table_.size());
    }

    void scatter(int layer, const std::vector<half> &k, const std::vector<half> &v, int tokens,
                 int position) {
        const size_t       elems = static_cast<size_t>(tokens) * g_.kvDim();
        DeviceBuffer<half> d_k(elems), d_v(elems);
        d_k.copyFromHost(k.data(), elems);
        d_v.copyFromHost(v.data(), elems);
        half *kp = d_k_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        half *vp = d_v_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        paged_scatter_blocks(d_k.data(), kp, d_table_.data(), tokens, position, g_.block_size,
                             g_.kvDim(), g_.max_num_blocks);
        paged_scatter_blocks(d_v.data(), vp, d_table_.data(), tokens, position, g_.block_size,
                             g_.kvDim(), g_.max_num_blocks);
    }

    BothPaths run(int layer, const std::vector<half> &q, int visible) {
        const size_t       q_elems = static_cast<size_t>(g_.num_q_heads) * g_.head_dim;
        DeviceBuffer<half> d_q(q_elems), d_legacy(q_elems), d_direct(q_elems);
        d_q.copyFromHost(q.data(), q_elems);
        d_len_.copyFromHost(&visible, 1);

        const half *kp = d_k_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        const half *vp = d_v_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        const float scale = 1.0f / std::sqrt(static_cast<float>(g_.head_dim));

        // legacy：先 gather 成连续 scratch，再走连续 attention
        paged_gather_blocks(d_k_scratch_.data(), kp, d_table_.data(), visible, g_.block_size,
                            g_.kvDim(), g_.max_num_blocks);
        paged_gather_blocks(d_v_scratch_.data(), vp, d_table_.data(), visible, g_.block_size,
                            g_.kvDim(), g_.max_num_blocks);
        attention_decode(d_q.data(), d_k_scratch_.data(), d_v_scratch_.data(), d_legacy.data(),
                         scale, g_.num_q_heads, g_.num_kv_heads, d_len_.data(), g_.head_dim);

        // direct：直接从 pool 寻址
        attention_decode_paged(d_q.data(), kp, vp, d_table_.data(), d_direct.data(), scale,
                               g_.num_q_heads, g_.num_kv_heads, g_.head_dim, d_len_.data(),
                               g_.block_size, g_.max_num_blocks, table_len_);

        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        BothPaths out;
        out.legacy.resize(q_elems);
        out.direct.resize(q_elems);
        d_legacy.copyToHost(out.legacy.data(), q_elems);
        d_direct.copyToHost(out.direct.data(), q_elems);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return out;
    }

    // 只跑 direct 路径。用于 legacy 在该输入上属于未定义行为（块表长度不足）的场景。
    std::vector<half> runDirectOnly(int layer, const std::vector<half> &q, int visible,
                                    int table_len) {
        const size_t       q_elems = static_cast<size_t>(g_.num_q_heads) * g_.head_dim;
        DeviceBuffer<half> d_q(q_elems), d_out(q_elems);
        d_q.copyFromHost(q.data(), q_elems);
        d_len_.copyFromHost(&visible, 1);

        const half *kp = d_k_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        const half *vp = d_v_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        const float scale = 1.0f / std::sqrt(static_cast<float>(g_.head_dim));

        attention_decode_paged(d_q.data(), kp, vp, d_table_.data(), d_out.data(), scale,
                               g_.num_q_heads, g_.num_kv_heads, g_.head_dim, d_len_.data(),
                               g_.block_size, g_.max_num_blocks, table_len);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<half> out(q_elems);
        d_out.copyToHost(out.data(), q_elems);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return out;
    }

  private:
    PagedGeometry      g_;
    std::vector<int>   table_;
    int                table_len_;
    DeviceBuffer<half> d_k_pool_;
    DeviceBuffer<half> d_v_pool_;
    DeviceBuffer<half> d_k_scratch_;
    DeviceBuffer<half> d_v_scratch_;
    DeviceBuffer<int>  d_table_;
    DeviceBuffer<int>  d_len_;
};

struct MatrixCase {
    int         block_size;
    int         visible;
    int         hq;
    int         hkv;
    int         head_dim;
    const char *label;
};

} // namespace

class PagedDirectTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";
        cudaSetDevice(0);
    }
    void TearDown() override { cudaDeviceSynchronize(); }
};

// ── 主门禁：direct vs legacy 逐元素相等 + direct vs 独立 oracle ────────────
TEST_F(PagedDirectTest, DirectMatchesLegacyBitwiseAndOracle) {
    const std::vector<MatrixCase> cases = {
        {1, 1, 4, 4, 32, "block=1 MHA 单 token"},
        {1, 17, 4, 2, 32, "block=1 GQA 跨块"},
        {16, 1, 4, 4, 32, "block=16 单 token"},
        {16, 15, 4, 4, 32, "block=16 block-1"},
        {16, 16, 4, 2, 32, "block=16 恰好一块 GQA"},
        {16, 17, 8, 2, 64, "block=16 block+1 GQA"},
        {16, 35, 8, 1, 64, "block=16 2*block+tail MQA"},
        {32, 33, 4, 4, 128, "block=32 跨块尾部 D=128"},
        {32, 64, 8, 4, 32, "block=32 两块"},
        {32, 96, 14, 2, 64, "block=32 三块 生产形状"},
        {128, 129, 4, 2, 64, "跨 tile 边界 (ATTEND_TILE=128)"},
        {64, 200, 14, 2, 128, "多 tile D=128"},
    };

    for (const auto &c : cases) {
        const int nb = requiredBlocks(c.visible, c.block_size);
        for (unsigned seed : {1u, 7u, 1234u}) {
            SCOPED_TRACE(std::string(c.label) + " seed=" + std::to_string(seed));
            PagedGeometry g = makeGeometry(c.hq, c.hkv, c.head_dim, c.block_size, nb + 3);
            ASSERT_TRUE(tiny_llm::test::geometryIsValid(g));

            const size_t k_elems = static_cast<size_t>(c.visible) * g.kvDim();
            const size_t q_elems = static_cast<size_t>(g.num_q_heads) * g.head_dim;
            const auto   k_logical = randomFp16(k_elems, seed * 10 + 1);
            const auto   v_logical = randomFp16(k_elems, seed * 10 + 2);
            const auto   q = randomFp16(q_elems, seed * 10 + 3);
            const auto   table = makeTable(nb, g.max_num_blocks, seed * 10 + 4);

            PagedDirectFixture fx(g, table, /*table_len=*/nb);
            fx.scatter(0, k_logical, v_logical, c.visible, 0);
            const auto out = fx.run(0, q, c.visible);

            // (1) 主门禁：两条生产路径逐位相同
            EXPECT_TRUE(bitwiseEqual(out.direct, out.legacy))
                << "direct 与 legacy 输出不逐位相同（寻址错误或两份实现数值漂移）"
                << " max|diff|=" << maxAbsDiff(toFloat(out.direct), toFloat(out.legacy));

            // (2) 独立 oracle（容差）：发现"两条生产路径一起错"的情况
            const auto ref = oracleAttentionDecode(toFloat(q), toFloat(k_logical),
                                                   toFloat(v_logical), c.visible, g);
            EXPECT_LT(maxAbsDiff(toFloat(out.direct), ref), kOracleTolerance);
        }
    }
}

// ── 非法块 id：两条路径都必须按零行处理且不 fault ─────────────────────────
TEST_F(PagedDirectTest, InvalidBlockIdsMatchLegacyZeroRows) {
    const int     block_size = 16;
    const int     visible = 20;
    const int     nb = requiredBlocks(visible, block_size);
    PagedGeometry g = makeGeometry(4, 4, 32, block_size, nb + 1);

    const size_t k_elems = static_cast<size_t>(visible) * g.kvDim();
    const auto   k_logical = randomFp16(k_elems, 900);
    const auto   v_logical = randomFp16(k_elems, 901);
    const auto   q = randomFp16(static_cast<size_t>(g.num_q_heads) * g.head_dim, 902);

    // 第二块越界（负值与 == max_num_blocks）
    for (int bad : {-1, g.max_num_blocks}) {
        SCOPED_TRACE("bad block id = " + std::to_string(bad));
        std::vector<int>   table = {1, bad};
        PagedDirectFixture fx(g, table, nb);
        fx.scatter(0, k_logical, v_logical, visible, 0);
        const auto out = fx.run(0, q, visible);

        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "非法块 id 不得触发 illegal address";
        EXPECT_TRUE(bitwiseEqual(out.direct, out.legacy));

        // 零行语义的独立确认：非法块覆盖的 token 贡献为 0，但仍参与 softmax
        std::vector<float> k_eff = toFloat(k_logical);
        std::vector<float> v_eff = toFloat(v_logical);
        for (int t = block_size; t < visible; ++t) {
            for (int c2 = 0; c2 < g.kvDim(); ++c2) {
                k_eff[static_cast<size_t>(t) * g.kvDim() + c2] = 0.0f;
                v_eff[static_cast<size_t>(t) * g.kvDim() + c2] = 0.0f;
            }
        }
        const auto ref = oracleAttentionDecode(toFloat(q), k_eff, v_eff, visible, g);
        EXPECT_LT(maxAbsDiff(toFloat(out.direct), ref), kOracleTolerance);
    }
}

// ── visible = 0：输出全 0，两条路径一致 ──────────────────────────────────
TEST_F(PagedDirectTest, EmptyVisibleWindowMatchesLegacy) {
    PagedGeometry g = makeGeometry(4, 2, 64, 16, 4);
    const auto    q = randomFp16(static_cast<size_t>(g.num_q_heads) * g.head_dim, 77);
    const auto    table = makeTable(1, g.max_num_blocks, 78);

    PagedDirectFixture fx(g, table, 1);
    const auto         out = fx.run(0, q, /*visible=*/0);

    EXPECT_TRUE(bitwiseEqual(out.direct, out.legacy));
    for (half h : out.direct)
        EXPECT_EQ(__half2float(h), 0.0f);
}

// ── 块表长度不足：direct 按零行处理且不越界读块表（legacy 此处是 UB，故只测 direct）──
TEST_F(PagedDirectTest, ShortBlockTableIsGuardedWithoutFault) {
    const int     block_size = 16;
    const int     visible = 32; // 需要 2 块
    PagedGeometry g = makeGeometry(4, 2, 64, block_size, 4);

    const auto table = makeTable(2, g.max_num_blocks, 55);
    const auto q = randomFp16(static_cast<size_t>(g.num_q_heads) * g.head_dim, 56);
    const auto k_logical = randomFp16(static_cast<size_t>(visible) * g.kvDim(), 57);
    const auto v_logical = randomFp16(static_cast<size_t>(visible) * g.kvDim(), 58);

    // table_len = 1，但 visible = 32 需要 2 块 → 第二个块必须按无效（零行）处理。
    // 注意 scatter 没有 table_len 参数，所以物理 pool 里其实写了第二块的内容；
    // 期望结果因此是"第二块覆盖的 token 视为零行"的独立参考。
    PagedDirectFixture fx(g, table, /*table_len=*/1);
    fx.scatter(0, k_logical, v_logical, visible, 0);

    const auto direct = fx.runDirectOnly(0, q, visible, /*table_len=*/1);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "块表长度不足必须按零行处理，不得越界读块表";

    std::vector<float> k_eff = toFloat(k_logical);
    std::vector<float> v_eff = toFloat(v_logical);
    for (int t = block_size; t < visible; ++t) {
        for (int c = 0; c < g.kvDim(); ++c) {
            k_eff[static_cast<size_t>(t) * g.kvDim() + c] = 0.0f;
            v_eff[static_cast<size_t>(t) * g.kvDim() + c] = 0.0f;
        }
    }
    const auto ref = oracleAttentionDecode(toFloat(q), k_eff, v_eff, visible, g);
    EXPECT_LT(maxAbsDiff(toFloat(direct), ref), kOracleTolerance)
        << "table_len 之外的 token 必须按零行参与，而不是读取 pool 内容";
}

// ── 多 layer pool offset：caller 计算的 layer 偏移必须可寻址且互不污染 ─────
TEST_F(PagedDirectTest, MultiLayerPoolOffsetIsAddressable) {
    constexpr int kLayers = 3;
    const int     block_size = 16;
    const int     visible = 20;
    const int     nb = requiredBlocks(visible, block_size);
    PagedGeometry g = makeGeometry(4, 2, 32, block_size, nb + 1, kLayers);

    const auto         table = makeTable(nb, g.max_num_blocks, 31337);
    PagedDirectFixture fx(g, table, nb);

    std::vector<std::vector<half>> k(kLayers), v(kLayers), q(kLayers);
    for (int l = 0; l < kLayers; ++l) {
        k[static_cast<size_t>(l)] =
            randomFp16(static_cast<size_t>(visible) * g.kvDim(), 200u + static_cast<unsigned>(l));
        v[static_cast<size_t>(l)] =
            randomFp16(static_cast<size_t>(visible) * g.kvDim(), 300u + static_cast<unsigned>(l));
        q[static_cast<size_t>(l)] = randomFp16(static_cast<size_t>(g.num_q_heads) * g.head_dim,
                                               400u + static_cast<unsigned>(l));
        fx.scatter(l, k[static_cast<size_t>(l)], v[static_cast<size_t>(l)], visible, 0);
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (int l = 0; l < kLayers; ++l) {
        SCOPED_TRACE("layer " + std::to_string(l));
        const auto out = fx.run(l, q[static_cast<size_t>(l)], visible);
        EXPECT_TRUE(bitwiseEqual(out.direct, out.legacy));
        const auto ref = oracleAttentionDecode(toFloat(q[static_cast<size_t>(l)]),
                                               toFloat(k[static_cast<size_t>(l)]),
                                               toFloat(v[static_cast<size_t>(l)]), visible, g);
        EXPECT_LT(maxAbsDiff(toFloat(out.direct), ref), kOracleTolerance);
    }
}
