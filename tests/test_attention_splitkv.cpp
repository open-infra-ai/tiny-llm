// TLLM-ATTN-SPLITKV：split-KV decode attention 的 kernel 级门禁。
//
// 设计包 §8 的矩阵，按强度排序：
//   1. **逐位锚点**：num_splits == 1 时，split 路径与单遍路径输出逐位相同
//      （连续与分页各一次）。这是把"重构风险"与"数值改动"分开的那条门禁；
//   2. **独立 oracle**：num_splits > 1 时落在 TLLM-P0-002 oracle 的容差内
//      （oracle 不调用任何生产 kernel，能发现"两条生产路径一起错"）；
//   3. **同归约、两种取址**：split 连续 vs split 分页逐位相同；
//   4. 边界：visible = 0、visible < num_splits（空段）、非整数倍、非法块 id、短块表；
//   5. **CUDA Graph**：捕获后改变 device 端可见长度并 replay，与非 graph 结果逐位相同
//      ——直接证明"grid 固定 + 段范围走 device int"这一设计前提成立。
//
// 边界：只测 kernel 级接口；LayerWorkspace 接线与开关属于 PR-C，不在本文件范围。
// 本文件不产生任何性能数字。

#include "attention.cuh"
#include "paged_attention_oracle.h"
#include "paged_kv.cuh"
#include "tiny_llm/cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
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
// partial 工作区按最大 split 数分配；测试里只用到 ≤ 16。
constexpr int kMaxSplits = 32;

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

// 一个几何 + pool + 块表 + scratch + partial 工作区的固定装置。
// 三条路径消费同一份物理 pool：单遍（gather + attention_decode）、split 连续、split 分页。
class SplitKvFixture {
  public:
    SplitKvFixture(const PagedGeometry &g, const std::vector<int> &table, int table_len)
        : g_(g), table_len_(table_len), d_k_pool_(g.poolElements()), d_v_pool_(g.poolElements()),
          d_k_scratch_(static_cast<size_t>(g.max_num_blocks) * static_cast<size_t>(g.block_size) *
                       static_cast<size_t>(g.kvDim())),
          d_v_scratch_(static_cast<size_t>(g.max_num_blocks) * static_cast<size_t>(g.block_size) *
                       static_cast<size_t>(g.kvDim())),
          d_partial_(static_cast<size_t>(g.num_q_heads) * kMaxSplits *
                     (2 + static_cast<size_t>(g.head_dim))),
          d_table_(static_cast<size_t>(g.max_num_blocks)), d_len_(1) {
        d_table_.copyFromHost(table.data(), table.size());
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

    // 单遍连续路径：gather 到 scratch 后调用既有 attention_decode（今天的行为）。
    std::vector<half> runSinglePassContiguous(int layer, const std::vector<half> &q, int visible,
                                              cudaStream_t stream = 0) {
        const size_t       q_elems = static_cast<size_t>(g_.num_q_heads) * g_.head_dim;
        DeviceBuffer<half> d_q(q_elems), d_out(q_elems);
        d_q.copyFromHost(q.data(), q_elems);
        d_len_.copyFromHost(&visible, 1, stream);

        const half *kp = d_k_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        const half *vp = d_v_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        paged_gather_blocks(d_k_scratch_.data(), kp, d_table_.data(), visible, g_.block_size,
                            g_.kvDim(), g_.max_num_blocks, stream);
        paged_gather_blocks(d_v_scratch_.data(), vp, d_table_.data(), visible, g_.block_size,
                            g_.kvDim(), g_.max_num_blocks, stream);
        attention_decode(d_q.data(), d_k_scratch_.data(), d_v_scratch_.data(), d_out.data(),
                         scale(), g_.num_q_heads, g_.num_kv_heads, d_len_.data(), g_.head_dim,
                         stream);
        return readBack(d_out, q_elems);
    }

    // 单遍分页路径（PR-2 的 direct kernel，今天的行为）。
    std::vector<half> runSinglePassPaged(int layer, const std::vector<half> &q, int visible,
                                         cudaStream_t stream = 0) {
        const size_t       q_elems = static_cast<size_t>(g_.num_q_heads) * g_.head_dim;
        DeviceBuffer<half> d_q(q_elems), d_out(q_elems);
        d_q.copyFromHost(q.data(), q_elems);
        d_len_.copyFromHost(&visible, 1, stream);

        const half *kp = d_k_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        const half *vp = d_v_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        attention_decode_paged(d_q.data(), kp, vp, d_table_.data(), d_out.data(), scale(),
                               g_.num_q_heads, g_.num_kv_heads, g_.head_dim, d_len_.data(),
                               g_.block_size, g_.max_num_blocks, table_len_, stream);
        return readBack(d_out, q_elems);
    }

    // split 连续路径：先 gather（复用单遍的 scratch），再走 split-KV。
    std::vector<half> runSplitContiguous(int layer, const std::vector<half> &q, int visible,
                                         int num_splits, cudaStream_t stream = 0) {
        const size_t       q_elems = static_cast<size_t>(g_.num_q_heads) * g_.head_dim;
        DeviceBuffer<half> d_q(q_elems), d_out(q_elems);
        d_q.copyFromHost(q.data(), q_elems);
        d_len_.copyFromHost(&visible, 1, stream);

        const half *kp = d_k_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        const half *vp = d_v_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        paged_gather_blocks(d_k_scratch_.data(), kp, d_table_.data(), visible, g_.block_size,
                            g_.kvDim(), g_.max_num_blocks, stream);
        paged_gather_blocks(d_v_scratch_.data(), vp, d_table_.data(), visible, g_.block_size,
                            g_.kvDim(), g_.max_num_blocks, stream);
        attention_decode_splitkv(d_q.data(), d_k_scratch_.data(), d_v_scratch_.data(), d_out.data(),
                                 scale(), g_.num_q_heads, g_.num_kv_heads, d_len_.data(),
                                 g_.head_dim, d_partial_.data(), num_splits, stream);
        return readBack(d_out, q_elems);
    }

    // split 分页路径：直接从 pool 寻址。
    std::vector<half> runSplitPaged(int layer, const std::vector<half> &q, int visible,
                                    int num_splits, cudaStream_t stream = 0) {
        const size_t       q_elems = static_cast<size_t>(g_.num_q_heads) * g_.head_dim;
        DeviceBuffer<half> d_q(q_elems), d_out(q_elems);
        d_q.copyFromHost(q.data(), q_elems);
        d_len_.copyFromHost(&visible, 1, stream);

        const half *kp = d_k_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        const half *vp = d_v_pool_.data() + static_cast<size_t>(layer) * g_.layerStride();
        attention_decode_paged_splitkv(d_q.data(), kp, vp, d_table_.data(), d_out.data(), scale(),
                                       g_.num_q_heads, g_.num_kv_heads, g_.head_dim, d_len_.data(),
                                       g_.block_size, g_.max_num_blocks, table_len_,
                                       d_partial_.data(), num_splits, stream);
        return readBack(d_out, q_elems);
    }

    // 供 CUDA Graph 测试直接拿 device 指针。
    half  *kPoolData() { return d_k_pool_.data(); }
    half  *vPoolData() { return d_v_pool_.data(); }
    int   *tableData() { return d_table_.data(); }
    float *partialData() { return d_partial_.data(); }
    int   *lenData() { return d_len_.data(); }
    int    tableLen() const { return table_len_; }
    float  scale() const { return 1.0f / std::sqrt(static_cast<float>(g_.head_dim)); }
    size_t qElems() const {
        return static_cast<size_t>(g_.num_q_heads) * static_cast<size_t>(g_.head_dim);
    }
    void setLen(int visible, cudaStream_t stream = 0) { d_len_.copyFromHost(&visible, 1, stream); }
    void sync() { ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess); }

  private:
    std::vector<half> readBack(DeviceBuffer<half> &d_out, size_t q_elems) {
        std::vector<half> out(q_elems);
        d_out.copyToHost(out.data(), q_elems);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return out;
    }

    PagedGeometry       g_;
    int                 table_len_;
    DeviceBuffer<half>  d_k_pool_;
    DeviceBuffer<half>  d_v_pool_;
    DeviceBuffer<half>  d_k_scratch_;
    DeviceBuffer<half>  d_v_scratch_;
    DeviceBuffer<float> d_partial_;
    DeviceBuffer<int>   d_table_;
    DeviceBuffer<int>   d_len_;
};

// 构造 fixture 并写入逻辑 K/V（按块表 scatter 进 pool）。
struct Built {
    std::unique_ptr<SplitKvFixture> fx;
    std::vector<half>               k_logical, v_logical, q;
};

Built buildCase(const PagedGeometry &g, int visible, unsigned seed) {
    const int    nb = requiredBlocks(visible, g.block_size);
    const auto   table = makeTable(nb, g.max_num_blocks, seed * 10 + 4);
    const size_t k_elems = static_cast<size_t>(visible) * g.kvDim();
    const size_t q_elems = static_cast<size_t>(g.num_q_heads) * g.head_dim;

    Built b;
    b.fx = std::make_unique<SplitKvFixture>(g, table, nb);
    b.k_logical = randomFp16(k_elems, seed * 10 + 1);
    b.v_logical = randomFp16(k_elems, seed * 10 + 2);
    b.q = randomFp16(q_elems, seed * 10 + 3);
    b.fx->scatter(0, b.k_logical, b.v_logical, visible, 0);
    b.fx->sync();
    return b;
}

} // namespace

class SplitKvTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";
        cudaSetDevice(0);
    }
    void TearDown() override { cudaDeviceSynchronize(); }
};

// ── 1. 逐位锚点（连续）──────────────────────────────────────────────────────
TEST_F(SplitKvTest, NumSplitsOneIsBitwiseIdenticalToSinglePassContiguous) {
    struct Case {
        int         bs, visible, hq, hkv, hd;
        const char *label;
    };
    const std::vector<Case> cases = {
        {16, 1, 4, 4, 32, "S=1"},
        {16, 17, 8, 2, 64, "跨块 GQA"},
        {16, 20, 14, 2, 64, "生产形状"},
        {32, 96, 14, 2, 64, "三块 生产形状"},
        {32, 129, 4, 2, 64, "跨 ATTEND_TILE"},
        {16, 200, 4, 4, 128, "多 tile D=128"},
        {8, 5, 8, 1, 64, "MQA"},
    };

    for (const auto &c : cases) {
        for (unsigned seed : {1u, 99u}) {
            SCOPED_TRACE(std::string(c.label) + " seed=" + std::to_string(seed));
            const int     nb = requiredBlocks(c.visible, c.bs);
            PagedGeometry g = makeGeometry(c.hq, c.hkv, c.hd, c.bs, nb + 3);
            ASSERT_TRUE(tiny_llm::test::geometryIsValid(g));
            auto b = buildCase(g, c.visible, seed);

            const auto single = b.fx->runSinglePassContiguous(0, b.q, c.visible);
            const auto split1 = b.fx->runSplitContiguous(0, b.q, c.visible, /*num_splits=*/1);

            EXPECT_TRUE(bitwiseEqual(single, split1))
                << "num_splits=1 必须与单遍路径逐位相同（重构锚点），max|diff|="
                << maxAbsDiff(toFloat(single), toFloat(split1));
        }
    }
}

// ── 1b. 逐位锚点（分页）────────────────────────────────────────────────────
TEST_F(SplitKvTest, NumSplitsOneIsBitwiseIdenticalToSinglePassPaged) {
    struct Case {
        int         bs, visible, hq, hkv, hd;
        const char *label;
    };
    const std::vector<Case> cases = {
        {1, 1, 4, 4, 32, "block=1"},     {1, 17, 4, 2, 32, "block=1 跨块"},
        {16, 16, 4, 2, 32, "恰好一块"},  {16, 35, 8, 1, 64, "2*block+tail MQA"},
        {32, 96, 14, 2, 64, "生产形状"}, {32, 129, 4, 2, 64, "跨 ATTEND_TILE"},
    };

    for (const auto &c : cases) {
        for (unsigned seed : {7u, 4321u}) {
            SCOPED_TRACE(std::string(c.label) + " seed=" + std::to_string(seed));
            const int     nb = requiredBlocks(c.visible, c.bs);
            PagedGeometry g = makeGeometry(c.hq, c.hkv, c.hd, c.bs, nb + 3);
            ASSERT_TRUE(tiny_llm::test::geometryIsValid(g));
            auto b = buildCase(g, c.visible, seed);

            const auto single = b.fx->runSinglePassPaged(0, b.q, c.visible);
            const auto split1 = b.fx->runSplitPaged(0, b.q, c.visible, /*num_splits=*/1);

            EXPECT_TRUE(bitwiseEqual(single, split1))
                << "num_splits=1 必须与单遍 paged 路径逐位相同，max|diff|="
                << maxAbsDiff(toFloat(single), toFloat(split1));
        }
    }
}

// ── 2. num_splits > 1：对独立 oracle 的容差 + 不产生 NaN ────────────────────
TEST_F(SplitKvTest, SplitMatchesIndependentOracle) {
    struct Case {
        int         bs, visible, hq, hkv, hd;
        const char *label;
    };
    const std::vector<Case> cases = {
        {16, 20, 14, 2, 64, "生产形状"},   {32, 96, 14, 2, 64, "三块 生产形状"},
        {16, 33, 8, 2, 64, "block+1 GQA"}, {16, 128, 4, 4, 32, "整 tile 边界"},
        {16, 129, 4, 2, 64, "跨 tile"},    {32, 200, 4, 4, 128, "多 tile D=128"},
        {8, 40, 8, 1, 64, "MQA"},
    };

    for (const auto &c : cases) {
        for (int num_splits : {2, 4, 8, 16}) {
            SCOPED_TRACE(std::string(c.label) + " num_splits=" + std::to_string(num_splits));
            const int     nb = requiredBlocks(c.visible, c.bs);
            PagedGeometry g = makeGeometry(c.hq, c.hkv, c.hd, c.bs, nb + 3);
            ASSERT_TRUE(tiny_llm::test::geometryIsValid(g));
            auto b = buildCase(g, c.visible, 1234u);

            const auto out = b.fx->runSplitPaged(0, b.q, c.visible, num_splits);
            const auto got = toFloat(out);
            for (float x : got)
                EXPECT_TRUE(std::isfinite(x)) << "split 输出出现非有限值";

            const auto ref = oracleAttentionDecode(toFloat(b.q), toFloat(b.k_logical),
                                                   toFloat(b.v_logical), c.visible, g);
            EXPECT_LT(maxAbsDiff(got, ref), kOracleTolerance)
                << "split 与独立 oracle 的偏差超出容差，max|diff|=" << maxAbsDiff(got, ref);
        }
    }
}

// ── 3. split 连续 vs split 分页：同一份 pool 上逐位相同 ─────────────────────
TEST_F(SplitKvTest, SplitContiguousAndPagedAgreeBitwise) {
    struct Case {
        int         bs, visible, hq, hkv, hd;
        const char *label;
    };
    const std::vector<Case> cases = {
        {16, 20, 14, 2, 64, "生产形状"},
        {1, 17, 4, 2, 32, "block=1 跨块"},
        {16, 129, 4, 2, 64, "跨 tile"},
        {32, 96, 8, 1, 64, "MQA"},
    };

    for (const auto &c : cases) {
        for (int num_splits : {1, 2, 8}) {
            SCOPED_TRACE(std::string(c.label) + " num_splits=" + std::to_string(num_splits));
            const int     nb = requiredBlocks(c.visible, c.bs);
            PagedGeometry g = makeGeometry(c.hq, c.hkv, c.hd, c.bs, nb + 3);
            ASSERT_TRUE(tiny_llm::test::geometryIsValid(g));
            auto b = buildCase(g, c.visible, 555u);

            const auto contig = b.fx->runSplitContiguous(0, b.q, c.visible, num_splits);
            const auto paged = b.fx->runSplitPaged(0, b.q, c.visible, num_splits);

            EXPECT_TRUE(bitwiseEqual(contig, paged)) << "同一归约、两种取址必须逐位相同，max|diff|="
                                                     << maxAbsDiff(toFloat(contig), toFloat(paged));
        }
    }
}

// ── 4a. visible = 0：全 0 输出，且不与单遍路径分叉 ────────────────────────
TEST_F(SplitKvTest, EmptyVisibleWindowIsAllZero) {
    PagedGeometry  g = makeGeometry(4, 2, 64, 16, 4);
    const auto     q = randomFp16(static_cast<size_t>(g.num_q_heads) * g.head_dim, 77);
    const auto     table = makeTable(1, g.max_num_blocks, 78);
    SplitKvFixture fx(g, table, /*table_len=*/1);

    for (int num_splits : {1, 8}) {
        SCOPED_TRACE("num_splits=" + std::to_string(num_splits));
        const auto out = fx.runSplitPaged(0, q, /*visible=*/0, num_splits);
        for (half h : out) {
            EXPECT_EQ(__half2float(h), 0.0f) << "visible=0 必须输出全 0（冻结语义）";
        }
        // 也不得出现 NaN：上面用 == 比较已经隐含（NaN != 0）
    }
}

// ── 4b. visible < num_splits：大量空段，必须中性且无 NaN ──────────────────
TEST_F(SplitKvTest, VisibleSmallerThanSplitsHasNeutralEmptySplits) {
    const int     visible = 3;
    const int     bs = 16;
    const int     nb = requiredBlocks(visible, bs);
    PagedGeometry g = makeGeometry(4, 2, 64, bs, nb + 3);
    auto          b = buildCase(g, visible, 4242u);

    for (int num_splits : {8, 16, 32}) {
        SCOPED_TRACE("num_splits=" + std::to_string(num_splits));
        const auto out = b.fx->runSplitPaged(0, b.q, visible, num_splits);
        const auto got = toFloat(out);
        for (float x : got)
            EXPECT_TRUE(std::isfinite(x)) << "空段导致了 NaN/Inf";

        const auto ref = oracleAttentionDecode(toFloat(b.q), toFloat(b.k_logical),
                                               toFloat(b.v_logical), visible, g);
        EXPECT_LT(maxAbsDiff(got, ref), kOracleTolerance);
    }
}

// ── 4c. visible 不是 num_splits 的整数倍（尾段截断）────────────────────────
TEST_F(SplitKvTest, VisibleNotMultipleOfSplitsIsCorrect) {
    const int     visible = 17;
    const int     bs = 16;
    const int     nb = requiredBlocks(visible, bs);
    PagedGeometry g = makeGeometry(8, 2, 64, bs, nb + 3);
    auto          b = buildCase(g, visible, 31337u);

    for (int num_splits : {3, 5, 7}) {
        SCOPED_TRACE("num_splits=" + std::to_string(num_splits));
        const auto got = toFloat(b.fx->runSplitPaged(0, b.q, visible, num_splits));
        const auto ref = oracleAttentionDecode(toFloat(b.q), toFloat(b.k_logical),
                                               toFloat(b.v_logical), visible, g);
        EXPECT_LT(maxAbsDiff(got, ref), kOracleTolerance);
    }
}

// ── 5. 非法块 id：零行语义不变，且 split 与单遍逐位一致 ────────────────────
TEST_F(SplitKvTest, InvalidBlockIdsFollowZeroRowSemantics) {
    const int     block_size = 16;
    const int     visible = 20;
    const int     nb = requiredBlocks(visible, block_size);
    PagedGeometry g = makeGeometry(4, 4, 32, block_size, nb + 1);

    const auto q = randomFp16(static_cast<size_t>(g.num_q_heads) * g.head_dim, 902);
    const auto k = randomFp16(static_cast<size_t>(visible) * g.kvDim(), 900);
    const auto v = randomFp16(static_cast<size_t>(visible) * g.kvDim(), 901);

    for (int bad : {-1, g.max_num_blocks}) {
        SCOPED_TRACE("bad block id = " + std::to_string(bad));
        // table_len = nb，但第二块 id 越界
        std::vector<int> table = {1, bad};
        SplitKvFixture   fx(g, table, nb);
        fx.scatter(0, k, v, visible, 0);
        fx.sync();

        const auto single = fx.runSinglePassPaged(0, q, visible);
        for (int num_splits : {1, 4}) {
            const auto split = fx.runSplitPaged(0, q, visible, num_splits);
            EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "非法块 id 不得触发 illegal address";
            EXPECT_TRUE(bitwiseEqual(single, split))
                << "非法块 id 下 split 与单遍必须同为零行语义（num_splits=" << num_splits << "）";
            const auto got = toFloat(split);
            for (float x : got)
                EXPECT_TRUE(std::isfinite(x));
        }
    }
}

// ── 6. 块表长度不足：split 侧按零行处理且不越界读块表 ─────────────────────
TEST_F(SplitKvTest, ShortBlockTableIsGuardedWithoutFault) {
    const int     block_size = 16;
    const int     visible = 32; // 需要 2 块
    const int     nb = requiredBlocks(visible, block_size);
    PagedGeometry g = makeGeometry(4, 2, 64, block_size, 4);

    const auto table = makeTable(nb, g.max_num_blocks, 55);
    const auto q = randomFp16(static_cast<size_t>(g.num_q_heads) * g.head_dim, 56);
    const auto k = randomFp16(static_cast<size_t>(visible) * g.kvDim(), 57);
    const auto v = randomFp16(static_cast<size_t>(visible) * g.kvDim(), 58);

    // table_len = 1 < nb：第二个块必须按零行参与 softmax。
    SplitKvFixture fx(g, table, /*table_len=*/1);
    fx.scatter(0, k, v, visible, 0);
    fx.sync();

    const auto got = toFloat(fx.runSplitPaged(0, q, visible, /*num_splits=*/4));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "块表长度不足不得越界读块表";

    std::vector<float> k_eff = toFloat(k), v_eff = toFloat(v);
    for (int t = block_size; t < visible; ++t) {
        for (int c = 0; c < g.kvDim(); ++c) {
            k_eff[static_cast<size_t>(t) * g.kvDim() + static_cast<size_t>(c)] = 0.0f;
            v_eff[static_cast<size_t>(t) * g.kvDim() + static_cast<size_t>(c)] = 0.0f;
        }
    }
    const auto ref = oracleAttentionDecode(toFloat(q), k_eff, v_eff, visible, g);
    EXPECT_LT(maxAbsDiff(got, ref), kOracleTolerance)
        << "table_len 之外的 token 必须按零行参与，而不是读取 pool 内容";
}

// ── 7. CUDA Graph：捕获后改变可见长度并 replay，与非 graph 逐位相同 ───────
TEST_F(SplitKvTest, GraphReplayWithGrowingVisibleMatchesEager) {
    const int     bs = 16;
    const int     max_visible = 96;
    const int     nb = requiredBlocks(max_visible, bs);
    PagedGeometry g = makeGeometry(8, 2, 64, bs, nb + 3);
    auto          b = buildCase(g, max_visible, 8888u);

    const size_t       q_elems = b.fx->qElems();
    DeviceBuffer<half> d_q(q_elems), d_out_graph(q_elems);
    d_q.copyFromHost(b.q.data(), q_elems);

    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);

    // 捕获：注意 num_splits 是 host 常量、grid 固定；可见长度走 device int。
    const int num_splits = 4;
    ASSERT_EQ(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    attention_decode_paged_splitkv(
        d_q.data(), b.fx->kPoolData(), b.fx->vPoolData(), b.fx->tableData(), d_out_graph.data(),
        b.fx->scale(), g.num_q_heads, g.num_kv_heads, g.head_dim, b.fx->lenData(), g.block_size,
        g.max_num_blocks, b.fx->tableLen(), b.fx->partialData(), num_splits, s);
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(s, &graph), cudaSuccess);
    ASSERT_NE(graph, nullptr);
    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);

    // replay：可见长度逐个增长（同一 graph，grid 不变）
    for (int visible : {1, 7, 17, 33, 64, max_visible}) {
        SCOPED_TRACE("visible=" + std::to_string(visible));

        b.fx->setLen(visible, s);
        ASSERT_EQ(cudaGraphLaunch(exec, s), cudaSuccess);
        std::vector<half> graph_out(q_elems);
        ASSERT_EQ(cudaMemcpyAsync(graph_out.data(), d_out_graph.data(), q_elems * sizeof(half),
                                  cudaMemcpyDeviceToHost, s),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);

        const auto eager = b.fx->runSplitPaged(0, b.q, visible, num_splits);
        EXPECT_TRUE(bitwiseEqual(graph_out, eager))
            << "graph replay 与 eager 不一致，max|diff|="
            << maxAbsDiff(toFloat(graph_out), toFloat(eager));
    }

    ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(s), cudaSuccess);
}

// ── 9. 多 tile 的在线 rescale 必须被真正压到（变异检验 m4 暴露的缺口）───────
//
// 背景：随机数据 + 短序列时，全局 max 几乎总落在第一个 tile，old_rescale 恒为
// exp(0)=1，于是"丢掉 rescale"这类实现错误不会显形（实测：初版矩阵下删掉
// running_sum / out_acc 的 rescale 后 11 项门禁全过）。这里**确定性构造**后置 max：
// 除最后一个 tile 的一个 token 外 K 全 0、Q 全 1 ⇒ 其余 score 恒为 0，全局 max
// 必然落在后面的 tile，old_rescale 必然 ≠ 1。
TEST_F(SplitKvTest, LateMaxForcesOnlineRescaleToMatter) {
    const int     bs = 16, hd = 64, hq = 4, hkv = 4, visible = 300;
    const int     nb = requiredBlocks(visible, bs);
    PagedGeometry g = makeGeometry(hq, hkv, hd, bs, nb + 3);

    std::vector<half> k(static_cast<size_t>(visible) * g.kvDim(), __float2half(0.0f));
    std::vector<half> v(static_cast<size_t>(visible) * g.kvDim());
    std::vector<half> q(static_cast<size_t>(hq) * hd, __float2half(1.0f));

    // V：每个 token 一个互不相同的常数，使"前段均值"与"最后一个 token"明显不同
    for (int t = 0; t < visible; ++t)
        for (int c = 0; c < g.kvDim(); ++c)
            v[static_cast<size_t>(t) * g.kvDim() + c] =
                __float2half(static_cast<float>(t % 7) + 1.0f);

    // 后置 max：token 290 落在 [256, 300)，K 全 200 ⇒ score 远大于 0
    const int late = visible - 10;
    for (int c = 0; c < g.kvDim(); ++c)
        k[static_cast<size_t>(late) * g.kvDim() + c] = __float2half(200.0f);

    auto b = buildCase(g, visible, 6060u);
    b.fx->scatter(0, k, v, visible, 0);
    b.fx->sync();

    const auto ref = oracleAttentionDecode(toFloat(q), toFloat(k), toFloat(v), visible, g);

    // 单遍路径跨 3 个 tile（128/128/44），必然经过 rescale
    const auto single = toFloat(b.fx->runSinglePassPaged(0, q, visible));
    EXPECT_LT(maxAbsDiff(single, ref), kOracleTolerance)
        << "单遍路径未正确 rescale（旧 max 的权重被错误保留），max|diff|="
        << maxAbsDiff(single, ref);

    // split 路径：num_splits=1 时同样跨 tile；>1 时考验 combine 的跨段权重
    for (int num_splits : {1, 2, 4, 8}) {
        SCOPED_TRACE("num_splits=" + std::to_string(num_splits));
        const auto got = toFloat(b.fx->runSplitPaged(0, q, visible, num_splits));
        for (float x : got)
            EXPECT_TRUE(std::isfinite(x));
        EXPECT_LT(maxAbsDiff(got, ref), kOracleTolerance)
            << "split 路径未正确合并后置 max，max|diff|=" << maxAbsDiff(got, ref);
    }
}

// ── 10. non-default stream 与 default stream 结果一致 ──────────────────────
TEST_F(SplitKvTest, NonDefaultStreamMatchesDefaultStream) {
    const int     bs = 16;
    const int     visible = 40;
    const int     nb = requiredBlocks(visible, bs);
    PagedGeometry g = makeGeometry(8, 2, 64, bs, nb + 3);
    auto          b = buildCase(g, visible, 2024u);

    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);

    const auto on_default = b.fx->runSplitPaged(0, b.q, visible, /*num_splits=*/4, /*stream=*/0);
    const auto on_stream = b.fx->runSplitPaged(0, b.q, visible, /*num_splits=*/4, s);

    EXPECT_TRUE(bitwiseEqual(on_default, on_stream))
        << "非默认 stream 上结果必须一致，max|diff|="
        << maxAbsDiff(toFloat(on_default), toFloat(on_stream));

    ASSERT_EQ(cudaStreamDestroy(s), cudaSuccess);
}
