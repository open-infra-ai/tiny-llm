// TLLM-P0-004 PR-3：TransformerLayer 分页 decode 的 runtime dispatch 门禁。
//
// 验证三件事：
//   1. TLLM_PAGED_ATTENTION 的取值语义（auto / legacy / direct、大小写、非法值）；
//   2. dispatch 真的把 decode 路由到了预期路径——用**共享 scratch 是否被写**直接观测：
//      legacy 会 gather 进 scratch，direct 不碰 scratch；
//   3. 两条路径在同一 pool、同一输入下产生**逐位相同**的层输出（层级端到端等价）。
//
// 边界：不测 FFI / C ABI（另一范畴），也不产生性能数字。默认值按设计包 §11 为
// legacy，因此本 PR 不改变生产默认行为。

#include "rope.cuh"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/transformer.h"
#include "transpose_weights.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace tiny_llm;
using namespace tiny_llm::kernels;

namespace {

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

float maxAbsDiff(const std::vector<half> &a, const std::vector<half> &b) {
    const auto   fa = toFloat(a), fb = toFloat(b);
    const size_t n = std::min(fa.size(), fb.size());
    float        m = 0.0f;
    for (size_t i = 0; i < n; ++i)
        m = std::max(m, std::fabs(fa[i] - fb[i]));
    return m;
}

bool bitwiseEqual(const std::vector<half> &a, const std::vector<half> &b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(half)) == 0;
}

// 设置 TLLM_PAGED_ATTENTION；value == nullptr 表示删除该变量（测默认值）。
class ScopedEnv {
  public:
    ScopedEnv(const char *name, const char *value) : name_(name) {
        if (value == nullptr) {
            unsetenv(name);
        } else {
            setenv(name, value, 1);
        }
    }
    ~ScopedEnv() { unsetenv(name_.c_str()); }
    ScopedEnv(const ScopedEnv &) = delete;
    ScopedEnv &operator=(const ScopedEnv &) = delete;

  private:
    std::string name_;
};

// QuantizedWeight 几何约定：rows = K（输入维），cols = N（输出维），
// scales = [ceil(K/group_size), N]（见 kernels/w8a16_matmul.cuh）。
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

constexpr float kSentinel = 7.0f;

} // namespace

class PagedDispatchTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";
        cudaSetDevice(0);
    }
    void TearDown() override { cudaDeviceSynchronize(); }

    static constexpr int kBlockSize = 16;
    static constexpr int kMaxBlocks = 8;
    static constexpr int kPrompt = 20;

    struct Fixture {
        ModelConfig                       config;
        LayerWorkspace                    ws;
        TransformerWeights                weights;
        std::unique_ptr<TransformerLayer> layer;
        DeviceBuffer<float>               d_cos;
        DeviceBuffer<float>               d_sin;
        DeviceBuffer<half>                k_pool, v_pool, k_scratch, v_scratch;
        DeviceBuffer<half>                hidden_a, hidden_b;
        DeviceBuffer<int>                 table, decode_len;
        PagedKVCacheView                  view;
        int                               kv_dim = 0;
        int                               hidden = 0;
    };

    static void buildModel(Fixture &f) {
        f.config.vocab_size = 64;
        f.config.hidden_dim = 64;
        f.config.num_layers = 1;
        f.config.num_heads = 2;
        f.config.num_kv_heads = 1;
        f.config.head_dim = 32;
        f.config.intermediate_dim = 32;
        f.config.max_seq_len = 64;
        f.config.rope_theta = 10000.0f;
        f.config.rms_norm_eps = 1e-5f;

        const int     q_dim = f.config.num_heads * f.config.head_dim;
        const int     kvdim = f.config.num_kv_heads * f.config.head_dim;
        constexpr int kGroup = 32;
        f.weights.wq = makeSyntheticWeight(f.config.hidden_dim, q_dim, kGroup, 11);
        f.weights.wk = makeSyntheticWeight(f.config.hidden_dim, kvdim, kGroup, 12);
        f.weights.wv = makeSyntheticWeight(f.config.hidden_dim, kvdim, kGroup, 13);
        f.weights.wo = makeSyntheticWeight(q_dim, f.config.hidden_dim, kGroup, 14);
        f.weights.w1 =
            makeSyntheticWeight(f.config.hidden_dim, f.config.intermediate_dim, kGroup, 15);
        f.weights.w2 =
            makeSyntheticWeight(f.config.intermediate_dim, f.config.hidden_dim, kGroup, 16);
        f.weights.w3 =
            makeSyntheticWeight(f.config.hidden_dim, f.config.intermediate_dim, kGroup, 17);

        std::vector<half> ones(static_cast<size_t>(f.config.hidden_dim), __float2half(1.0f));
        CUDA_CHECK(cudaMalloc(&f.weights.rms_att_weight, ones.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&f.weights.rms_ffn_weight, ones.size() * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(f.weights.rms_att_weight, ones.data(), ones.size() * sizeof(half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(f.weights.rms_ffn_weight, ones.data(), ones.size() * sizeof(half),
                              cudaMemcpyHostToDevice));

        f.ws.allocate(f.config);
        f.layer = std::make_unique<TransformerLayer>(0, f.weights, f.config, &f.ws);

        const int half_d = f.config.head_dim / 2;
        f.d_cos = DeviceBuffer<float>(static_cast<size_t>(f.config.max_seq_len) * half_d);
        f.d_sin = DeviceBuffer<float>(static_cast<size_t>(f.config.max_seq_len) * half_d);
        rope_precompute_cache(f.d_cos.data(), f.d_sin.data(), f.config.max_seq_len,
                              f.config.head_dim, f.config.rope_theta, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    static void freeModel(Fixture &f) {
        f.layer.reset();
        freeQuantizedWeight(f.weights.wq);
        freeQuantizedWeight(f.weights.wk);
        freeQuantizedWeight(f.weights.wv);
        freeQuantizedWeight(f.weights.wo);
        freeQuantizedWeight(f.weights.w1);
        freeQuantizedWeight(f.weights.w2);
        freeQuantizedWeight(f.weights.w3);
        if (f.weights.rms_att_weight) cudaFree(f.weights.rms_att_weight);
        if (f.weights.rms_ffn_weight) cudaFree(f.weights.rms_ffn_weight);
        f.weights.rms_att_weight = nullptr;
        f.weights.rms_ffn_weight = nullptr;
    }

    // 建模型 + pool + scratch + 块表，并 prefill 一次（hidden_a / hidden_b 同步推进）。
    static void buildFixture(Fixture &f) {
        buildModel(f);
        f.hidden = f.config.hidden_dim;
        f.kv_dim = f.config.num_kv_heads * f.config.head_dim;

        const size_t pool_elems =
            static_cast<size_t>(kMaxBlocks) * static_cast<size_t>(kBlockSize) * f.kv_dim;
        f.k_pool = DeviceBuffer<half>(pool_elems);
        f.v_pool = DeviceBuffer<half>(pool_elems);
        f.k_scratch = DeviceBuffer<half>(pool_elems);
        f.v_scratch = DeviceBuffer<half>(pool_elems);
        f.hidden_a = DeviceBuffer<half>(static_cast<size_t>(f.config.max_seq_len) * f.hidden);
        f.hidden_b = DeviceBuffer<half>(static_cast<size_t>(f.config.max_seq_len) * f.hidden);
        f.table = DeviceBuffer<int>(static_cast<size_t>(kMaxBlocks));
        f.decode_len = DeviceBuffer<int>(1);

        const std::vector<int> table_host = {3, 5, 1, 7, 0, 2, 6, 4};
        f.table.copyFromHost(table_host.data(), table_host.size());

        const auto input = randomFp16(static_cast<size_t>(f.config.max_seq_len) * f.hidden, 4242);
        f.hidden_a.copyFromHost(input.data(), input.size());
        f.hidden_b.copyFromHost(input.data(), input.size());

        f.view.k_pool = f.k_pool.data();
        f.view.v_pool = f.v_pool.data();
        f.view.block_table = f.table.data();
        f.view.k_scratch = f.k_scratch.data();
        f.view.v_scratch = f.v_scratch.data();
        f.view.visible_blocks = static_cast<int>(table_host.size());
        f.view.block_size = kBlockSize;
        f.view.max_num_blocks = kMaxBlocks;
        f.view.max_visible_tokens = kMaxBlocks * kBlockSize;

        const int         zero = 0;
        DeviceBuffer<int> d_pos(1);
        d_pos.copyFromHost(&zero, 1);
        f.view.position = 0;
        f.view.decode_len = nullptr;

        // 幂等：两次 prefill 写入同一 pool 内容，使 hidden_b 与 hidden_a 处于同一状态
        for (DeviceBuffer<half> *h : {&f.hidden_a, &f.hidden_b}) {
            auto r = f.layer->forwardPaged(h->data(), f.view, kPrompt, d_pos.data(), f.d_cos.data(),
                                           f.d_sin.data(), 0);
            ASSERT_TRUE(r.isOk()) << r.error();
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    // 把 scratch 全部写成哨兵值，用于观测 legacy 是否真的 gather 过。
    static void poisonScratch(Fixture &f) {
        const size_t      n = f.k_scratch.size();
        std::vector<half> s(n, __float2half(kSentinel));
        f.k_scratch.copyFromHost(s.data(), n);
        f.v_scratch.copyFromHost(s.data(), n);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    static bool scratchAllSentinel(Fixture &f) {
        std::vector<half> k(f.k_scratch.size()), v(f.v_scratch.size());
        f.k_scratch.copyToHost(k.data(), k.size());
        f.v_scratch.copyToHost(v.data(), v.size());
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (half h : k)
            if (__half2float(h) != kSentinel) return false;
        for (half h : v)
            if (__half2float(h) != kSentinel) return false;
        return true;
    }

    // 跑一次 decode（位置 = kPrompt + step），返回该步的层输出。
    static std::vector<half> runDecode(Fixture &f, DeviceBuffer<half> &hidden, int step) {
        const int         pos = kPrompt + step;
        const int         vis = pos + 1;
        DeviceBuffer<int> d_pos(1);
        d_pos.copyFromHost(&pos, 1);
        f.decode_len.copyFromHost(&vis, 1);
        f.view.position = pos;
        f.view.decode_len = f.decode_len.data();

        auto r = f.layer->forwardPaged(hidden.data() + static_cast<size_t>(pos) * f.hidden, f.view,
                                       1, d_pos.data(), f.d_cos.data(), f.d_sin.data(), 0);
        EXPECT_TRUE(r.isOk()) << r.error();
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<half> out(static_cast<size_t>(f.hidden));
        CUDA_CHECK(cudaMemcpy(out.data(), hidden.data() + static_cast<size_t>(pos) * f.hidden,
                              static_cast<size_t>(f.hidden) * sizeof(half),
                              cudaMemcpyDeviceToHost));
        return out;
    }
};

// ── 默认（未设置）必须是 legacy：scratch 被 gather 写入 ─────────────────────
TEST_F(PagedDispatchTest, DefaultModeIsLegacySoGatherRuns) {
    ScopedEnv env("TLLM_PAGED_ATTENTION", nullptr);
    Fixture   f;
    buildFixture(f);
    poisonScratch(f);
    runDecode(f, f.hidden_a, 0);
    EXPECT_FALSE(scratchAllSentinel(f))
        << "默认应当是 legacy（设计包 §11），因此 gather 必须写过 scratch";
    freeModel(f);
}

// ── direct：scratch 必须保持原样（gather 被跳过）────────────────────────────
TEST_F(PagedDispatchTest, DirectModeSkipsGather) {
    ScopedEnv env("TLLM_PAGED_ATTENTION", "direct");
    Fixture   f;
    buildFixture(f);
    poisonScratch(f);
    runDecode(f, f.hidden_a, 0);
    EXPECT_TRUE(scratchAllSentinel(f))
        << "direct 路径不得写 scratch——被写说明 dispatch 仍走了 legacy 的 gather";
    freeModel(f);
}

// ── auto 当前等价于 direct；取值大小写不敏感 ──────────────────────────────
TEST_F(PagedDispatchTest, AutoAndDirectAreCaseInsensitiveAndSkipGather) {
    for (const char *value : {"auto", "AUTO", "Auto", "direct", "DIRECT", "Direct"}) {
        SCOPED_TRACE(std::string("TLLM_PAGED_ATTENTION=") + value);
        ScopedEnv env("TLLM_PAGED_ATTENTION", value);
        Fixture   f;
        buildFixture(f);
        poisonScratch(f);
        runDecode(f, f.hidden_a, 0);
        EXPECT_TRUE(scratchAllSentinel(f)) << value << " 应当走 direct（不写 scratch）";
        freeModel(f);
    }
}

TEST_F(PagedDispatchTest, LegacyIsCaseInsensitive) {
    ScopedEnv env("TLLM_PAGED_ATTENTION", "LeGaCy");
    Fixture   f;
    buildFixture(f);
    poisonScratch(f);
    runDecode(f, f.hidden_a, 0);
    EXPECT_FALSE(scratchAllSentinel(f));
    freeModel(f);
}

// ── 层级数值等价：同一 pool 上 direct 与 legacy 的 decode 输出逐位相同 ──────
TEST_F(PagedDispatchTest, DirectAndLegacyProduceIdenticalLayerOutput) {
    Fixture           f;
    std::vector<half> legacy_out, direct_out;
    buildFixture(f);

    {
        ScopedEnv env("TLLM_PAGED_ATTENTION", "legacy");
        poisonScratch(f);
        legacy_out = runDecode(f, f.hidden_a, 0);
        EXPECT_FALSE(scratchAllSentinel(f));
    }
    {
        ScopedEnv env("TLLM_PAGED_ATTENTION", "direct");
        poisonScratch(f);
        direct_out = runDecode(f, f.hidden_b, 0);
        EXPECT_TRUE(scratchAllSentinel(f));
    }

    EXPECT_TRUE(bitwiseEqual(direct_out, legacy_out))
        << "direct 与 legacy 的层输出不逐位相同；max|diff|=" << maxAbsDiff(direct_out, legacy_out);

    freeModel(f);
}

// ── 多个 decode 步持续一致（不只第一步）─────────────────────────────────
TEST_F(PagedDispatchTest, DirectAndLegacyAgreeAcrossSteps) {
    ScopedEnv env("TLLM_PAGED_ATTENTION", "legacy");
    Fixture   a;
    buildFixture(a);
    std::vector<half> legacy_step1 = runDecode(a, a.hidden_a, 0);
    std::vector<half> legacy_step2 = runDecode(a, a.hidden_a, 1);

    ScopedEnv env2("TLLM_PAGED_ATTENTION", "direct");
    Fixture   b;
    buildFixture(b);
    std::vector<half> direct_step1 = runDecode(b, b.hidden_a, 0);
    std::vector<half> direct_step2 = runDecode(b, b.hidden_a, 1);

    EXPECT_TRUE(bitwiseEqual(direct_step1, legacy_step1)) << maxAbsDiff(direct_step1, legacy_step1);
    EXPECT_TRUE(bitwiseEqual(direct_step2, legacy_step2)) << maxAbsDiff(direct_step2, legacy_step2);

    freeModel(a);
    freeModel(b);
}

// ── 非法取值必须显式失败，不静默回退 ──────────────────────────────────────
TEST_F(PagedDispatchTest, InvalidModeValueFailsLoudly) {
    ScopedEnv env("TLLM_PAGED_ATTENTION", "direkt"); // 拼错
    Fixture   f;
    buildModel(f);

    // 只需要进入 attentionPaged；先构造最小视图（指针非空即可通过前面的几何校验）
    f.hidden = f.config.hidden_dim;
    f.kv_dim = f.config.num_kv_heads * f.config.head_dim;
    const size_t pool_elems =
        static_cast<size_t>(kMaxBlocks) * static_cast<size_t>(kBlockSize) * f.kv_dim;
    f.k_pool = DeviceBuffer<half>(pool_elems);
    f.v_pool = DeviceBuffer<half>(pool_elems);
    f.k_scratch = DeviceBuffer<half>(pool_elems);
    f.v_scratch = DeviceBuffer<half>(pool_elems);
    f.hidden_a = DeviceBuffer<half>(static_cast<size_t>(f.config.max_seq_len) * f.hidden);
    f.table = DeviceBuffer<int>(static_cast<size_t>(kMaxBlocks));

    f.view.k_pool = f.k_pool.data();
    f.view.v_pool = f.v_pool.data();
    f.view.block_table = f.table.data();
    f.view.k_scratch = f.k_scratch.data();
    f.view.v_scratch = f.v_scratch.data();
    f.view.visible_blocks = 1;
    f.view.block_size = kBlockSize;
    f.view.max_num_blocks = kMaxBlocks;
    f.view.max_visible_tokens = kMaxBlocks * kBlockSize;
    f.view.position = 0;
    f.view.decode_len = nullptr;

    const int         zero = 0;
    DeviceBuffer<int> d_pos(1);
    d_pos.copyFromHost(&zero, 1);

    auto r = f.layer->forwardPaged(f.hidden_a.data(), f.view, 1, d_pos.data(), f.d_cos.data(),
                                   f.d_sin.data(), 0);
    ASSERT_TRUE(r.isErr()) << "非法 TLLM_PAGED_ATTENTION 取值必须返回错误，不得静默回退";
    EXPECT_NE(r.error().find("TLLM_PAGED_ATTENTION"), std::string::npos) << r.error();

    freeModel(f);
}

// ── prefill 不受开关影响：即使 direct，prefill 仍走 gather ─────────────────
TEST_F(PagedDispatchTest, PrefillIgnoresTheSwitch) {
    ScopedEnv env("TLLM_PAGED_ATTENTION", "direct");
    Fixture   f;
    buildFixture(f);
    poisonScratch(f);

    DeviceBuffer<half> h(static_cast<size_t>(f.config.max_seq_len) * f.hidden);
    const auto         input = randomFp16(h.size(), 999);
    h.copyFromHost(input.data(), input.size());

    const int         zero = 0;
    DeviceBuffer<int> d_pos(1);
    d_pos.copyFromHost(&zero, 1);
    f.view.position = 0;
    f.view.decode_len = nullptr;

    auto r =
        f.layer->forwardPaged(h.data(), f.view, 4, d_pos.data(), f.d_cos.data(), f.d_sin.data(), 0);
    ASSERT_TRUE(r.isOk()) << r.error();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_FALSE(scratchAllSentinel(f))
        << "prefill 必须保留 legacy 路径（设计包 §1 non-goals），gather 应当写过 scratch";
    freeModel(f);
}
