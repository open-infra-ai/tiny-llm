// Validator::validateModelConfig 及其在 C ABI 边界的接线。
//
// 背景：C ABI 路径（tinyllm_load）此前完全不校验模型几何，只有
// InferenceEngine::Load 调用过 validateModelConfig。而 attention kernel 用
// kv_head(q_head) = q_head / (num_heads / num_kv_heads) 做整数除法：
// num_heads 不被 num_kv_heads 整除时 kv_head 会超出 [0, num_kv_heads)，
// 最后一个 token 的 K/V 读越过缓冲末尾（Compute Sanitizer 可复现的 illegal
// address，会毒化 CUDA 上下文）。
//
// 本文件是纯 host 测试（无 CUDA 依赖）：校验发生在任何 CUDA 调用之前，
// 因此在无 GPU 的 CI 机器上同样运行。

#include "tiny_llm/ffi.h"
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/validator.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace tiny_llm;

namespace {

constexpr uint32_t U32_TYPE = 4;
constexpr uint32_t STR_TYPE = 8;
constexpr uint32_t ARR_TYPE = 9;

void pushU32(std::vector<uint8_t> &b, uint32_t v) {
    for (int i = 0; i < 4; ++i)
        b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void pushU64(std::vector<uint8_t> &b, uint64_t v) {
    for (int i = 0; i < 8; ++i)
        b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void pushStr(std::vector<uint8_t> &b, const std::string &s) {
    pushU64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
void pushU32Kv(std::vector<uint8_t> &b, const std::string &key, uint32_t value) {
    pushStr(b, key);
    pushU32(b, U32_TYPE);
    pushU32(b, value);
}
void pushStrKv(std::vector<uint8_t> &b, const std::string &key, const std::string &value) {
    pushStr(b, key);
    pushU32(b, STR_TYPE);
    pushStr(b, value);
}

// 构造一个最小但可解析的 GGUF：只含元数据，tensor_count = 0。
// head_count / head_count_kv 由调用方给定，用于制造几何不一致。
std::vector<uint8_t> minimalGguf(uint32_t num_heads, uint32_t num_kv_heads) {
    constexpr uint64_t   kKvCount = 8;
    std::vector<uint8_t> b;
    pushU32(b, GGUF_MAGIC);
    pushU32(b, 3);
    pushU64(b, 0); // tensor_count
    pushU64(b, kKvCount);

    pushStrKv(b, "general.architecture", "qwen2");
    pushU32Kv(b, "qwen2.embedding_length", 896); // 896 / 14 = 64（head_dim 偶数）
    pushU32Kv(b, "qwen2.block_count", 1);
    pushU32Kv(b, "qwen2.attention.head_count", num_heads);
    pushU32Kv(b, "qwen2.attention.head_count_kv", num_kv_heads);
    pushU32Kv(b, "qwen2.context_length", 128);
    pushU32Kv(b, "qwen2.feed_forward_length", 64);

    // tokenizer.ggml.tokens：vocab_size 由数组长度派生，必须非空
    pushStr(b, "tokenizer.ggml.tokens");
    pushU32(b, ARR_TYPE);
    pushU32(b, STR_TYPE);
    constexpr uint64_t kVocab = 64;
    pushU64(b, kVocab);
    for (uint64_t i = 0; i < kVocab; ++i)
        pushStr(b, "t" + std::to_string(i));
    return b;
}

std::string writeTempGguf(const std::vector<uint8_t> &buf, const char *tag) {
    const std::string path =
        std::string("/tmp/tiny_llm_cfgval_") + tag + "_" + std::to_string(::getpid()) + ".gguf";
    std::ofstream f(path, std::ios::binary);
    EXPECT_TRUE(f.is_open()) << "cannot write " << path;
    f.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
    f.close();
    return path;
}

// 一个真实可用的几何（Qwen2.5-0.5B 量级）：896 = 14 * 64，head_dim 为偶数。
ModelConfig validConfig() {
    ModelConfig c;
    c.vocab_size = 151936;
    c.hidden_dim = 896;
    c.num_layers = 24;
    c.num_heads = 14;
    c.num_kv_heads = 2;
    c.head_dim = 64;
    c.intermediate_dim = 4864;
    c.max_seq_len = 32768;
    c.rope_theta = 10000.0f;
    c.rms_norm_eps = 1e-5f;
    return c;
}

} // namespace

TEST(ValidateModelConfigTest, AcceptsRealisticGqaConfig) {
    auto r = Validator::validateModelConfig(validConfig());
    EXPECT_TRUE(r.isOk()) << r.error();
}

TEST(ValidateModelConfigTest, AcceptsMhaConfig) {
    ModelConfig c = validConfig();
    c.num_kv_heads = c.num_heads; // MHA：group_size == 1
    auto r = Validator::validateModelConfig(c);
    EXPECT_TRUE(r.isOk()) << r.error();
}

TEST(ValidateModelConfigTest, AcceptsMqaConfig) {
    ModelConfig c = validConfig();
    c.num_kv_heads = 1; // MQA
    auto r = Validator::validateModelConfig(c);
    EXPECT_TRUE(r.isOk()) << r.error();
}

// 回归锚点：这个组合曾让 attention_decode 越界读 K/V（sanitizer 13 errors）。
TEST(ValidateModelConfigTest, RejectsHeadCountsThatWouldOverrunKvHead) {
    ModelConfig c = validConfig();
    c.num_heads = 14;
    c.num_kv_heads = 3; // 14 % 3 != 0 → q_head/4 最大为 3，超出 [0,3)
    auto r = Validator::validateModelConfig(c);
    ASSERT_TRUE(r.isErr());
    EXPECT_NE(r.error().find("divisible"), std::string::npos) << r.error();
}

TEST(ValidateModelConfigTest, RejectsAllNonDivisibleHeadCounts) {
    for (int heads = 1; heads <= 32; ++heads) {
        for (int kv_heads = 1; kv_heads <= 32; ++kv_heads) {
            if (heads % kv_heads == 0) continue;
            ModelConfig c = validConfig();
            c.num_heads = heads;
            c.num_kv_heads = kv_heads;
            // 让 hidden_dim 仍是 num_heads 的整数倍，隔离出整除性这一条
            c.hidden_dim = heads * 64;
            c.head_dim = 64;
            EXPECT_TRUE(Validator::validateModelConfig(c).isErr())
                << "num_heads=" << heads << " num_kv_heads=" << kv_heads << " 应被拒绝";
        }
    }
}

TEST(ValidateModelConfigTest, RejectsHiddenDimNotDivisibleByHeads) {
    ModelConfig c = validConfig();
    c.num_heads = 13;
    c.num_kv_heads = 1;
    c.hidden_dim = 896; // 896 % 13 != 0
    auto r = Validator::validateModelConfig(c);
    ASSERT_TRUE(r.isErr());
    EXPECT_NE(r.error().find("divisible"), std::string::npos) << r.error();
}

// RoPE 的 half-split 依赖 head_dim 为偶数；奇数会静默截断频率对。
TEST(ValidateModelConfigTest, RejectsOddHeadDim) {
    ModelConfig c = validConfig();
    c.hidden_dim = 14 * 63;
    c.head_dim = 63;
    auto r = Validator::validateModelConfig(c);
    ASSERT_TRUE(r.isErr());
    EXPECT_NE(r.error().find("even"), std::string::npos) << r.error();
}

TEST(ValidateModelConfigTest, RejectsNonPositiveDimensions) {
    const char *fields[] = {"vocab_size",   "hidden_dim", "num_layers", "num_heads",
                            "num_kv_heads", "head_dim",   "max_seq_len"};
    for (const char *field : fields) {
        ModelConfig c = validConfig();
        if (std::string(field) == "vocab_size") c.vocab_size = 0;
        if (std::string(field) == "hidden_dim") c.hidden_dim = 0;
        if (std::string(field) == "num_layers") c.num_layers = -1;
        if (std::string(field) == "num_heads") c.num_heads = 0;
        if (std::string(field) == "num_kv_heads") c.num_kv_heads = 0;
        if (std::string(field) == "head_dim") c.head_dim = -64;
        if (std::string(field) == "max_seq_len") c.max_seq_len = 0;
        EXPECT_TRUE(Validator::validateModelConfig(c).isErr()) << field << " 应被拒绝";
    }
}

TEST(ValidateModelConfigTest, RejectsNonPositiveRmsNormEps) {
    ModelConfig c = validConfig();
    c.rms_norm_eps = 0.0f;
    EXPECT_TRUE(Validator::validateModelConfig(c).isErr());
}

// ── C ABI 边界：tinyllm_load 必须拒绝几何不一致的 GGUF ─────────────────────
//
// 这是修复的端到端回归锚点：没有 tinyllm_load 里的 validateModelConfig 调用时，
// 14/3 这种几何会一路走到 attention kernel，造成越界读。

TEST(ModelConfigBoundaryTest, TinyLlmLoadRejectsNonDivisibleHeadCounts) {
    const auto        buf = minimalGguf(/*num_heads=*/14, /*num_kv_heads=*/3);
    const std::string path = writeTempGguf(buf, "badheads");

    char           err[512] = {0};
    TinyLlmHandle *h = tinyllm_load(path.c_str(), nullptr, err, sizeof(err));

    EXPECT_EQ(h, nullptr) << "几何不一致的模型必须被拒绝，而不是进入 attention kernel";
    EXPECT_NE(std::string(err).find("invalid model config"), std::string::npos)
        << "err_buf = " << err;

    std::remove(path.c_str());
}

// 反向保护：合法几何不得被误拒（误拒会让任何模型都加载不了）。
// 合成文件没有张量，因此权重加载必然失败——这里只断言失败原因**不是** config 校验。
TEST(ModelConfigBoundaryTest, TinyLlmLoadDoesNotRejectValidGeometry) {
    const auto        buf = minimalGguf(/*num_heads=*/14, /*num_kv_heads=*/2);
    const std::string path = writeTempGguf(buf, "goodheads");

    char           err[512] = {0};
    TinyLlmHandle *h = tinyllm_load(path.c_str(), nullptr, err, sizeof(err));

    EXPECT_EQ(h, nullptr); // 无张量，必然在后续阶段失败
    EXPECT_EQ(std::string(err).find("invalid model config"), std::string::npos)
        << "合法几何不应被 config 校验拒绝: " << err;

    std::remove(path.c_str());
}
