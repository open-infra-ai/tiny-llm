// tiny_llm_kernel_bench: decode-path kernel microbenchmark.
//
// Purpose: replace ncu / nsys stats (unavailable on this box — WSL2 without
// perf-counter permission and missing nsys importer) with a reproducible,
// in-repo measurement of "how many ms does each kernel take".
//
// Measures the exact shapes used by Qwen2.5-0.5B decode:
//   W8A16 GEMM      M=1, K=896, N in {128, 896, 4864}
//   W8A16 GEMM down M=1, K=4864, N=896
//   FP16 lm_head    M=1, K=896, N=151936
//   attention_decode S in {8,32,64,128}, Hq=14, Hkv=2, D=64
//   rmsnorm         batch=1, hidden=896
//   RoPE            num_tokens=1, Hq=14, Hkv=2, D=64, pos=0
//   add             n=896
//   silu_mul        n=4864
//
// Timing: warmup 20 runs, then 200 runs (lm_head 100), cudaDeviceSynchronize
// before and after the loop, std::chrono::steady_clock mean ms.
// Output: CSV lines <name>,<shape>,<ms> for copy-paste into reports.
//
// This file deliberately contains NO optimization experiments — it only calls
// the existing public kernel interfaces.

#include "attention.cuh"
#include "elementwise.cuh"
#include "rmsnorm.cuh"
#include "rope.cuh"
#include "transpose_weights.cuh"
#include "w8a16_matmul.cuh"

#include "paged_kv.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const char *msg) {
    std::fprintf(stderr, "kernel_bench: %s (%s)\n", msg, cudaGetErrorString(cudaGetLastError()));
    std::exit(1);
}

void check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kernel_bench: %s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

// Warm up `warmup` times, then time `iters` runs with a sync before/after.
// Each call to `f` launches work on the default stream.
template <typename F>
double bench(F &&f, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i)
        f();
    check(cudaDeviceSynchronize(), "sync before timed loop");
    auto start = Clock::now();
    for (int i = 0; i < iters; ++i)
        f();
    check(cudaDeviceSynchronize(), "sync after timed loop");
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() /
           static_cast<double>(iters);
}

// ---------------------------------------------------------------------------
// W8A16 GEMM  (weight [K, N], scales [K/group, N])
// ---------------------------------------------------------------------------
double benchW8A16(const char *name, const char *shape, int M, int N, int K, int group_size,
                  int warmup, int iters) {
    (void)name; // 输出行仅使用 shape 列
    const int scale_rows = (K + group_size - 1) / group_size;

    std::vector<__half> h_input(static_cast<size_t>(M) * K, __float2half(0.5f));
    std::vector<int8_t> h_weight(static_cast<size_t>(K) * N, 1);
    std::vector<__half> h_scales(static_cast<size_t>(scale_rows) * N, __float2half(0.5f));
    std::vector<__half> h_output(static_cast<size_t>(M) * N);

    __half *d_input = nullptr, *d_scales = nullptr, *d_output = nullptr;
    int8_t *d_weight = nullptr, *d_weight_t = nullptr;
    __half *d_scales_t = nullptr;
    check(cudaMalloc(&d_input, h_input.size() * sizeof(__half)), "cudaMalloc input");
    check(cudaMalloc(&d_weight, h_weight.size()), "cudaMalloc weight");
    check(cudaMalloc(&d_scales, h_scales.size() * sizeof(__half)), "cudaMalloc scales");
    check(cudaMalloc(&d_output, h_output.size() * sizeof(__half)), "cudaMalloc output");
    check(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__half),
                     cudaMemcpyHostToDevice),
          "copy input");
    check(cudaMemcpy(d_weight, h_weight.data(), h_weight.size(), cudaMemcpyHostToDevice),
          "copy weight");
    check(cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(__half),
                     cudaMemcpyHostToDevice),
          "copy scales");

    // 任务 C1：构建转置布局 [N, K] / [N, scale_rows]（M==1 decode 生产路径），
    // 测量的是与推理引擎 decode 相同的 coalesced 快路径。
    check(cudaMalloc(&d_weight_t, h_weight.size()), "cudaMalloc weight_t");
    check(cudaMalloc(&d_scales_t, h_scales.size() * sizeof(__half)), "cudaMalloc scales_t");
    tiny_llm::kernels::transpose_int8(d_weight, d_weight_t, K, N, 0);
    tiny_llm::kernels::transpose_scales(d_scales, d_scales_t, scale_rows, N, 0);

    double ms = bench(
        [&] {
            tiny_llm::kernels::w8a16_matmul(d_input, d_weight, d_scales, d_weight_t, d_scales_t,
                                            d_output, M, N, K, group_size, 0);
        },
        warmup, iters);

    std::printf("w8a16_matmul,%s,%.4f\n", shape, ms);
    check(cudaFree(d_input), "cudaFree input");
    check(cudaFree(d_weight), "cudaFree weight");
    check(cudaFree(d_scales), "cudaFree scales");
    check(cudaFree(d_weight_t), "cudaFree weight_t");
    check(cudaFree(d_scales_t), "cudaFree scales_t");
    check(cudaFree(d_output), "cudaFree output");
    return ms;
}

// ---------------------------------------------------------------------------
// FP16 lm_head  (weight [K, N])
// ---------------------------------------------------------------------------
double benchFP16(const char *name, const char *shape, int M, int N, int K, int warmup, int iters) {
    (void)name; // 输出行仅使用 shape 列
    std::vector<__half> h_input(static_cast<size_t>(M) * K, __float2half(0.5f));
    std::vector<__half> h_weight(static_cast<size_t>(K) * N, __float2half(0.5f));
    std::vector<__half> h_output(static_cast<size_t>(M) * N);

    __half *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr, *d_weight_t = nullptr;
    check(cudaMalloc(&d_input, h_input.size() * sizeof(__half)), "cudaMalloc input");
    check(cudaMalloc(&d_weight, h_weight.size() * sizeof(__half)), "cudaMalloc weight");
    check(cudaMalloc(&d_output, h_output.size() * sizeof(__half)), "cudaMalloc output");
    check(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__half),
                     cudaMemcpyHostToDevice),
          "copy input");
    check(cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(__half),
                     cudaMemcpyHostToDevice),
          "copy weight");

    // 任务 C1：构建转置布局 [N, K]（M==1 decode 生产路径）
    check(cudaMalloc(&d_weight_t, h_weight.size() * sizeof(__half)), "cudaMalloc weight_t");
    tiny_llm::kernels::transpose_fp16(d_weight, d_weight_t, K, N, 0);

    double ms = bench(
        [&] {
            tiny_llm::kernels::fp16_matmul(d_input, d_weight, d_weight_t, d_output, M, N, K, 0);
        },
        warmup, iters);

    std::printf("fp16_matmul,%s,%.4f\n", shape, ms);
    check(cudaFree(d_input), "cudaFree input");
    check(cudaFree(d_weight), "cudaFree weight");
    check(cudaFree(d_weight_t), "cudaFree weight_t");
    check(cudaFree(d_output), "cudaFree output");
    return ms;
}

// ---------------------------------------------------------------------------
// attention_decode  (Q [1, Hq, D], K/V cache [S, Hkv, D], O [1, Hq, D])
// ---------------------------------------------------------------------------
double benchAttentionDecode(int S, int Hq, int Hkv, int D, int warmup, int iters) {
    const size_t q_elems = static_cast<size_t>(Hq) * D;
    const size_t kv_elems = static_cast<size_t>(S) * Hkv * D;

    __half *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    int    *d_len = nullptr;
    check(cudaMalloc(&d_q, q_elems * sizeof(__half)), "cudaMalloc q");
    check(cudaMalloc(&d_k, kv_elems * sizeof(__half)), "cudaMalloc k");
    check(cudaMalloc(&d_v, kv_elems * sizeof(__half)), "cudaMalloc v");
    check(cudaMalloc(&d_out, q_elems * sizeof(__half)), "cudaMalloc out");
    check(cudaMalloc(&d_len, sizeof(int)), "cudaMalloc len");
    check(cudaMemset(d_q, 0, q_elems * sizeof(__half)), "memset q");
    check(cudaMemset(d_k, 0, kv_elems * sizeof(__half)), "memset k");
    check(cudaMemset(d_v, 0, kv_elems * sizeof(__half)), "memset v");
    check(cudaMemset(d_out, 0, q_elems * sizeof(__half)), "memset out");
    check(cudaMemcpy(d_len, &S, sizeof(int), cudaMemcpyHostToDevice), "copy len");

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    double      ms = bench(
        [&] {
            tiny_llm::kernels::attention_decode(d_q, d_k, d_v, d_out, scale, Hq, Hkv, d_len, D, 0);
        },
        warmup, iters);

    char shape[64];
    std::snprintf(shape, sizeof(shape), "S=%d,Hq=%d,Hkv=%d,D=%d", S, Hq, Hkv, D);
    std::printf("attention_decode,%s,%.4f\n", shape, ms);

    check(cudaFree(d_q), "cudaFree q");
    check(cudaFree(d_k), "cudaFree k");
    check(cudaFree(d_v), "cudaFree v");
    check(cudaFree(d_out), "cudaFree out");
    check(cudaFree(d_len), "cudaFree len");
    return ms;
}

// ---------------------------------------------------------------------------
// rmsnorm  (batch=1, hidden)
// ---------------------------------------------------------------------------
double benchRMSNorm(int batch, int hidden, int warmup, int iters) {
    __half *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, static_cast<size_t>(batch) * hidden * sizeof(__half)), "cudaMalloc x");
    check(cudaMalloc(&d_w, static_cast<size_t>(hidden) * sizeof(__half)), "cudaMalloc w");
    check(cudaMalloc(&d_y, static_cast<size_t>(batch) * hidden * sizeof(__half)), "cudaMalloc y");
    check(cudaMemset(d_x, 0, static_cast<size_t>(batch) * hidden * sizeof(__half)), "memset x");
    check(cudaMemset(d_w, 0, static_cast<size_t>(hidden) * sizeof(__half)), "memset w");

    double ms = bench([&] { tiny_llm::kernels::rmsnorm(d_x, d_w, d_y, batch, hidden, 1e-6f, 0); },
                      warmup, iters);

    char shape[64];
    std::snprintf(shape, sizeof(shape), "batch=%d,hidden=%d", batch, hidden);
    std::printf("rmsnorm,%s,%.4f\n", shape, ms);

    check(cudaFree(d_x), "cudaFree x");
    check(cudaFree(d_w), "cudaFree w");
    check(cudaFree(d_y), "cudaFree y");
    return ms;
}

// ---------------------------------------------------------------------------
// RoPE (apply_rope_inplace, num_tokens=1)
// ---------------------------------------------------------------------------
double benchRoPE(int Hq, int Hkv, int D, int warmup, int iters) {
    const size_t q_elems = static_cast<size_t>(Hq) * D;
    const size_t k_elems = static_cast<size_t>(Hkv) * D;
    const size_t half_d = static_cast<size_t>(D) / 2;

    __half *d_q = nullptr, *d_k = nullptr;
    float  *d_cos = nullptr, *d_sin = nullptr;
    int    *d_pos = nullptr;
    check(cudaMalloc(&d_q, q_elems * sizeof(__half)), "cudaMalloc q");
    check(cudaMalloc(&d_k, k_elems * sizeof(__half)), "cudaMalloc k");
    check(cudaMalloc(&d_cos, half_d * sizeof(float)), "cudaMalloc cos");
    check(cudaMalloc(&d_sin, half_d * sizeof(float)), "cudaMalloc sin");
    check(cudaMalloc(&d_pos, sizeof(int)), "cudaMalloc pos");
    check(cudaMemset(d_q, 0, q_elems * sizeof(__half)), "memset q");
    check(cudaMemset(d_k, 0, k_elems * sizeof(__half)), "memset k");
    check(cudaMemset(d_cos, 0, half_d * sizeof(float)), "memset cos");
    check(cudaMemset(d_sin, 0, half_d * sizeof(float)), "memset sin");
    const int pos = 0;
    check(cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice), "copy pos");

    double ms = bench(
        [&] {
            tiny_llm::kernels::apply_rope_inplace(d_q, d_k, d_cos, d_sin, 1, d_pos, Hq, Hkv, D, 0);
        },
        warmup, iters);

    char shape[64];
    std::snprintf(shape, sizeof(shape), "tokens=1,Hq=%d,Hkv=%d,D=%d,pos=0", Hq, Hkv, D);
    std::printf("apply_rope_inplace,%s,%.4f\n", shape, ms);

    check(cudaFree(d_q), "cudaFree q");
    check(cudaFree(d_k), "cudaFree k");
    check(cudaFree(d_cos), "cudaFree cos");
    check(cudaFree(d_sin), "cudaFree sin");
    check(cudaFree(d_pos), "cudaFree pos");
    return ms;
}

// ---------------------------------------------------------------------------
// add_inplace / silu_mul_inplace
// ---------------------------------------------------------------------------
double benchElementwise(const char *name, int n, int warmup, int iters) {
    __half *d_a = nullptr, *d_b = nullptr;
    check(cudaMalloc(&d_a, static_cast<size_t>(n) * sizeof(__half)), "cudaMalloc a");
    check(cudaMalloc(&d_b, static_cast<size_t>(n) * sizeof(__half)), "cudaMalloc b");
    check(cudaMemset(d_a, 0, static_cast<size_t>(n) * sizeof(__half)), "memset a");
    check(cudaMemset(d_b, 0, static_cast<size_t>(n) * sizeof(__half)), "memset b");

    double ms = 0.0;
    if (std::string(name) == "add_inplace") {
        ms = bench([&] { tiny_llm::kernels::add_inplace(d_a, d_b, n, 0); }, warmup, iters);
    } else if (std::string(name) == "silu_mul_inplace") {
        ms = bench([&] { tiny_llm::kernels::silu_mul_inplace(d_a, d_b, n, 0); }, warmup, iters);
    } else {
        fail("unknown elementwise bench name");
    }

    std::printf("%s,n=%d,%.4f\n", name, n, ms);

    check(cudaFree(d_a), "cudaFree a");
    check(cudaFree(d_b), "cudaFree b");
    return ms;
}

// ===========================================================================
// TLLM-P0-004 PR-5：三路 decode attention kernel benchmark（设计包 §9）
//
// 只做 kernel 级测量，不产生 TTFT/TPOT。三条路径消费同一份逻辑 K/V：
//   legacy     = paged_gather_blocks(K) + paged_gather_blocks(V) + attention_decode
//   contiguous = attention_decode（连续 scratch，仅作上界参考）
//   direct     = attention_decode_paged
// gather_k / gather_v 单独计时，用于区分「省下的 gather」与「direct 自身开销」。
//
// §10.1 记录的两个测量陷阱在本文件内规避：
//   1. GPU 空闲时 SM 时钟停在低频、小 kernel 推不动 boost → 计时前先做持续负载预热；
//   2. host-int 重载每次调用附带一次 4 字节 H2D memcpy → 一律走 device-int 重载。
// 第三个陷阱（误编到 sm_75）由构建参数保证：必须以 -DCMAKE_CUDA_ARCHITECTURES=native
// 配置，程序启动时把设备 compute capability 写进 provenance 供核对。
//
// 采样：每个 sample 计时 batch 次调用再除以 batch（摊薄 launch 开销）；路径按 round
// 轮转顺序交替（order-balanced，抑制时钟漂移）；报告 median / p10 / p90 / CV。
// CV > 10% 或 3 次重复的 median 差异 > 10% 的 shape 标 not_converged，但不丢弃。
// ===========================================================================

struct DpaArgs {
    bool                                             enabled = false;
    int                                              warmup = 20;
    int                                              reps = 200;
    int                                              batch = 10;
    int                                              repeats = 3;
    double                                           clock_warmup_s = 4.0;
    unsigned                                         seed = 20260914u;
    const char                                      *out_path = nullptr;
    int                                              only_visible = 0;
    int                                              only_block_size = 0;
    std::vector<std::pair<std::string, std::string>> meta;
};

struct ShapeGeom {
    int visible = 0;
    int block_size = 0;
    int table_len = 0;
    int max_num_blocks = 0;
};

// 固定生产几何：Qwen2.5-0.5B decode（Hq=14, Hkv=2, D=64）。
constexpr int kDpaHq = 14;
constexpr int kDpaHkv = 2;
constexpr int kDpaHeadDim = 64;

enum DpaPath { kLegacy = 0, kContiguous, kDirect, kGatherK, kGatherV, kNumPaths };
const char *const kDpaPathNames[kNumPaths] = {"legacy", "contiguous", "direct", "gather_k",
                                              "gather_v"};

std::vector<ShapeGeom> dpaShapes() {
    std::vector<ShapeGeom> shapes;
    auto                   add = [&](int visible, int block_size) {
        for (const auto &s : shapes)
            if (s.visible == visible && s.block_size == block_size) return;
        ShapeGeom g;
        g.visible = visible;
        g.block_size = block_size;
        g.table_len = (visible + block_size - 1) / block_size;
        g.max_num_blocks = g.table_len + 3;
        shapes.push_back(g);
    };
    for (int bs : {16, 32}) {
        for (int v : {8, 32, 64, 128, 512, 1024, 2048})
            add(v, bs);
        // 块边界形状：k·block_size ± 1（§9）。
        for (int k : {1, 2, 4, 8})
            for (int d : {-1, 0, 1})
                if (k * bs + d > 0) add(k * bs + d, bs);
    }
    std::sort(shapes.begin(), shapes.end(), [](const ShapeGeom &a, const ShapeGeom &b) {
        return a.block_size != b.block_size ? a.block_size < b.block_size : a.visible < b.visible;
    });
    return shapes;
}

std::vector<half> dpaRandom(size_t n, unsigned seed) {
    std::mt19937                          gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<half>                     v(n);
    for (auto &x : v)
        x = __float2half(dist(gen));
    return v;
}

// 非连续物理块表：把 [0, max_num_blocks) 洗牌后取前 table_len 个。
std::vector<int> dpaTable(int table_len, int max_num_blocks, unsigned seed) {
    std::vector<int> ids(static_cast<size_t>(max_num_blocks));
    for (int i = 0; i < max_num_blocks; ++i)
        ids[static_cast<size_t>(i)] = i;
    std::mt19937 gen(seed);
    std::shuffle(ids.begin(), ids.end(), gen);
    return std::vector<int>(ids.begin(), ids.begin() + table_len);
}

double dpaPercentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p * static_cast<double>(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    const double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

double dpaMean(const std::vector<double> &v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v)
        s += x;
    return s / static_cast<double>(v.size());
}

double dpaStdev(const std::vector<double> &v) {
    if (v.size() < 2) return 0.0;
    const double m = dpaMean(v);
    double       s = 0.0;
    for (double x : v)
        s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

double dpaCvPercent(const std::vector<double> &v) {
    const double m = dpaMean(v);
    return m > 0.0 ? dpaStdev(v) / m * 100.0 : 0.0;
}

std::string dpaJsonEscape(const std::string &s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
            out += buf;
        } else {
            out.push_back(c);
        }
    }
    return out;
}

// 持续负载时钟预热：GPU 空闲时 SM 停在低频，小 attention kernel 拉不动 boost，
// 会让首轮样本系统性偏低（§10.1 陷阱 1）。用既有 GEMM 内核空转约 seconds 秒。
void dpaClockWarmup(double seconds) {
    if (seconds <= 0.0) return;
    const int         M = 1, N = 4864, K = 896;
    std::vector<half> h_input(static_cast<size_t>(M) * K, __float2half(0.5f));
    std::vector<half> h_weight(static_cast<size_t>(K) * N, __float2half(0.5f));
    std::vector<half> h_output(static_cast<size_t>(M) * N);

    __half *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr, *d_weight_t = nullptr;
    check(cudaMalloc(&d_input, h_input.size() * sizeof(half)), "cudaMalloc warmup input");
    check(cudaMalloc(&d_weight, h_weight.size() * sizeof(half)), "cudaMalloc warmup weight");
    check(cudaMalloc(&d_output, h_output.size() * sizeof(half)), "cudaMalloc warmup output");
    check(cudaMalloc(&d_weight_t, h_weight.size() * sizeof(half)), "cudaMalloc warmup weight_t");
    check(
        cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half), cudaMemcpyHostToDevice),
        "copy warmup input");
    check(cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(half),
                     cudaMemcpyHostToDevice),
          "copy warmup weight");
    tiny_llm::kernels::transpose_fp16(d_weight, d_weight_t, K, N, 0);
    check(cudaDeviceSynchronize(), "warmup transpose sync");

    std::fprintf(stderr, "dpa: clock warmup %.1fs ...\n", seconds);
    const auto deadline = Clock::now() + std::chrono::duration<double>(seconds);
    while (Clock::now() < deadline) {
        for (int i = 0; i < 20; ++i) {
            tiny_llm::kernels::fp16_matmul(d_input, d_weight, d_weight_t, d_output, M, N, K, 0);
        }
        check(cudaDeviceSynchronize(), "warmup sync");
    }

    check(cudaFree(d_input), "cudaFree warmup input");
    check(cudaFree(d_weight), "cudaFree warmup weight");
    check(cudaFree(d_weight_t), "cudaFree warmup weight_t");
    check(cudaFree(d_output), "cudaFree warmup output");
}

void dpaEmitProvenance(FILE *out, const DpaArgs &args, const cudaDeviceProp &prop,
                       int driver_version, int runtime_version,
                       const std::vector<ShapeGeom> &shapes) {
    char cc[16];
    std::snprintf(cc, sizeof(cc), "%d.%d", prop.major, prop.minor);

    std::fprintf(out, "{\"type\":\"provenance\",\"schema\":\"tllm-dpa-kernel-bench-v1\"");
    std::fprintf(
        out, ",\"gpu\":{\"name\":\"%s\",\"total_memory_mib\":%lu,\"compute_capability\":\"%s\"}",
        dpaJsonEscape(prop.name).c_str(),
        static_cast<unsigned long>(prop.totalGlobalMem / (1024 * 1024)), cc);
    std::fprintf(out, ",\"cuda_driver_version\":%d,\"cuda_runtime_version\":%d", driver_version,
                 runtime_version);
#ifdef NDEBUG
    std::fprintf(out, ",\"build_type\":\"RelWithDebInfo/Release (NDEBUG)\"");
#else
    std::fprintf(out, ",\"build_type\":\"Debug\"");
#endif
    std::fprintf(out, ",\"seed\":%u,\"warmup\":%d,\"reps\":%d,\"batch\":%d,\"repeats\":%d",
                 args.seed, args.warmup, args.reps, args.batch, args.repeats);
    std::fprintf(out, ",\"geometry\":{\"num_q_heads\":%d,\"num_kv_heads\":%d,\"head_dim\":%d}",
                 kDpaHq, kDpaHkv, kDpaHeadDim);
    std::fprintf(out, ",\"shapes\":[");
    for (size_t i = 0; i < shapes.size(); ++i) {
        std::fprintf(out, "%s{\"visible_tokens\":%d,\"block_size\":%d,\"table_len\":%d}",
                     i == 0 ? "" : ",", shapes[i].visible, shapes[i].block_size,
                     shapes[i].table_len);
    }
    std::fprintf(out, "]");
    std::fprintf(out, ",\"limitations\":\"kernel-level only; no TTFT/TPOT claim; scratch kept on "
                      "the direct path (design package 5)\"");
    if (!args.meta.empty()) {
        std::fprintf(out, ",\"meta\":{");
        for (size_t i = 0; i < args.meta.size(); ++i) {
            std::fprintf(out, "%s\"%s\":\"%s\"", i == 0 ? "" : ",",
                         dpaJsonEscape(args.meta[i].first).c_str(),
                         dpaJsonEscape(args.meta[i].second).c_str());
        }
        std::fprintf(out, "}");
    }
    std::fprintf(out, "}\n");
    std::fflush(out);
}

void dpaRunShape(FILE *out, const DpaArgs &args, const ShapeGeom &g, int rounds) {
    const int    kv_dim = kDpaHkv * kDpaHeadDim;
    const size_t pool_elems =
        static_cast<size_t>(g.max_num_blocks) * static_cast<size_t>(g.block_size) * kv_dim;
    const size_t q_elems = static_cast<size_t>(kDpaHq) * kDpaHeadDim;

    half *d_k_pool = nullptr, *d_v_pool = nullptr, *d_k_scratch = nullptr, *d_v_scratch = nullptr;
    half *d_q = nullptr, *d_out_legacy = nullptr, *d_out_contig = nullptr, *d_out_direct = nullptr;
    int  *d_table = nullptr, *d_len = nullptr;
    check(cudaMalloc(&d_k_pool, pool_elems * sizeof(half)), "cudaMalloc k_pool");
    check(cudaMalloc(&d_v_pool, pool_elems * sizeof(half)), "cudaMalloc v_pool");
    check(cudaMalloc(&d_k_scratch, pool_elems * sizeof(half)), "cudaMalloc k_scratch");
    check(cudaMalloc(&d_v_scratch, pool_elems * sizeof(half)), "cudaMalloc v_scratch");
    check(cudaMalloc(&d_q, q_elems * sizeof(half)), "cudaMalloc q");
    check(cudaMalloc(&d_out_legacy, q_elems * sizeof(half)), "cudaMalloc out_legacy");
    check(cudaMalloc(&d_out_contig, q_elems * sizeof(half)), "cudaMalloc out_contig");
    check(cudaMalloc(&d_out_direct, q_elems * sizeof(half)), "cudaMalloc out_direct");
    check(cudaMalloc(&d_table, static_cast<size_t>(g.max_num_blocks) * sizeof(int)),
          "cudaMalloc table");
    check(cudaMalloc(&d_len, sizeof(int)), "cudaMalloc len");

    const unsigned seed = args.seed + static_cast<unsigned>(g.visible * 131 + g.block_size);
    const std::vector<half> h_q = dpaRandom(q_elems, seed + 1);
    const std::vector<half> h_table_src =
        dpaRandom(static_cast<size_t>(g.visible) * kv_dim, seed + 2); // 逻辑 K/V（仅用于填充 pool）
    const std::vector<int> h_table = dpaTable(g.table_len, g.max_num_blocks, seed + 3);
    const int              visible = g.visible;

    check(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(half), cudaMemcpyHostToDevice), "copy q");
    check(cudaMemcpy(d_table, h_table.data(), h_table.size() * sizeof(int), cudaMemcpyHostToDevice),
          "copy table");
    check(cudaMemcpy(d_len, &visible, sizeof(int), cudaMemcpyHostToDevice), "copy len");
    check(cudaMemset(d_k_pool, 0, pool_elems * sizeof(half)), "memset k_pool");
    check(cudaMemset(d_v_pool, 0, pool_elems * sizeof(half)), "memset v_pool");

    // 把逻辑 K/V 按块表散布进 pool，使三条路径读到同一份真实数据。
    {
        const size_t      elems = static_cast<size_t>(g.visible) * kv_dim;
        std::vector<half> h_k = h_table_src;
        std::vector<half> h_v = h_table_src;
        for (auto &x : h_v)
            x = __float2half(__half2float(x) * 0.5f);
        half *d_tmp_k = nullptr, *d_tmp_v = nullptr;
        check(cudaMalloc(&d_tmp_k, elems * sizeof(half)), "cudaMalloc tmp_k");
        check(cudaMalloc(&d_tmp_v, elems * sizeof(half)), "cudaMalloc tmp_v");
        check(cudaMemcpy(d_tmp_k, h_k.data(), elems * sizeof(half), cudaMemcpyHostToDevice),
              "copy tmp_k");
        check(cudaMemcpy(d_tmp_v, h_v.data(), elems * sizeof(half), cudaMemcpyHostToDevice),
              "copy tmp_v");
        tiny_llm::kernels::paged_scatter_blocks(d_tmp_k, d_k_pool, d_table, g.visible, 0,
                                                g.block_size, kv_dim, g.max_num_blocks, 0);
        tiny_llm::kernels::paged_scatter_blocks(d_tmp_v, d_v_pool, d_table, g.visible, 0,
                                                g.block_size, kv_dim, g.max_num_blocks, 0);
        check(cudaDeviceSynchronize(), "scatter sync");
        check(cudaFree(d_tmp_k), "cudaFree tmp_k");
        check(cudaFree(d_tmp_v), "cudaFree tmp_v");
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(kDpaHeadDim));

    auto runPath = [&](DpaPath p) {
        switch (p) {
        case kLegacy:
            tiny_llm::kernels::paged_gather_blocks(d_k_scratch, d_k_pool, d_table, g.visible,
                                                   g.block_size, kv_dim, g.max_num_blocks, 0);
            tiny_llm::kernels::paged_gather_blocks(d_v_scratch, d_v_pool, d_table, g.visible,
                                                   g.block_size, kv_dim, g.max_num_blocks, 0);
            tiny_llm::kernels::attention_decode(d_q, d_k_scratch, d_v_scratch, d_out_legacy, scale,
                                                kDpaHq, kDpaHkv, d_len, kDpaHeadDim, 0);
            break;
        case kContiguous:
            tiny_llm::kernels::attention_decode(d_q, d_k_scratch, d_v_scratch, d_out_contig, scale,
                                                kDpaHq, kDpaHkv, d_len, kDpaHeadDim, 0);
            break;
        case kDirect:
            tiny_llm::kernels::attention_decode_paged(
                d_q, d_k_pool, d_v_pool, d_table, d_out_direct, scale, kDpaHq, kDpaHkv, kDpaHeadDim,
                d_len, g.block_size, g.max_num_blocks, g.table_len, 0);
            break;
        case kGatherK:
            tiny_llm::kernels::paged_gather_blocks(d_k_scratch, d_k_pool, d_table, g.visible,
                                                   g.block_size, kv_dim, g.max_num_blocks, 0);
            break;
        case kGatherV:
            tiny_llm::kernels::paged_gather_blocks(d_v_scratch, d_v_pool, d_table, g.visible,
                                                   g.block_size, kv_dim, g.max_num_blocks, 0);
            break;
        default:
            break;
        }
    };

    // ── 先正确性、后计时：三条路径在同一输入上必须逐位一致（设计包 §8）─────────
    bool  equiv_ok = true;
    float max_abs_diff = 0.0f;
    {
        for (int p = kLegacy; p <= kDirect; ++p)
            runPath(static_cast<DpaPath>(p));
        check(cudaDeviceSynchronize(), "equivalence sync");

        std::vector<half> h_legacy(q_elems), h_direct(q_elems), h_contig(q_elems);
        check(cudaMemcpy(h_legacy.data(), d_out_legacy, q_elems * sizeof(half),
                         cudaMemcpyDeviceToHost),
              "copy out_legacy");
        check(cudaMemcpy(h_direct.data(), d_out_direct, q_elems * sizeof(half),
                         cudaMemcpyDeviceToHost),
              "copy out_direct");
        check(cudaMemcpy(h_contig.data(), d_out_contig, q_elems * sizeof(half),
                         cudaMemcpyDeviceToHost),
              "copy out_contig");

        equiv_ok = std::memcmp(h_legacy.data(), h_direct.data(), q_elems * sizeof(half)) == 0;
        for (size_t i = 0; i < q_elems; ++i) {
            max_abs_diff = std::max(
                max_abs_diff, std::fabs(__half2float(h_legacy[i]) - __half2float(h_direct[i])));
        }
        std::fprintf(
            out,
            "{\"type\":\"equivalence\",\"shape\":{\"visible_tokens\":%d,\"block_size\":%d},"
            "\"legacy_vs_direct_bitwise_equal\":%s,\"legacy_vs_direct_max_abs_diff\":%.6g,"
            "\"legacy_vs_contiguous_bitwise_equal\":%s}\n",
            g.visible, g.block_size, equiv_ok ? "true" : "false", max_abs_diff,
            std::memcmp(h_legacy.data(), h_contig.data(), q_elems * sizeof(half)) == 0 ? "true"
                                                                                       : "false");
        std::fflush(out);
    }

    // ── 预热（每次调用都跑，不带 sync）──────────────────────────────────────
    for (int p = 0; p < kNumPaths; ++p)
        for (int i = 0; i < args.warmup; ++i)
            runPath(static_cast<DpaPath>(p));
    check(cudaDeviceSynchronize(), "warmup sync");

    // ── 采样：路径按 round 轮转，抑制时钟漂移 ───────────────────────────────
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    check(cudaEventCreate(&e0), "cudaEventCreate");
    check(cudaEventCreate(&e1), "cudaEventCreate");

    std::vector<std::vector<std::vector<double>>> samples(
        static_cast<size_t>(kNumPaths),
        std::vector<std::vector<double>>(static_cast<size_t>(args.repeats)));

    auto timePath = [&](DpaPath p) {
        check(cudaEventRecord(e0, 0), "event record start");
        for (int i = 0; i < args.batch; ++i)
            runPath(p);
        check(cudaEventRecord(e1, 0), "event record stop");
        check(cudaEventSynchronize(e1), "event sync");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, e0, e1), "event elapsed");
        return static_cast<double>(ms) / static_cast<double>(args.batch);
    };

    for (int rep = 0; rep < args.repeats; ++rep) {
        for (int r = 0; r < rounds; ++r) {
            for (int off = 0; off < kNumPaths; ++off) {
                const int    p = (r + off) % kNumPaths;
                const double ms = timePath(static_cast<DpaPath>(p));
                samples[static_cast<size_t>(p)][static_cast<size_t>(rep)].push_back(ms);
                std::fprintf(out,
                             "{\"type\":\"sample\",\"shape\":{\"visible_tokens\":%d,"
                             "\"block_size\":%d},\"path\":\"%s\",\"repeat\":%d,\"order\":%d,"
                             "\"ms\":%.6f}\n",
                             g.visible, g.block_size, kDpaPathNames[p], rep, p, ms);
            }
        }
        std::fflush(out);
        std::fprintf(stderr, "dpa: visible=%d block=%d repeat %d/%d done\n", g.visible,
                     g.block_size, rep + 1, args.repeats);
    }
    check(cudaEventDestroy(e0), "cudaEventDestroy");
    check(cudaEventDestroy(e1), "cudaEventDestroy");

    // ── 聚合 + 收敛判定 ────────────────────────────────────────────────────
    // 收敛门禁只约束三条**对比路径**（legacy / contiguous / direct）。gather_k /
    // gather_v 是辅助诊断（§9：用于区分「省下的 gather」与 direct 自身开销），
    // 它们的单次耗时约 5 µs、落在 launch 噪声底之上，CV 天然偏高，单独报告但
    // 不参与 shape 级收敛判定。
    bool        converged = true;
    bool        gathers_converged = true;
    std::string not_converged_reason;
    for (int p = 0; p < kNumPaths; ++p) {
        const auto         &per_rep = samples[static_cast<size_t>(p)];
        std::vector<double> pooled;
        std::vector<double> rep_medians;
        for (const auto &v : per_rep) {
            pooled.insert(pooled.end(), v.begin(), v.end());
            rep_medians.push_back(dpaPercentile(v, 0.5));
        }
        const double cv = dpaCvPercent(pooled);
        const double pooled_median = dpaPercentile(pooled, 0.5);
        const bool   is_gather = (p == kGatherK || p == kGatherV);
        for (double m : rep_medians) {
            if (pooled_median > 0.0 &&
                std::fabs(m - pooled_median) / pooled_median * 100.0 > 10.0) {
                if (is_gather) {
                    gathers_converged = false;
                } else {
                    converged = false;
                    not_converged_reason =
                        std::string(kDpaPathNames[p]) + " repeat-median spread > 10%";
                }
            }
        }
        if (cv > 10.0) {
            if (is_gather) {
                gathers_converged = false;
            } else {
                converged = false;
                not_converged_reason = std::string(kDpaPathNames[p]) + " CV > 10%";
            }
        }
        std::fprintf(out,
                     "{\"type\":\"path_stats\",\"shape\":{\"visible_tokens\":%d,\"block_size\":%d},"
                     "\"path\":\"%s\",\"n\":%zu,\"median_ms\":%.6f,\"p10_ms\":%.6f,\"p90_ms\":%.6f,"
                     "\"mean_ms\":%.6f,\"cv_percent\":%.3f,\"repeat_medians_ms\":[",
                     g.visible, g.block_size, kDpaPathNames[p], pooled.size(), pooled_median,
                     dpaPercentile(pooled, 0.10), dpaPercentile(pooled, 0.90), dpaMean(pooled), cv);
        for (size_t i = 0; i < rep_medians.size(); ++i)
            std::fprintf(out, "%s%.6f", i == 0 ? "" : ",", rep_medians[i]);
        std::fprintf(out, "]}\n");
    }

    std::vector<double> legacy_all, direct_all, contig_all;
    for (const auto &v : samples[static_cast<size_t>(kLegacy)])
        legacy_all.insert(legacy_all.end(), v.begin(), v.end());
    for (const auto &v : samples[static_cast<size_t>(kDirect)])
        direct_all.insert(direct_all.end(), v.begin(), v.end());
    for (const auto &v : samples[static_cast<size_t>(kContiguous)])
        contig_all.insert(contig_all.end(), v.begin(), v.end());
    const double legacy_median = dpaPercentile(legacy_all, 0.5);
    const double direct_median = dpaPercentile(direct_all, 0.5);
    const double contig_median = dpaPercentile(contig_all, 0.5);

    std::fprintf(
        out,
        "{\"type\":\"shape_summary\",\"shape\":{\"visible_tokens\":%d,\"block_size\":%d,"
        "\"table_len\":%d,\"max_num_blocks\":%d},"
        "\"legacy_median_ms\":%.6f,\"contiguous_median_ms\":%.6f,\"direct_median_ms\":%.6f,"
        "\"speedup_direct_vs_legacy\":%.4f,\"speedup_direct_vs_contiguous\":%.4f,"
        "\"equiv_bitwise\":%s,\"converged\":%s,\"gathers_converged\":%s,"
        "\"not_converged_reason\":\"%s\"}\n",
        g.visible, g.block_size, g.table_len, g.max_num_blocks, legacy_median, contig_median,
        direct_median, direct_median > 0.0 ? legacy_median / direct_median : 0.0,
        direct_median > 0.0 ? contig_median / direct_median : 0.0, equiv_ok ? "true" : "false",
        converged ? "true" : "false", gathers_converged ? "true" : "false",
        dpaJsonEscape(not_converged_reason).c_str());
    std::fflush(out);

    check(cudaFree(d_k_pool), "cudaFree k_pool");
    check(cudaFree(d_v_pool), "cudaFree v_pool");
    check(cudaFree(d_k_scratch), "cudaFree k_scratch");
    check(cudaFree(d_v_scratch), "cudaFree v_scratch");
    check(cudaFree(d_q), "cudaFree q");
    check(cudaFree(d_out_legacy), "cudaFree out_legacy");
    check(cudaFree(d_out_contig), "cudaFree out_contig");
    check(cudaFree(d_out_direct), "cudaFree out_direct");
    check(cudaFree(d_table), "cudaFree table");
    check(cudaFree(d_len), "cudaFree len");
}

int runDpaBench(const DpaArgs &args) {
    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::fprintf(stderr, "kernel_bench: no CUDA device found\n");
        return 1;
    }
    check(cudaSetDevice(0), "cudaSetDevice");

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    int driver_version = 0, runtime_version = 0;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);

    FILE *out = stdout;
    if (args.out_path != nullptr) {
        out = std::fopen(args.out_path, "w");
        if (out == nullptr) {
            std::perror("kernel_bench: fopen");
            return 1;
        }
    }

    const std::vector<ShapeGeom> all_shapes = dpaShapes();
    std::vector<ShapeGeom>       shapes;
    for (const auto &g : all_shapes) {
        if (args.only_visible != 0 && g.visible != args.only_visible) continue;
        if (args.only_block_size != 0 && g.block_size != args.only_block_size) continue;
        shapes.push_back(g);
    }
    if (shapes.empty()) {
        std::fprintf(stderr, "kernel_bench: no shape matches the requested filter\n");
        return 1;
    }
    const int rounds = std::max(1, args.reps / std::max(1, args.batch));

    dpaEmitProvenance(out, args, prop, driver_version, runtime_version, shapes);
    dpaClockWarmup(args.clock_warmup_s);

    for (const auto &g : shapes)
        dpaRunShape(out, args, g, rounds);

    if (out != stdout) {
        std::fclose(out);
    }
    return 0;
}

DpaArgs parseDpaArgs(int argc, char **argv) {
    DpaArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto              next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "kernel_bench: %s requires a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--dpa-bench") {
            args.enabled = true;
        } else if (a == "--warmup") {
            args.warmup = std::stoi(next("--warmup"));
        } else if (a == "--reps") {
            args.reps = std::stoi(next("--reps"));
        } else if (a == "--batch") {
            args.batch = std::stoi(next("--batch"));
        } else if (a == "--repeats") {
            args.repeats = std::stoi(next("--repeats"));
        } else if (a == "--clock-warmup") {
            args.clock_warmup_s = std::stod(next("--clock-warmup"));
        } else if (a == "--seed") {
            args.seed = static_cast<unsigned>(std::stoul(next("--seed")));
        } else if (a == "--only-visible") {
            args.only_visible = std::stoi(next("--only-visible"));
        } else if (a == "--only-block-size") {
            args.only_block_size = std::stoi(next("--only-block-size"));
        } else if (a == "--out") {
            args.out_path = argv[++i];
        } else if (a == "--meta") {
            const std::string kv = next("--meta");
            const size_t      eq = kv.find('=');
            if (eq == std::string::npos) {
                std::fprintf(stderr, "kernel_bench: --meta expects key=value\n");
                std::exit(2);
            }
            args.meta.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        } else if (a == "--help" || a == "-h") {
            std::fprintf(stderr,
                         "usage: tiny_llm_kernel_bench [--dpa-bench] [--warmup N] [--reps N]\n"
                         "       [--batch N] [--repeats N] [--clock-warmup SECONDS] [--seed N]\n"
                         "       [--only-visible N] [--only-block-size N] [--out PATH]\n"
                         "       [--meta key=value]\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "kernel_bench: unknown argument \"%s\" (see --help)\n", a.c_str());
            std::exit(2);
        }
    }
    return args;
}

} // namespace

int main(int argc, char **argv) {
    const DpaArgs dpa = parseDpaArgs(argc, argv);
    if (dpa.enabled) return runDpaBench(dpa);

    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::fprintf(stderr, "kernel_bench: no CUDA device found\n");
        return 1;
    }
    check(cudaSetDevice(0), "cudaSetDevice");

    std::printf("name,shape,ms\n");

    // W8A16 GEMM: M=1, K=896, N in {128, 896, 4864}
    benchW8A16("w8a16_matmul", "M=1,K=896,N=128", 1, 128, 896, 128, 20, 200);
    benchW8A16("w8a16_matmul", "M=1,K=896,N=896", 1, 896, 896, 128, 20, 200);
    benchW8A16("w8a16_matmul", "M=1,K=896,N=4864", 1, 4864, 896, 128, 20, 200);
    // W8A16 GEMM down: M=1, K=4864, N=896
    benchW8A16("w8a16_matmul", "M=1,K=4864,N=896", 1, 896, 4864, 128, 20, 200);

    // FP16 lm_head: M=1, K=896, N=151936 (100 iters)
    benchFP16("fp16_matmul", "M=1,K=896,N=151936", 1, 151936, 896, 20, 100);

    // attention_decode: S in {8, 32, 64, 128}, Hq=14, Hkv=2, D=64
    for (int S : {8, 32, 64, 128}) {
        benchAttentionDecode(S, 14, 2, 64, 20, 200);
    }

    // rmsnorm / RoPE / add / silu_mul
    benchRMSNorm(1, 896, 20, 200);
    benchRoPE(14, 2, 64, 20, 200);
    benchElementwise("add_inplace", 896, 20, 200);
    benchElementwise("silu_mul_inplace", 4864, 20, 200);

    return 0;
}
