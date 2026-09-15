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
// TLLM-ATTN-SPLITKV PR-D：--num-splits 把每条基准路径扩展出 *_splitkv 变体
// （contiguous_splitkv / direct_splitkv / legacy_splitkv），schema 升 v2。
//
// 只做 kernel 级测量，不产生 TTFT/TPOT。路径消费同一份逻辑 K/V：
//   legacy     = paged_gather_blocks(K) + paged_gather_blocks(V) + attention_decode
//   contiguous = attention_decode（连续 scratch，仅作上界参考）
//   direct     = attention_decode_paged
//   *_splitkv  = 同取址方式、attention_decode[_paged]_splitkv（partial + combine）
// gather_k / gather_v 单独计时，用于区分「省下的 gather」与「direct 自身开销」。
//
// §10.1 记录的三个测量陷阱在本文件内规避：
//   1. GPU 空闲时 SM 时钟停在低频、小 kernel 推不动 boost → 计时前先做持续负载预热；
//   2. host-int 重载每次调用附带一次 4 字节 H2D memcpy → 一律走 device-int 重载；
//   3. 误编到 sm_75 而设备是 sm_120 → 构建必须用 -DCMAKE_CUDA_ARCHITECTURES=native；
//      程序把设备 compute capability 写进 provenance，便于一眼核对。
//
// 采样：每个 sample 计时 batch 次调用再除以 batch（摊薄 launch 开销）；路径按 round
// 轮转顺序交替（order-balanced，抑制时钟漂移）；报告 median / p10 / p90 / CV。
// CV > 10% 或重复间 median 差异 > 10% 的行标 not_converged（只约束三条对比路径），
// 但不丢弃。
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
    std::vector<int>                                 num_splits;
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

enum DpaPath { kLegacy = 0, kContiguous, kDirect, kGatherK, kGatherV };
const char *const kDpaPathNames[] = {"legacy", "contiguous", "direct", "gather_k", "gather_v"};

// TLLM-ATTN-SPLITKV PR-D：split-KV 变体按 (kind, num_splits) 参数化。
// num_splits == 0 表示非 split 的单遍路径；>0 走 *_splitkv 入口（含 num_splits == 1，
// 用于直接量出「多一次 combine launch」的纯开销，回答设计包 §2 的未知数 3）。
struct DpaPathSpec {
    DpaPath     kind;
    int         num_splits;
    std::string name; // "legacy" / "contiguous_splitkv" / ...
};

std::vector<DpaPathSpec> dpaPaths(const DpaArgs &args) {
    std::vector<DpaPathSpec> paths = {{kLegacy, 0, "legacy"},
                                      {kContiguous, 0, "contiguous"},
                                      {kDirect, 0, "direct"},
                                      {kGatherK, 0, "gather_k"},
                                      {kGatherV, 0, "gather_v"}};
    std::vector<int>         uniq;
    for (int ns : args.num_splits) {
        if (ns < 1) {
            std::fprintf(stderr, "kernel_bench: --num-splits must be >= 1 (got %d)\n", ns);
            std::exit(2);
        }
        if (std::find(uniq.begin(), uniq.end(), ns) == uniq.end()) uniq.push_back(ns);
    }
    for (int ns : uniq) {
        paths.push_back({kContiguous, ns, "contiguous_splitkv"});
        paths.push_back({kDirect, ns, "direct_splitkv"});
        paths.push_back({kLegacy, ns, "legacy_splitkv"});
    }
    return paths;
}

// splitkv 路径的语义基准：同 kind 的单遍路径（§8：num_splits=1 必须逐位相同，
// >1 落在 oracle 容差内——此处与 §9 的计时门禁共用同一组对照）。
DpaPath dpaBaseKind(DpaPath kind) {
    switch (kind) {
    case kLegacy:
        return kLegacy;
    case kDirect:
        return kDirect;
    default:
        return kContiguous;
    }
}

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

    // v2 = v1 + num_splits sweep（TLLM-ATTN-SPLITKV PR-D）：sample/path_stats/
    // shape_summary 记录多一个 "num_splits" 字段（非 split 路径为 0）。
    std::fprintf(out, "{\"type\":\"provenance\",\"schema\":\"tllm-dpa-kernel-bench-v2\"");
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
    if (!args.num_splits.empty()) {
        std::fprintf(out, ",\"num_splits_sweep\":[");
        for (size_t i = 0; i < args.num_splits.size(); ++i)
            std::fprintf(out, "%s%d", i == 0 ? "" : ",", args.num_splits[i]);
        std::fprintf(out, "]");
    }
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

    const std::vector<DpaPathSpec> paths = dpaPaths(args);
    const size_t                   npaths = paths.size();

    half *d_k_pool = nullptr, *d_v_pool = nullptr, *d_k_scratch = nullptr, *d_v_scratch = nullptr;
    half *d_q = nullptr;
    int  *d_table = nullptr, *d_len = nullptr;
    check(cudaMalloc(&d_k_pool, pool_elems * sizeof(half)), "cudaMalloc k_pool");
    check(cudaMalloc(&d_v_pool, pool_elems * sizeof(half)), "cudaMalloc v_pool");
    check(cudaMalloc(&d_k_scratch, pool_elems * sizeof(half)), "cudaMalloc k_scratch");
    check(cudaMalloc(&d_v_scratch, pool_elems * sizeof(half)), "cudaMalloc v_scratch");
    check(cudaMalloc(&d_q, q_elems * sizeof(half)), "cudaMalloc q");
    // 每条路径一个输出缓冲，等价性检查需要同时拿到所有路径的输出。
    std::vector<half *> d_out(npaths, nullptr);
    for (size_t i = 0; i < npaths; ++i)
        check(cudaMalloc(&d_out[i], q_elems * sizeof(half)), "cudaMalloc out");
    check(cudaMalloc(&d_table, static_cast<size_t>(g.max_num_blocks) * sizeof(int)),
          "cudaMalloc table");
    check(cudaMalloc(&d_len, sizeof(int)), "cudaMalloc len");

    // split-KV partial 缓冲：按本 shape 内最大的 num_splits 一次性分配，
    // 所有 split 路径复用（调用串行、同 stream，无并发）。布局见
    // docs/architecture/decode-attention-splitkv-design.md §4.3。
    float *d_partial = nullptr;
    int    max_splits = 0;
    for (const auto &p : paths)
        max_splits = std::max(max_splits, p.num_splits);
    if (max_splits > 0) {
        const size_t floats =
            static_cast<size_t>(kDpaHq) * static_cast<size_t>(max_splits) * (2 + kDpaHeadDim);
        check(cudaMalloc(&d_partial, floats * sizeof(float)), "cudaMalloc partial");
    }

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

    auto runPath = [&](size_t pi) {
        const DpaPathSpec &s = paths[pi];
        switch (s.kind) {
        case kLegacy:
            tiny_llm::kernels::paged_gather_blocks(d_k_scratch, d_k_pool, d_table, g.visible,
                                                   g.block_size, kv_dim, g.max_num_blocks, 0);
            tiny_llm::kernels::paged_gather_blocks(d_v_scratch, d_v_pool, d_table, g.visible,
                                                   g.block_size, kv_dim, g.max_num_blocks, 0);
            if (s.num_splits > 0) {
                tiny_llm::kernels::attention_decode_splitkv(
                    d_q, d_k_scratch, d_v_scratch, d_out[pi], scale, kDpaHq, kDpaHkv, d_len,
                    kDpaHeadDim, d_partial, s.num_splits, 0);
            } else {
                tiny_llm::kernels::attention_decode(d_q, d_k_scratch, d_v_scratch, d_out[pi], scale,
                                                    kDpaHq, kDpaHkv, d_len, kDpaHeadDim, 0);
            }
            break;
        case kContiguous:
            if (s.num_splits > 0) {
                tiny_llm::kernels::attention_decode_splitkv(
                    d_q, d_k_scratch, d_v_scratch, d_out[pi], scale, kDpaHq, kDpaHkv, d_len,
                    kDpaHeadDim, d_partial, s.num_splits, 0);
            } else {
                tiny_llm::kernels::attention_decode(d_q, d_k_scratch, d_v_scratch, d_out[pi], scale,
                                                    kDpaHq, kDpaHkv, d_len, kDpaHeadDim, 0);
            }
            break;
        case kDirect:
            if (s.num_splits > 0) {
                tiny_llm::kernels::attention_decode_paged_splitkv(
                    d_q, d_k_pool, d_v_pool, d_table, d_out[pi], scale, kDpaHq, kDpaHkv,
                    kDpaHeadDim, d_len, g.block_size, g.max_num_blocks, g.table_len, d_partial,
                    s.num_splits, 0);
            } else {
                tiny_llm::kernels::attention_decode_paged(
                    d_q, d_k_pool, d_v_pool, d_table, d_out[pi], scale, kDpaHq, kDpaHkv,
                    kDpaHeadDim, d_len, g.block_size, g.max_num_blocks, g.table_len, 0);
            }
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

    // ── 先正确性、后计时：所有路径在同一输入上对比（设计包 §8）───────────────
    // 单遍三路：legacy/direct 必须逐位一致（v1 门禁原样保留为 equiv_bitwise）。
    // split 路径：对照同 kind 的单遍输出；num_splits==1 必须逐位相同（锚点），
    // num_splits>1 允许 fp32 求和顺序差异（oracle 容差 2e-3，记录 max|diff|）。
    bool equiv_bitwise = true;
    bool equiv_ok = true;
    {
        for (size_t pi = 0; pi < npaths; ++pi)
            runPath(pi);
        check(cudaDeviceSynchronize(), "equivalence sync");

        std::vector<std::vector<half>> h_out(npaths, std::vector<half>(q_elems));
        for (size_t pi = 0; pi < npaths; ++pi)
            check(cudaMemcpy(h_out[pi].data(), d_out[pi], q_elems * sizeof(half),
                             cudaMemcpyDeviceToHost),
                  "copy out");

        // 单遍基准路径在 paths 中的固定下标（dpaPaths 构造顺序）。
        const size_t i_legacy = 0, i_contig = 1, i_direct = 2;
        equiv_bitwise = std::memcmp(h_out[i_legacy].data(), h_out[i_direct].data(),
                                    q_elems * sizeof(half)) == 0;
        equiv_ok = equiv_bitwise;
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < q_elems; ++i) {
            max_abs_diff = std::max(max_abs_diff, std::fabs(__half2float(h_out[i_legacy][i]) -
                                                            __half2float(h_out[i_direct][i])));
        }
        std::fprintf(
            out,
            "{\"type\":\"equivalence\",\"shape\":{\"visible_tokens\":%d,\"block_size\":%d},"
            "\"legacy_vs_direct_bitwise_equal\":%s,\"legacy_vs_direct_max_abs_diff\":%.6g,"
            "\"legacy_vs_contiguous_bitwise_equal\":%s}\n",
            g.visible, g.block_size, equiv_bitwise ? "true" : "false", max_abs_diff,
            std::memcmp(h_out[i_legacy].data(), h_out[i_contig].data(), q_elems * sizeof(half)) == 0
                ? "true"
                : "false");

        // 每个 split 路径一条 equivalence_splitkv 记录；ref 是同 kind 单遍路径。
        for (size_t pi = 0; pi < npaths; ++pi) {
            const DpaPathSpec &s = paths[pi];
            if (s.num_splits == 0) continue;
            const DpaPath ref_kind = dpaBaseKind(s.kind);
            size_t        ref = npaths;
            for (size_t r = 0; r < npaths; ++r)
                if (paths[r].kind == ref_kind && paths[r].num_splits == 0) ref = r;
            if (ref == npaths) continue;
            const bool bitwise =
                std::memcmp(h_out[pi].data(), h_out[ref].data(), q_elems * sizeof(half)) == 0;
            float diff = 0.0f;
            for (size_t i = 0; i < q_elems; ++i)
                diff = std::max(
                    diff, std::fabs(__half2float(h_out[pi][i]) - __half2float(h_out[ref][i])));
            // num_splits==1 的锚点失败与 >1 超出容差都计入 equiv_ok。
            const bool ok = s.num_splits == 1 ? bitwise : diff <= 2e-3f;
            equiv_ok = equiv_ok && ok;
            std::fprintf(out,
                         "{\"type\":\"equivalence_splitkv\",\"shape\":{\"visible_tokens\":%d,"
                         "\"block_size\":%d},\"path\":\"%s\",\"num_splits\":%d,\"ref_path\":\"%s\","
                         "\"bitwise_equal\":%s,\"max_abs_diff\":%.6g,\"within_tolerance\":%s}\n",
                         g.visible, g.block_size, s.name.c_str(), s.num_splits,
                         kDpaPathNames[ref_kind], bitwise ? "true" : "false", diff,
                         ok ? "true" : "false");
        }
        std::fflush(out);
    }

    // ── 预热（每次调用都跑，不带 sync）──────────────────────────────────────
    for (size_t pi = 0; pi < npaths; ++pi)
        for (int i = 0; i < args.warmup; ++i)
            runPath(pi);
    check(cudaDeviceSynchronize(), "warmup sync");

    // ── 采样：路径按 round 轮转，抑制时钟漂移 ───────────────────────────────
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    check(cudaEventCreate(&e0), "cudaEventCreate");
    check(cudaEventCreate(&e1), "cudaEventCreate");

    std::vector<std::vector<std::vector<double>>> samples(
        npaths, std::vector<std::vector<double>>(static_cast<size_t>(args.repeats)));

    auto timePath = [&](size_t pi) {
        check(cudaEventRecord(e0, 0), "event record start");
        for (int i = 0; i < args.batch; ++i)
            runPath(pi);
        check(cudaEventRecord(e1, 0), "event record stop");
        check(cudaEventSynchronize(e1), "event sync");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, e0, e1), "event elapsed");
        return static_cast<double>(ms) / static_cast<double>(args.batch);
    };

    for (int rep = 0; rep < args.repeats; ++rep) {
        for (int r = 0; r < rounds; ++r) {
            for (size_t off = 0; off < npaths; ++off) {
                const size_t pi = (static_cast<size_t>(r) + off) % npaths;
                const double ms = timePath(pi);
                samples[pi][static_cast<size_t>(rep)].push_back(ms);
                std::fprintf(out,
                             "{\"type\":\"sample\",\"shape\":{\"visible_tokens\":%d,"
                             "\"block_size\":%d},\"path\":\"%s\",\"num_splits\":%d,\"repeat\":%d,"
                             "\"order\":%zu,\"ms\":%.6f}\n",
                             g.visible, g.block_size, paths[pi].name.c_str(), paths[pi].num_splits,
                             rep, pi, ms);
            }
        }
        std::fflush(out);
        std::fprintf(stderr, "dpa: visible=%d block=%d repeat %d/%d done\n", g.visible,
                     g.block_size, rep + 1, args.repeats);
    }
    check(cudaEventDestroy(e0), "cudaEventDestroy");
    check(cudaEventDestroy(e1), "cudaEventDestroy");

    // ── 聚合 + 收敛判定 ────────────────────────────────────────────────────
    // 收敛门禁约束所有**被比较路径**：三条单遍路径 + 全部 splitkv 变体
    // （splitkv 是本 benchmark 的测量对象）。gather_k / gather_v 仍是辅助诊断
    // （§9：用于区分「省下的 gather」与 direct 自身开销），单次约 5 µs、落在
    // launch 噪声底之上，CV 天然偏高，单独报告但不参与 shape 级收敛判定。
    bool                converged = true;
    bool                gathers_converged = true;
    std::string         not_converged_reason;
    std::vector<double> medians(npaths, 0.0);
    for (size_t pi = 0; pi < npaths; ++pi) {
        const auto         &per_rep = samples[pi];
        std::vector<double> pooled;
        std::vector<double> rep_medians;
        for (const auto &v : per_rep) {
            pooled.insert(pooled.end(), v.begin(), v.end());
            rep_medians.push_back(dpaPercentile(v, 0.5));
        }
        const double cv = dpaCvPercent(pooled);
        const double pooled_median = dpaPercentile(pooled, 0.5);
        medians[pi] = pooled_median;
        const bool        is_gather = (paths[pi].kind == kGatherK || paths[pi].kind == kGatherV);
        const std::string pname = paths[pi].num_splits > 0
                                      ? paths[pi].name + "@" + std::to_string(paths[pi].num_splits)
                                      : paths[pi].name;
        for (double m : rep_medians) {
            if (pooled_median > 0.0 &&
                std::fabs(m - pooled_median) / pooled_median * 100.0 > 10.0) {
                if (is_gather) {
                    gathers_converged = false;
                } else {
                    converged = false;
                    not_converged_reason = pname + " repeat-median spread > 10%";
                }
            }
        }
        if (cv > 10.0) {
            if (is_gather) {
                gathers_converged = false;
            } else {
                converged = false;
                not_converged_reason = pname + " CV > 10%";
            }
        }
        std::fprintf(out,
                     "{\"type\":\"path_stats\",\"shape\":{\"visible_tokens\":%d,\"block_size\":%d},"
                     "\"path\":\"%s\",\"num_splits\":%d,\"n\":%zu,\"median_ms\":%.6f,"
                     "\"p10_ms\":%.6f,\"p90_ms\":%.6f,"
                     "\"mean_ms\":%.6f,\"cv_percent\":%.3f,\"repeat_medians_ms\":[",
                     g.visible, g.block_size, paths[pi].name.c_str(), paths[pi].num_splits,
                     pooled.size(), pooled_median, dpaPercentile(pooled, 0.10),
                     dpaPercentile(pooled, 0.90), dpaMean(pooled), cv);
        for (size_t i = 0; i < rep_medians.size(); ++i)
            std::fprintf(out, "%s%.6f", i == 0 ? "" : ",", rep_medians[i]);
        std::fprintf(out, "]}\n");
    }

    // 单遍三路在 paths 中的固定下标（dpaPaths 构造顺序）。
    const double legacy_median = medians[0];
    const double contig_median = medians[1];
    const double direct_median = medians[2];

    std::fprintf(
        out,
        "{\"type\":\"shape_summary\",\"shape\":{\"visible_tokens\":%d,\"block_size\":%d,"
        "\"table_len\":%d,\"max_num_blocks\":%d},"
        "\"legacy_median_ms\":%.6f,\"contiguous_median_ms\":%.6f,\"direct_median_ms\":%.6f,"
        "\"speedup_direct_vs_legacy\":%.4f,\"speedup_direct_vs_contiguous\":%.4f,",
        g.visible, g.block_size, g.table_len, g.max_num_blocks, legacy_median, contig_median,
        direct_median, direct_median > 0.0 ? legacy_median / direct_median : 0.0,
        direct_median > 0.0 ? contig_median / direct_median : 0.0);
    // splitkv 变体的中位数全部进 shape_summary，报告端按 num_splits 重建矩阵。
    if (npaths > 5) {
        std::fprintf(out, "\"splitkv_medians\":[");
        bool first = true;
        for (size_t pi = 0; pi < npaths; ++pi) {
            if (paths[pi].num_splits == 0) continue;
            std::fprintf(out, "%s{\"path\":\"%s\",\"num_splits\":%d,\"median_ms\":%.6f}",
                         first ? "" : ",", paths[pi].name.c_str(), paths[pi].num_splits,
                         medians[pi]);
            first = false;
        }
        std::fprintf(out, "],");
    }
    std::fprintf(out,
                 "\"equiv_bitwise\":%s,\"equiv_ok\":%s,\"converged\":%s,"
                 "\"gathers_converged\":%s,\"not_converged_reason\":\"%s\"}\n",
                 equiv_bitwise ? "true" : "false", equiv_ok ? "true" : "false",
                 converged ? "true" : "false", gathers_converged ? "true" : "false",
                 dpaJsonEscape(not_converged_reason).c_str());
    std::fflush(out);

    check(cudaFree(d_k_pool), "cudaFree k_pool");
    check(cudaFree(d_v_pool), "cudaFree v_pool");
    check(cudaFree(d_k_scratch), "cudaFree k_scratch");
    check(cudaFree(d_v_scratch), "cudaFree v_scratch");
    check(cudaFree(d_q), "cudaFree q");
    for (size_t pi = 0; pi < npaths; ++pi)
        check(cudaFree(d_out[pi]), "cudaFree out");
    check(cudaFree(d_table), "cudaFree table");
    check(cudaFree(d_len), "cudaFree len");
    if (d_partial != nullptr) check(cudaFree(d_partial), "cudaFree partial");
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
        } else if (a == "--num-splits") {
            // TLLM-ATTN-SPLITKV PR-D：可重复，也接受逗号分隔（如 2,4,8,16）。
            // 每个取值生成 contiguous_splitkv / direct_splitkv / legacy_splitkv
            // 三条路径；含 1 时量出纯 combine launch 开销。不给则不跑 splitkv。
            const std::string v = next("--num-splits");
            size_t            pos = 0;
            while (pos <= v.size()) {
                const size_t      comma = v.find(',', pos);
                const std::string tok =
                    v.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!tok.empty()) args.num_splits.push_back(std::stoi(tok));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
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
                         "       [--num-splits N[,M,...]] [--meta key=value]\n");
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
