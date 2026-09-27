# Tiny-LLM 历史开发计划（2026-08 快照）

> **归档说明（2026-08-24）**：本文保留当时的逐任务实施上下文、代码草案与验收命令，
> 不再描述当前代码状态。文中“未实现”“待做”和测试数量均是历史快照，不能作为
> README、简历或新开发任务的依据。
>
> 当前能力与进行中的工作以 `README.md`、`ROADMAP.md` 和
> `docs/performance/results/` 为准；可引用的 CUDA Graph 结果仅是
> `2026-08-23-cuda-graphs-ab.md` 所定义的 **CUDA Graph decode A/B**，不代表整体推理加速。
>
> 本文写成时的基线核验：147 个测试，141 通过，6 跳过（跳过项需要
> `TLLM_GGUF_TEST_MODEL` 指向真实 GGUF 文件）。

---

## 0. 执行协议（先读，每条都要遵守）

1. **单任务单提交**：一个任务只改该任务列出的文件，完成后 `git commit` 一次。
2. **验收命令必须全绿**：每个任务结尾的 build + test 命令必须通过，失败就继续修，
   不允许跳过或注释掉测试。
3. **禁止通过改测试让错误实现通过**。只有当产品行为**有意变更**时（例如
   `advanceSeqLen` 从静默截断改为返回错误），才允许同步修改测试，且必须在
   commit message 中说明。
4. **禁区**（改动风险高、收益低，除非任务明确要求，否则不要碰）：
   - `src/tokenizer.cpp` / `scripts/gen_unicode_tables.py`：已与 HuggingFace 逐 id 对齐。
   - `src/gguf_parser.cpp` / `src/quantization.cpp` 的反量化格式：已与 Python gguf 参考对齐。
   - KV Cache 的内存布局：`ffi.h` 的 C ABI 契约依赖它。
5. **保持 C++17 / CUDA 11+ 兼容**：不要引入 C++20 特性；kernel 使用
   `__expf/__half2float/__float2half`，不要用 `std::` 数学函数。
6. **提交前删除临时文件**：不要提交 `.bak`、`*.tmp`、core dump。
7. 所有新代码沿用 `tiny_llm` 命名空间和 `Result<T>` 错误处理风格。

---

## 1. 历史基线快照（2026-08，非当前状态）

### 1.1 上一轮已完成的修复（不要重复做）

| 修复 | 位置 | 验证状态 |
|---|---|---|
| Attention online-softmax（shared memory O(1)，不再随 seq_len 增长，数值等价） | `kernels/attention.cu` | 已有 `LongSequenceDecodeMatchesCpuReference`（seq_len=3000）测试通过 |
| W8A16 GEMM `M==1` decode 快速路径（每 warp 一列，32 lane 归约 K） | `kernels/w8a16_matmul.cu` | `W8A16MatMulTest` 全通过 |
| 采样改为 thread_local RNG + CDF 二分查找（不再每次构造 `discrete_distribution`） | `src/inference_engine.cpp` | `InferenceEngineTest` 采样测试全通过 |
| `advanceSeqLen` 溢出从静默 clamp 改为返回错误；`allocateSequence` 清零复用的 slot | `src/kv_cache.cpp` | `KVCacheTest` 全通过（测试已同步改为 `AdvanceSeqLenFailsOnOverflow`） |
| 长序列 decode attention 数值测试 | `tests/test_kernels.cu` | 通过 |

### 1.2 当时的已知半成品（后续已完成，保留实施背景）

- `kernels/w8a16_matmul.cuh` **已经声明** `fp16_matmul(...)`，但 `.cu` 中**没有实现**，
  也没有任何调用方使用它。
- 当前 `lm_head`（FP16，`[1, hidden] @ [hidden, vocab]`，vocab=151936）仍走
  `fp16_matmul_reference`：每个输出元素一个线程、串行归约 K，这是 decode 阶段
  最明显的剩余瓶颈之一。`src/inference_engine.cpp:430` 和 `src/ffi.cpp:80`
  两处调用点需要切换。

### 1.3 测试基线命令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/tiny_llm_tests
# 期望：147 tests, 141 passed, 6 skipped
```

真实模型相关测试（可选，有模型时跑）：

```bash
export TLLM_GGUF_TEST_MODEL=/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf
./build/tiny_llm_tests
```

---

## 2. 任务总览与依赖

```
阶段 1（P0，先做）：收尾半成品
  1.1 实现 fp16_matmul decode fast path 并接线        ← 当前代码已有声明缺实现
  1.2 attention prefill 长序列测试与 tile 边界测试

阶段 2（P0）：benchmark 与对比基线
  2.1 benchmark 驱动（TTFT / TPOT / tok/s / 常驻显存差值；峰值显存外部测量）
  2.2 llama.cpp 对比方法论文档
  2.3 nsys/ncu profiling 指南与结果归档模板
        依赖：真实 GGUF 模型（TLLM_GGUF_TEST_MODEL 或 CLI 参数）

阶段 3（P1）：decode 优化主题 —— CUDA Graphs
  3.1 attention_decode 的 visible_len 参数间接化（graph 可重放的准备）
  3.2 decode graph capture/replay + 环境变量开关
  3.3 正确性差分测试 + benchmark before/after
        依赖：2.1

阶段 4（P1）：正确性与工程完整性
  4.1 失败路径审计与测试（损坏文件 / OOM / 超长输入）
  4.2 GQA/MQA 第二个真实模型验证
  4.3 FFI 执行路径一致性收口（低风险重构）
  4.4 与 paged-serving 对接 ✅（ABI v2 + 分页 KV 策略 1 已启用，见 ROADMAP/README）

完成定义：见第 7 节 checklist。
```

---

## 3. 阶段 1：收尾半成品（P0）

### 任务 1.1 实现 `fp16_matmul` decode fast path 并接线

**背景**：`fp16_matmul` 已在头文件声明。decode 时 `M==1`，输出列数 N 可达
vocab_size=151936；reference kernel 每个输出元素一个线程、K 全串行，速度差。
仿照已实现的 `w8a16_matmul_m1_kernel`：每个 warp 负责一个输出列，32 个 lane
沿 K 维并行归约。

**改动文件**：

1. `kernels/w8a16_matmul.cu`
2. `src/inference_engine.cpp`
3. `src/ffi.cpp`

**实施步骤**：

#### 步骤 1：在 `kernels/w8a16_matmul.cu` 中插入实现

插入位置：`fp16_matmul_reference(...)` 函数**结束之后**、
`// W8A16 reference kernel` 注释之前（当前约第 72 行）。

参考实现（可直接使用）：

```cuda
// Decode-optimized FP16 GEMM for M == 1.  Same warp-per-output-column scheme
// as w8a16_matmul_m1_kernel; this keeps the fp16 lm_head path from becoming
// the decode bottleneck for large vocabularies.
__global__ void fp16_matmul_m1_kernel(const half *__restrict__ input,
                                      const half *__restrict__ weight,
                                      half *__restrict__ output, int N, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    if (col >= N) return;

    const int lane = threadIdx.x & 31;
    float sum = 0.0f;

    for (int k = lane; k < K; k += 32) {
        float a = __half2float(input[k]);
        float w = __half2float(weight[k * N + col]);
        sum += a * w;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        output[col] = __float2half(sum);
    }
}

void fp16_matmul(const half *input, const half *weight, half *output, int M, int N, int K,
                 cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    if (M == 1) {
        constexpr int WARPS_PER_BLOCK = 4; // 128 threads
        dim3 block(WARPS_PER_BLOCK * 32);
        dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        fp16_matmul_m1_kernel<<<grid, block, 0, stream>>>(input, weight, output, N, K);
        return;
    }

    fp16_matmul_reference(input, weight, output, M, N, K, stream);
}
```

注意：该文件已经 `#include "warp_utils.cuh"`（上一轮已加），`warp_reduce_sum`
可直接使用；不要重复 include。

#### 步骤 2：替换两个调用点

`src/inference_engine.cpp` 的 `InferenceEngine::computeLogits`（约 430 行）：

```cpp
// 替换前
kernels::fp16_matmul_reference(hidden_states, weights_.lm_head_fp16, logits, num_tokens,
                               config_.vocab_size, config_.hidden_dim, stream_);
// 替换后
kernels::fp16_matmul(hidden_states, weights_.lm_head_fp16, logits, num_tokens,
                     config_.vocab_size, config_.hidden_dim, stream_);
```

`src/ffi.cpp` 的 `sample_from_hidden`（约 80 行）：

```cpp
// 替换前
tiny_llm::kernels::fp16_matmul_reference(hidden, h->weights.lm_head_fp16, h->logits_buf, 1,
                                         h->config.vocab_size, hidden_dim, h->stream);
// 替换后
tiny_llm::kernels::fp16_matmul(hidden, h->weights.lm_head_fp16, h->logits_buf, 1,
                               h->config.vocab_size, hidden_dim, h->stream);
```

#### 步骤 3（推荐）：为 `fp16_matmul` 的 M==1 路径补数值测试

在 `tests/test_w8a16_matmul.cu` 中新增一个测试：
- 构造 M=1、K=128、N=1024 的随机 FP16 输入/权重；
- 分别调用 `fp16_matmul` 与 CPU 参考（float 累加后转 half）；
- `EXPECT_NEAR` 容差建议 `1e-1f`（FP16 输出 + 不同归约顺序）。

**验收命令**：

```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests --gtest_filter='*W8A16*:*FFI*'
./build/tiny_llm_tests
# 期望：不比基线少通过任何测试；新增测试通过
```

**常见错误提示**：
- kernel 内不要使用 `std::exp` 等宿主函数。
- `blockDim.x / 32` 是运行时值，可以用于索引；`WARPS_PER_BLOCK` 必须在
  launcher 中固定为编译期常量。
- 不要修改 `fp16_matmul_reference` 本身——它是测试基准。

---

### 任务 1.2 attention prefill 长序列 + 非整 tile 边界测试

**背景**：上一轮只给 decode 加了长序列数值测试。prefill 的 online-softmax 路径
也需要覆盖，尤其是 `seq_len % 128 != 0` 的边界（如 1025 = 8×128 + 1）。

**改动文件**：`tests/test_kernels.cu`

**实施步骤**：

1. 参照已有 `LongSequenceDecodeMatchesCpuReference` 的写法（位于
   `GQADecodeMatchesCpuReference` 之后），新增
   `LongSequencePrefillMatchesCpuReference`。
2. 参数建议：
   - `num_q_heads = 4, num_kv_heads = 2, seq_len = 1025, head_dim = 64`
   - `scale = 1/sqrt(head_dim)`
   - Q/K/V 随机种子分别用 700/701/702
3. CPU 参考逻辑：
   - 对每个 `(query_pos, q_head)`：只对 `key_pos <= query_pos` 求 score；
   - 标准 softmax 后加权 V；
   - 与 GPU 输出逐元素比较，容差 `8e-2f`。
4. 只比较全部 head × head_dim 输出（1025×4×64 的 CPU 参考是 O(1025²×64×4)，
   约 2.7 亿次乘加，单测可接受；如果 CI 过慢，改为只对
   `query_pos ∈ {0, 1, 127, 128, 129, 1024}` 做全维度比较，其余 query 只检查
   `has_nonzero`）。

**验收命令**：

```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests --gtest_filter='*Attention*'
# 期望：AttentionTest 全部通过，新增 LongSequencePrefillMatchesCpuReference 通过
```

**常见错误提示**：
- 不要把 causal mask 写成 `key_pos < query_pos`，正确是 `<=`。
- 如果 GPU 输出与 CPU 参考误差大，先检查 `kv_stride = num_kv_heads * head_dim`
  和 token-major 布局索引，这是本项目最容易写错的索引。

---

## 4. 阶段 2：benchmark 与对比基线（P0）

> 阶段 2 依赖真实 GGUF 模型。没有模型时先完成任务 2.1 的代码和
> “无模型时报错清晰”路径，模型到位后再补 2.2/2.3 的实测数字。

### 任务 2.1 benchmark 驱动

**目标**：产出 `tiny_llm_bench` 可执行文件，输出可复现的标准表格。

**改动文件**：
- 新增 `src/benchmark.cpp`
- `CMakeLists.txt`

**CLI 设计**：

```text
tiny_llm_bench <model.gguf> --prompt "..." --max-tokens 128 \
    --warmup 3 --iters 10 [--json] [--use-reference]
```

**指标定义（必须照此实现，否则数字不可比）**：

| 指标 | 定义 |
|---|---|
| TTFT (ms) | 从调用 `generate()` 到第一个新 token 采样完成的墙钟时间（含 prefill + 第一次 logits） |
| TPOT (ms/token) | decode 阶段墙钟时间 ÷（生成 token 数 − 1）；只生成 1 个 token 时记 N/A |
| decode tok/s | 1 / TPOT × 1000 |
| 常驻显存差值 (MB) | 加载模型前与 benchmark 完成后各记录一次 `cudaMemGetInfo`，差值换算 MB；不得称为峰值 |
| prompt tokens / new tokens | 直接打印，供复现 |

如需报告峰值显存，必须另用外部采样器或 profiler，并同时记录采样频率；不能从两个
离散时间点推导峰值。

**统计口径**：
- warmup 3 次不统计；
- 正式迭代 10 次；
- 输出 mean / p50 / p95 / min / max；
- 每次迭代之间显式 `cudaDeviceSynchronize()`。

**实现要点**：
- 复用 `src/main.cpp` 的 `runGeneration` 思路（GGUF parse → tokenizer build →
  `InferenceEngine::load` → `generate`），但**不要复制采样和生成代码**，
  只调用公开 API。
- 计时使用 `std::chrono::steady_clock` 包住 `engine->generate(...)`；
  `GenerationStats` 里的 prefill/decode 时间可以作为交叉校验打印，但基准
  数字以墙钟为准。
- 每次迭代结束后销毁并重建 engine？**不需要**：`generate()` 自己会
  `releaseSequence`；重复调用同一 engine 即可。这样测的是稳定态。
- `--use-reference` 转发到 `kernels::g_force_reference`（在 `main.cpp` 已有
  类似开关，可参照）。
- `--json` 输出一行 JSON，方便脚本收集。

**CMake 修改**：

```cmake
add_executable(tiny_llm_bench src/benchmark.cpp)
target_link_libraries(tiny_llm_bench PRIVATE tiny_llm)
```

**验收命令**：

```bash
cmake --build build -j$(nproc)
./build/tiny_llm_bench 2>&1 | head -5
# 期望：无模型参数时打印 usage 并返回非 0；参数错误信息明确
```

有模型时：

```bash
./build/tiny_llm_bench /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --prompt "你好" --max-tokens 64 --warmup 3 --iters 10
# 期望：输出上述标准表格
```

**完成后再做**：把数字填到 README「项目状态」表，并把「端到端性能基准」
状态改为 ✅（ROADMAP 要求每完成一项就更新状态表）。

---

### 任务 2.2 llama.cpp 对比方法论文档

**改动文件**：新增 `docs/performance/benchmark-methodology.md`

**必须写清以下内容，面试和复现都会用到**：

1. 硬件：GPU 型号 / 驱动 / CUDA 版本 / 显存。
2. 软件：本仓库 commit hash、llama.cpp commit hash、编译选项。
3. 模型：Qwen2.5-0.5B-Instruct，GGUF Q4_K_M；说明 llama.cpp 是**直接按
   Q4_K_M 计算**，而 tiny-llm 是**反量化 → 转置 → 重新量化为 W8A16**。
4. 公平性声明：
   - 相同 prompt、相同 greedy 采样、相同 max_tokens；
   - llama.cpp 用 `llama-cli` 或 `llama-bench`，记录 `-ngl 99`（全 GPU）；
   - 预热与迭代次数保持一致（llama.cpp 的 `-n, -r` 参数）。
5. 命令示例（可直接复制）。
6. 结果表模板：指标 × tiny-llm × llama.cpp × 比值。
7. 预期差距来源分析段（先留空或写假设）：GEMM 实现、attention 实现、
   continuous batching 缺失等。

**验收**：文档中的命令能原样复制执行；表格有真实数字（模型到位后）。

---

### 任务 2.3 profiling 指南与结果归档模板

**改动文件**：
- 新增 `docs/performance/profiling-guide.md`
- 新增 `docs/performance/results/TEMPLATE.md`

**内容要求**：

`profiling-guide.md` 给出可直接复制的命令：

```bash
# nsys 时间线
nsys profile -o reports/decode \
    ./build/tiny_llm_bench model.gguf --prompt "你好" --max-tokens 64 --warmup 1 --iters 3

# ncu 单 kernel 分析（示例：attention_decode）
ncu --set full --launch-count 3 --launch-skip 10 \
    ./build/tiny_llm_bench model.gguf --prompt "你好" --max-tokens 64
```

要求记录：
- kernel 时间分布表（matmul / attention / rmsnorm / rope / elementwise /
  memcpy 各占多少）；
- 每个 GEMM 的 M/N/K 和 grid/block 配置；
- decode 阶段 launch 数量与总 launch 开销（用于 CUDA Graphs 的 before/after）。

`results/TEMPLATE.md` 是结果归档模板：日期、环境、commit、原始日志路径、
结论摘要、瓶颈图。

**验收**：按文档执行一次 profiling 并把结果提交到 `docs/performance/results/`。

---

## 5. 阶段 3：CUDA Graphs 加速 decode（P1，推荐）

> 这是本项目最能讲成故事的优化。预期收益：decode 阶段减少大量 kernel
> launch 开销（24 层 × 每层约 10+ kernel），TPOT 改善 20–50%（取决于
> 模型大小和 GPU）。
>
> **难点**：attention 的 `visible_len` 每步 +1，是 kernel 参数；graph 捕获
> 后参数会过期。必须先把该参数“移出”kernel 参数列表，放到 device memory
> 里，graph 才能无更新重放。

### 任务 3.1 `visible_len` 参数间接化（graph 重放前置条件）

**目标**：让 `attention_decode` 从 GPU global memory 读取 `visible_len`，
而不是从 kernel 参数读取。graph 捕获后，host 只需更新一个 int 变量再 replay。

**改动文件**：
- `kernels/attention.cu`
- `kernels/attention.cuh`
- `src/transformer.cpp`
- `src/inference_engine.h` / `src/inference_engine.cpp`（新增 device int 缓冲）
- 所有直接调用 `attention_decode` 的测试（用新的封装签名，测试尽量少改）

**实施步骤**：

1. 在 `attention.cu` 中新增全局内存变量参数版本：

```cuda
// 新签名：visible_len 由 device 端变量提供，便于 CUDA Graph 重放。
void attention_decode(const half *query, const half *k_cache, const half *v_cache,
                      half *output, float scale, int num_q_heads, int num_kv_heads,
                      const int *device_visible_len, int head_dim, cudaStream_t stream);
```

   内部 kernel 不变（online-softmax 版本），launcher 把
   `device_visible_len` 指针传给 kernel；kernel 开头：
   `int visible_len = *device_visible_len;`。
   保留旧 6 参数版本作为薄封装（内部分配/使用一个临时值或直接改为
   传 `const int*` 的 wrapper），避免一次性改所有测试。

2. `InferenceEngine` 增加成员 `DeviceBuffer<int> decode_len_`（长度 1），
   构造时分配；`TransformerLayer::attention` 增加 `const int *device_len`
   参数（或复用现有指针参数，把 `position` 相关语义说清楚）。
   **推荐**：`TransformerLayer` 的方法签名增加一个
   `const int *decode_len` 参数，只在 `num_tokens == 1` 分支使用。

3. `attention_decode` 的 launcher 现在从 device 变量读取长度，调用点
   `transformer.cpp` 改为传入 `device_len`。prefill 路径不变。

**验收命令**：

```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests --gtest_filter='*Attention*:*Transformer*'
# 期望：全部通过，输出与改前一致
```

**常见错误**：
- `device_visible_len` 与 kernel 必须在同一 stream 上先被写入（
  `cudaMemcpyAsync` 到该 stream 或 replay 前同步）。
- 不要把 `device_len` 读进 shared memory 后不同步——kernel 开头读一次即可。

### 任务 3.2 decode graph capture / replay

**目标**：`TLLM_CUDA_GRAPHS=1` 时 decode 阶段使用 CUDA Graph 重放。

**改动文件**：
- `src/inference_engine.h` / `src/inference_engine.cpp`
- 可选：`src/benchmark.cpp` 增加 `--graphs` 开关

**设计要点**：

1. 开关：环境变量 `TLLM_CUDA_GRAPHS=1` 启用；默认关闭（保证老行为）。
   构造 `InferenceEngine` 时读取。
2. capture 范围（只 capture 确定性的 device 工作）：
   - `embedTokens`（单 token，input token id 先 `cudaMemcpyAsync` 到固定
     `DeviceBuffer<int> graph_token_`）
   - 24 层 `layer->forward(...)`（内部所有 GEMM / attention / RMSNorm /
     appendKV 的 D2D memcpy 都可 capture）
   - `finalNorm` + `computeLogits`
   - **不 capture**：`advanceSeqLen`（host 操作）、logits D2H copy、采样。
3. capture 时机：第一次 decode 时正常执行并同步 stream，然后
   `cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal)`，
   重放一次上述 device 序列（此时 `decode_len_` 的值是当前位置），
   `cudaStreamEndCapture`，`cudaGraphInstantiate`。
   之后每次 decode：
   - 写 `decode_len_` 的 device 值（长度 = 当前可见 KV 长度）；
   - `cudaMemcpyAsync` token id 到 `graph_token_`；
   - `cudaGraphLaunch`；
   - host 侧 `advanceSeqLen`；
   - D2H 读 logits + 采样。
4. **数值一致性**：graph 重放前后，`W8A16` 和 attention 输出必须逐 token
   一致（greedy）。如果不一致，先开 `TLLM_CUDA_GRAPHS=0` 验证是 graph 引入
   的问题，再检查是否有未固定缓冲（如 `LayerWorkspace` 中某 buffer 被
   重复写入顺序依赖 host）。
5. fallback：`cudaStreamBeginCapture` 返回错误或 capture 中任何 CUDA 调用
   失败时，关闭 graphs 并回退到原路径（`TLLM_WARN` 记录原因）。

**验收命令**：

```bash
TLLM_CUDA_GRAPHS=1 ./build/tiny_llm_demo model.gguf --prompt "你好" --max-tokens 32 --show-tokens
TLLM_CUDA_GRAPHS=0 ./build/tiny_llm_demo model.gguf --prompt "你好" --max-tokens 32 --show-tokens
diff <(第一个输出) <(第二个输出)   # 必须逐 token 一致
./build/tiny_llm_bench model.gguf --prompt "你好" --max-tokens 64 --warmup 3 --iters 10
TLLM_CUDA_GRAPHS=1 ./build/tiny_llm_bench ...   # 对比 TPOT before/after
```

### 任务 3.3 正确性差分测试与文档

**改动文件**：
- `tests/test_inference_engine.cu` 或 `tests/test_integration.cu`：新增
  graphs 开/关的 `generate()` 差分测试（若测试环境无真实模型则门控于
  `TLLM_GGUF_TEST_MODEL`，与现有门控测试风格一致）。
- `docs/performance/cuda-graphs.md`：记录设计、capture 范围、before/after
  数字、失败原因与 fallback 行为。

**验收**：门控测试在有模型时通过；文档包含实测 before/after 表格。

---

## 6. 阶段 4：正确性与工程完整性（P1）

### 任务 4.1 失败路径审计与测试

**背景**：面试常见追问是“损坏文件 / OOM / 超长输入怎么处理”。当前已有
部分覆盖，补成系统性的三组测试。

**改动文件**：`tests/test_model_loader.cpp`、`tests/test_kv_cache.cpp`、
`tests/test_inference_engine.cu`

测试清单：
1. **损坏 GGUF**：合法 header 但截断的 metadata / tensor 数据；magic 错误；
   版本越界。断言返回 `Result` 错误且进程不崩溃。
2. **OOM**：`KVCacheManager::create` 用超大 `max_batch_size` 或
   `max_seq_len` 触发 `cudaMalloc` 失败；断言返回错误且不泄漏（可用
   `cudaMemGetInfo` 前后对比辅助）。
3. **超长输入**：`generate()` 传入 `prompt_tokens.size() + max_new_tokens >
   max_seq_len`；断言在分配 KV 前返回验证错误，且 `getActiveSequenceCount`
   不增加。
4. **采样边界**已覆盖，不用重复。

**验收**：`./build/tiny_llm_tests` 全绿（新增用例通过）。

### 任务 4.2 GQA/MQA 第二个真实模型验证

**目标**：现有真实验证只有 Qwen2.5-0.5B（GQA 14→2）。再验证一个不同
GQA 配置或 MQA 模型，证明 attention 的 `group_size` 映射不是只对一种
配置正确。

**推荐模型（任选其一，需用户自备 GGUF）**：
- Llama-3.2-1B-Instruct（GQA 32→8，与 Qwen 差异明显）
- 任意 MQA 模型（`num_kv_heads == 1`）

**改动文件**：
- `tests/test_gguf_real.cpp`（或复用现有 GGUFRealModelTest 的门控测试结构）
- README 状态表补充验证记录

**验收**：设 `TLLM_GGUF_TEST_MODEL` 指向新模型，门控测试通过；README 更新。

### 任务 4.3 FFI 执行路径一致性收口（低风险）

**背景**：`src/ffi.cpp` 的 prefill/decode 编排与 `InferenceEngine` 重复，
长期维护会出现两边行为漂移。本次只做**低风险**收口：把
`sample_from_hidden` 中重复的 `rmsnorm + lm_head + greedy` 改为调用共享
helper。

**改动文件**：
- 新增 `src/execution_common.cpp` + `include/tiny_llm/execution_common.h`
  （或直接在 `inference_engine.h` 上增加 free function 声明，二选一，
  commit message 说明选择原因）。
- helper 签名建议：

```cpp
namespace tiny_llm {
// final norm -> lm_head（优先 FP16）-> logits 写入；不负责采样。
void computeLogitsFromHidden(const half *hidden, const ModelWeights &weights,
                             const ModelConfig &config, half *logits,
                             cudaStream_t stream);
}
```

- `InferenceEngine::computeLogits` 和 `ffi.cpp::sample_from_hidden` 都调用它。
- 不改变任何数值行为（等价重构）。

**验收**：`./build/tiny_llm_tests --gtest_filter='*FFI*:*Integration*'` 全绿。

### 任务 4.4 与 paged-serving 对接 ✅（2026-08-18，Batch D）

已完成：C ABI 升级到 ABI v2（`TinyLlmConfig` 9 int + `tinyllm_step` 增加
`num_blocks`）；实现分页 KV（策略 1：block_tables + scatter/gather 池）与
连续 KV（策略 2）双路径；策略 1/2 真模型差分逐 token 一致；paged-serving 默认
走策略 1，`PAGED_INFER_TINY_LLM_STRATEGY=2` 可回退。3 并发端到端与 llama.cpp
greedy 逐 token 对齐。Rust 侧启用 `tiny-llm` feature + build.rs 链接
`libtiny_llm.a` 已就绪。

---

## 7. 完成定义 checklist（面试自查）

达到以下所有项即认为**本项目开发结束**：

- [ ] `DEVELOPMENT_PLAN.md` 中阶段 1–3 任务全部完成并合并。
- [x] `tiny_llm_bench` schema v2 能从同一次请求输出 TTFT / TPOT / tok/s / 常驻显存差值；
      单 token 的 TPOT/tok/s 为 `null`，stdout 是单个合法 JSON 对象。
- [x] 在 clean commit 上按 schema v2 重跑正式 A/B；报告归档 5 组配对、10 个进程的
      原始 JSONL、机器可读聚合、模型哈希与限制。若声称峰值显存，仍须附外部采样器或
      profiler 的工具、版本与采样频率。
- [ ] README 出现一张与 llama.cpp 的对比表（同模型、同 prompt、同 greedy、
      同硬件），并注明“W8A16 重量化 vs Q4_K_M 直接计算”的公平性声明。
- [ ] 至少一份可导出的 nsys/ncu 报告归档在 `docs/performance/results/`，
      能解释 decode 瓶颈前三名；本机当前受 `ERR_NVGPUCTRPERM` 与 nsys importer 缺失阻塞，
      仓库内 kernel 微基准仅作为替代证据。
- [x] CUDA Graphs 有 schema v2 clean-commit 五组配对 A/B，默认开启，
      `--graphs` / `--no-graphs` 与 `TLLM_CUDA_GRAPHS=1/0` 可复现；TPOT 5/5 组下降。
- [x] `TLLM_CUDA_GRAPHS=1/0` 输出逐 token 一致。
- [ ] 失败路径测试：损坏 GGUF / OOM / 超长输入 / KV 溢出。
- [ ] 两个不同 GQA 配置（或一个 GQA + 一个 MQA）的真实模型验证记录。
- [x] 193 个测试全绿（无模型时 10 个门控测试按设计跳过）；真实模型 Graph 与 FFI
      门控测试另行执行并记录。
- [ ] 能回答三个面试问题：
  1. 为什么不用 llama.cpp / vLLM？——学习目的、可控性、从它们借鉴的设计。
  2. 每个性能数字的基线、硬件、复现命令是什么？
  3. 你的 W8A16 与 GGUF Q4_K_M 的精度/性能 tradeoff 是什么？

---

## 8. 风险与禁区（再次强调）

- **不要**“顺手重构” `tokenizer`、`gguf_parser`、`quantization` 反量化部分：
  它们已有权威参考差分验证，改动收益为负。
- **不要**改变 KV Cache 内存布局或 C ABI 字段顺序：`paged-serving` 的 Rust
  侧有布局守卫测试。
- **不要**在任务 3 之前修改 `attention_decode` 的语义（除 3.1 明确要求外）。
- **不要**把 benchmark 数字写进文档却没有记录硬件/commit/命令。
- **不要**提交 `.bak` / `reports/` 下的大文件（nsys 报告放外部或 git-lfs，
  仓库内只放 markdown 摘要）。

---

## 9. 任务卡速查

| 编号 | 任务 | 优先级 | 依赖 | 预计改动 |
|---|---|---|---|---|
| 1.1 | 实现并接线 `fp16_matmul` | P0 | 无 | 3 文件 + 1 测试 |
| 1.2 | attention prefill 长序列测试 | P0 | 无 | 1 测试文件 |
| 2.1 | benchmark 驱动 | P0 | 1.1 | 新增 2 文件 |
| 2.2 | llama.cpp 对比方法论 | P0 | 2.1 + 模型 | 文档 |
| 2.3 | profiling 指南与模板 | P0 | 2.1 + 模型 | 文档 |
| 3.1 | visible_len 间接化 | P1 | 2.1 | attention + transformer |
| 3.2 | decode graph capture/replay | P1 | 3.1 | inference_engine |
| 3.3 | 差分测试 + before/after 文档 | P1 | 3.2 | 测试 + 文档 |
| 4.1 | 失败路径审计 | P1 | 无 | 3 个测试文件 |
| 4.2 | 第二模型 GQA/MQA 验证 | P1 | 模型 | 测试 + README |
| 4.3 | FFI 一致性收口 | P1 | 1.1 | 新 helper + 2 调用点 |
| 4.4 | paged-serving 对接 | P2 | 外部仓库 | CMake install + Rust |
