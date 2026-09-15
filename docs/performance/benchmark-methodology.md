# llama.cpp 对比基准方法论

> 本页面定义 tiny-llm 与 llama.cpp 在同一硬件/模型上对比的**可复现**方法。
> 原则：所有数字必须能用本文档中的命令原样复现；任何数字必须同时记录
> 硬件、commit、编译选项、命令（与 DEVELOPMENT_PLAN.md 风险区要求一致）。

## 1. 环境记录模板

每次跑对比前，把以下信息填进结果表（`docs/performance/results/` 下归档）：

### 硬件

| 项 | 值（示例） |
|----|-----------|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| 显存 | 6144 MiB |
| 驱动 | 591.44 |
| CUDA Toolkit | 12.0（`nvcc --version` 确认） |
| CPU / 内存 | （可选记录） |

### 软件

| 项 | 值（示例） |
|----|-----------|
| tiny-llm commit | `git rev-parse HEAD` |
| tiny-llm 编译 | Release + `-DBUILD_TESTS=ON`（默认 nvcc 目标架构 `native`） |
| llama.cpp commit | `git -C /path/to/llama.cpp rev-parse HEAD` |
| llama.cpp 编译 | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`（默认 GPU 后端） |
| 对比日期 | YYYY-MM-DD |

## 2. 模型与量化差异（必须声明）

- 模型：`Qwen2.5-0.5B-Instruct`，GGUF `Q4_K_M`。
- **llama.cpp 直接在 Q4_K_M 上计算**（GGML 内核原生支持 K 系列量化）。
- **tiny-llm 的路径是：反量化 → 转置 → 重新量化为 W8A16**（INT8 权重 +
  FP16 scale，group_size=128），加载后以 W8A16 权重推理。

因此两者**并非同一种量化格式**，延迟对比反映的是“原生 Q4_K_M 计算” vs
“W8A16 重量化后计算”两个完整路径的差距，不能解读为纯 GEMM 实现差距。

## 3. 对比边界（先区分两类实验）

1. **合成 decode 吞吐**：`llama-bench -p/-n` 使用指定数量的合成 token，不接收
   文本 prompt。它适合对比 `tg64` 等吞吐，不是同 prompt 端到端测试。
2. **同 prompt 行为/墙钟**：使用 `llama-cli -p "你好" --temp 0` 与 tiny-llm
   `do_sample=false`。`-t 1` 在 llama.cpp 中表示 CPU 线程数，**不控制采样**；llama.cpp
   当前默认 temperature 为 0.8，不能省略 `--temp 0` 后声称 greedy。
3. llama.cpp 使用 `-ngl 99`（全层 GPU），确保
   不会因 CPU 卸载把数字拖低。
4. 预热与迭代次数保持一致：
   - tiny-llm：`--warmup 3 --iters 10`；
   - llama.cpp：`llama-bench` 的 `-r`（repeat）与 `-n` 对齐，或用
     `llama-cli -n` 连续多次取稳定值。
5. 同一块 GPU、同一时段交错运行 A/B，记录温度、功耗与是否锁频。
6. tiny-llm 内置字段是“加载前 vs 运行后的常驻显存差值”，不是峰值。只有双方都用
   同一个外部采样器、同一采样频率时，才比较峰值显存。

## 4. 可复制命令

### tiny-llm

```bash
# 在 tiny-llm 仓库根目录
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

./build/tiny_llm_bench /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --prompt "你好" --max-tokens 64 --warmup 3 --iters 10 --json --graphs
# 配对运行把 --graphs 改为 --no-graphs；JSON 会记录实际 enabled/captured 状态。
# 正式 A/B 至少做 5 个独立进程对，并交错 on/off 先后顺序。
```

### llama.cpp（合成 decode 吞吐）

```bash
# 在 llama.cpp 仓库 build 目录
./build/bin/llama-bench \
    -m /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -ngl 99 \
    -p 1 \
    -n 64 \
    -t 1 \
    -r 3
```

`llama-bench` 输出 prompt processing (`pp*`) 与 token generation (`tg*`)；`tg64`
可换算 decode TPOT。`pp1` 只是一个合成 prompt token 的处理时间，不包含用户可观察的
排队、首 token 采样与输出，不能标成 TTFT。`-t 1` 只固定 CPU 线程数。

### llama.cpp（llama-cli，端到端兜底）

```bash
./build/bin/llama-cli \
    -m /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -ngl 99 -t 1 -n 64 \
    -p "你好" --temp 0 --seed 1 --no-display-prompt
```

> 注意：llama-cli 的输出带装饰信息，建议用 `llama-bench` 取数字，
> 用 `llama-cli` 做同 prompt greedy 行为抽查。由于 Q4_K_M 与 W8A16 量化不同，
> 输出可能在 argmax 边界分叉；只能按实测报告完整一致、公共前缀或 EOS，不能预设逐 token 一致。
> 参数语义以 [llama.cpp CLI 官方说明](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md)
> 为准。

## 5. 结果表模板

归档位置：`docs/performance/results/<date>-<gpu>.md`（模板见
`docs/performance/results/TEMPLATE.md`）。**实测快照（2026-08-18，RTX 3060
Laptop，详见 [2026-08-18-rtx3060](results/2026-08-18-rtx3060.md) 与
[2026-08-18-decode-optimization](results/2026-08-18-decode-optimization.md)）**：

| 指标 | tiny-llm | llama.cpp | 比值 (tiny/llama) |
|------|----------|-----------|-------------------|
| TTFT (ms) | 10.6（1-token prompt，含首次 logits） | 未按同口径测量；`pp1=4.9ms` 不是 TTFT | 不可比 |
| TPOT (ms/token) | 6.1 | 3.7（`tg64`: 272 t/s） | 1.65 |
| decode tok/s | 164.3 | 272.2 | 0.60 |
| 常驻显存差值 (MB) | 3368（加载前 vs 运行后，含 M==1 转置权重副本） | 未测（同口径） | — |

每张表下方必须附：第 1、2 节的环境快照 + 实际执行的完整命令 + 原始日志路径。

> C1 前（转置快路径落地前）tiny-llm TPOT ≈ 24.3 ms / 41 tok/s（比值 ~6.6）；
> C1/C2 后 TPOT ≈ 6.1 ms / 164 tok/s（比值 ~1.65）。kernel 级证据见
> [2026-08-18-decode-optimization](results/2026-08-18-decode-optimization.md) 第 4.2 节。

## 6. 预期差距来源分析（先写假设，实测后回填）

对 0.5B 模型 + decode 场景，预期差距主要来自（按影响排序的假设）：

1. **GEMM 实现差距**：llama.cpp 的 Q4_K_M 内核针对 K 量化格式做了 SIMD
   向量化与分块；tiny-llm 的 W8A16 m1 kernel 是每 warp 一列、32 lane 归约
   K 的简单实现（C1 起对 M==1 使用 [N,K] 转置布局 + coalesced 快路径，
   lm_head 由 ~10ms 降到 ~0.98ms）。
2. **attention 实现差距**：tiny-llm 的 decode attention 未做 KV 缓存
   L2/共享内存复用与页式布局优化。
3. **运行时/launch 开销**：24 层 × 每层多个 kernel；CUDA Graphs 已默认开启
   消解 launch 串行（见 `cuda-graphs.md`）。
4. **continuous batching 缺失**：tiny-llm 单序列；llama.cpp 即使单序列
   也有更紧凑的调度。

> 实测后：把每一行差距归因到具体 kernel（用仓库内 `tiny_llm_kernel_bench`
> 数据，见第 8 节），而不是笼统写“实现差距”。

## 7. 不在对比范围内的事项

- 多请求并发 / batch > 1 吞吐（tiny-llm 无 continuous batching）。
- 采样配置差异（temperature/top-p）——对比固定 greedy。
- 非本机、非同一时段的数据。

## 8. 本机 profiler 限制与替代方案

2026-08-23 在本机 WSL2 / 驱动 610.88 复核：`nsys`/`ncu` 命令存在，但完整分析仍受限：

- **`ncu`**：采样返回 `ERR_NVGPUCTRPERM`，没有 kernel 指标可收集；
- **`nsys profile`**：能生成 `.qdstrm`，但本机安装缺 importer，不能转换为报告并执行 stats。

因此 kernel 级瓶颈分析改用仓库内微基准 **`tiny_llm_kernel_bench`**：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j$(nproc)
./build/tiny_llm_kernel_bench
# 输出 CSV：<name>,<shape>,<ms>
# 每项先 warmup 20 次，再测 200 次（lm_head 100 次），
# 循环前后各一次 cudaDeviceSynchronize，std::chrono::steady_clock 均值。

# 三路 decode attention 对比（TLLM-P0-004 PR-5）：
./build/tiny_llm_kernel_bench --dpa-bench --warmup 20 --reps 1000 --batch 100 \
  --repeats 3 --clock-warmup 4 --out /tmp/dpa.raw.jsonl
# --dpa-bench 输出 JSONL（provenance / equivalence / sample / path_stats / shape_summary）。
# 每个样本 = batch 次背靠背调用的均值；--only-visible/--only-block-size 用于隔离单形状
# （ncu 剖析时需要）。构建必须用 -DCMAKE_CUDA_ARCHITECTURES=native，否则会编到
# sm_75 而设备是 sm_120；--clock-warmup 用于规避空闲时 SM 停在低频。
# 结果归档见 results/2026-09-14-rtx5070ti-dpa.md。

# split-KV 扫描（TLLM-ATTN-SPLITKV PR-D，schema v2）：
./build/tiny_llm_kernel_bench --dpa-bench --num-splits 1,2,4,8,16 \
  --warmup 20 --reps 1000 --batch 100 --repeats 3 --clock-warmup 4 \
  --out /tmp/splitkv.raw.jsonl
# --num-splits 逗号分隔，取值 1..32；num_splits=1 走 split-KV 入口且逐位
# 等价单遍（anchor），>1 按 2e-3 容差比对。v1 字段保留兼容。
# 结果归档见 results/2026-09-15-rtx5070ti-splitkv.md。
```

测量对象为 decode 路径真实 shape（Qwen2.5-0.5B）：W8A16 GEMM（M=1, K=896,
N∈{128,896,4864} 与 down M=1,K=4864,N=896）、FP16 lm_head（M=1,K=896,
N=151936）、attention_decode（S∈{8,32,64,128}）、rmsnorm、RoPE、add、
silu_mul。C1 起该工具测量的是 M==1 转置快路径（与推理引擎 decode 一致的
公开接口）。

### profiler 可用性（2026-09-14 复核，推翻上表结论的一部分）

| 工具 | 2026-08-23（驱动 610.88 / ncu 2022.4.1） | 2026-09-14（驱动 615.65.06 / ncu 2026.2.1.0） |
|------|------------------------------------------|-----------------------------------------------|
| `ncu` | `ERR_NVGPUCTRPERM`，无指标 | **可用**，可收集 `gpu__time_duration.sum`、`dram__bytes.sum`、`lts__t_bytes.sum`、`l1tex__t_bytes.sum`、`sm__throughput.*`、`smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` |
| `nsys` | 可生成 `.qdstrm`，缺 importer | 未复核 |

注意：`dram__bytes_read.sum` 在 2026-09-14 的驱动上返回 `n/a`，改用 `dram__bytes.sum`。
ncu 会 replay kernel，绝对时长高于墙钟，**只可比值，不可与 `--dpa-bench` 的墙钟数混用**。

