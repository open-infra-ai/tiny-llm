# 性能

> **诚实声明**：所有数字均可通过仓库内脚本（`tiny_llm_bench`）复现，
> 并记录硬件 / commit / 命令（见 [对比方法论](./benchmark-methodology)）。

## 当前已验证的内容

| 项目 | 验证方式 | 状态 |
|------|----------|------|
| W8A16 矩阵乘数值正确性 | GTest + 与 FP16 参考实现差分对比 | ✅ |
| Attention / RMSNorm / RoPE kernel | GTest 单元测试 | ✅ |
| KV Cache 分配与回收 | 单元测试 + 不变量检查 | ✅ |
| GGUF 解析与反量化（F16/F32/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K） | 与 Python gguf 参考差分对比 | ✅ |
| 真实模型端到端生成 | Qwen2.5-0.5B-Instruct（Q4_K_M）在 RTX 3060 上验证 | ✅ |
| 吞吐 / 延迟 / 显存基准 | `tiny_llm_bench`（同请求 TTFT / TPOT / tok/s / 常驻显存差值） | ✅ |

## 正式基准快照（2026-08-23，schema v2）

| 环境 | 值 |
|------|----|
| 硬件 | RTX 3060 Laptop 6GB，CUDA runtime 12.0，驱动 610.88 |
| 模型 | Qwen2.5-0.5B-Instruct GGUF Q4_K_M（重量化为 W8A16） |
| commit | `565da79`（clean） |
| 协议 | 5 组独立进程对，每进程 3 warmup + 10 timed iterations |

| 指标 | Graph off | Graph on | 变化 |
|------|-----------|----------|------|
| TTFT p50 跨进程中位数 | 9.072 ms | 8.822 ms | -2.8%（配对口径无稳定改善） |
| TPOT mean 跨进程中位数 | 8.322 ms | 5.225 ms | **-37.2%** |
| decode 吞吐跨进程中位数 | 120.168 tok/s | 191.384 tok/s | **+59.3%** |
| 常驻显存差值 | 3364 MB | 3368 MB | +4 MB（非真实峰值） |

复现命令、10 个进程的原始 JSONL、机器可读聚合、模型哈希与结论边界见
[2026-08-23-cuda-graphs-ab](results/2026-08-23-cuda-graphs-ab.md)。
2026-08-18 schema v1 和 kernel microbench 只保留为优化沿革，不与 schema v2 混算；
见 [2026-08-18-decode-optimization](results/2026-08-18-decode-optimization.md)。

## kernel 级基准：direct paged decode 三路对比（2026-09-14，schema `tllm-dpa-kernel-bench-v1`）

`tiny_llm_kernel_bench --dpa-bench` 在**同一份逻辑 K/V** 上比较 `legacy`
（gather ×2 + 连续 attention）、`contiguous`（连续 attention，上界参考）与 `direct`
（`attention_decode_paged`），并单独报告两条 gather 的耗时。正确性逐位相等（32/32）
先于计时。

| 项 | 值 |
|----|----|
| 硬件 | RTX 5070 Ti 16GB（sm_120），CUDA 13.3.73 / 驱动 615.65.06 |
| commit | `b6f6138`（clean）；被测 kernel 为 `ce93564`（PR-3 / #9） |
| 采样 | 每样本 100 次调用均值 × 1000 次/repeat × 3 repeat；4 s 时钟预热；32 shape 中 17 个收敛 |
| 结论 | **`direct` 在 `visible ≤ 128` 更快（主要省 2 次 launch），在 `visible ≥ 512` 慢 3%–15%；不支持把默认值改为 `auto`，默认保持 `legacy`** |

完整表格、分解、收敛限制与 ncu 证据见
[2026-09-14-rtx5070ti-dpa](results/2026-09-14-rtx5070ti-dpa.md)。

## 章节

- [基准测试](./benchmarks) - 基准测试方法与计划
- [对比方法论](./benchmark-methodology) - 与 llama.cpp 对比的可复现方法
- [分析指南](./profiling-guide) - nsys / ncu 具体操作与必记指标
- [优化](./optimization) - 已实现的优化与路线图
- [分析](./profiling) - Nsight 方法概览
