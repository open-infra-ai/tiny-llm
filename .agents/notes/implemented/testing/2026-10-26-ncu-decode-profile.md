# Agent Note: decode profiler 证据改用 ncu 采样窗口

Status: implemented

## Problem

ROADMAP 记录 ncu/nsys 在本机不可用（`ERR_NVGPUCTRPERM` 与 nsys importer 缺失），
kernel 级结论只能依赖仓库内微基准。2026-10-26 复测发现 ncu 已能采集计数器，
但 nsys 仍只生成 `.qdstrm`、无法导入。缺一份能入仓、可复现的 decode kernel
归因证据。

## Decision

用 ncu 单工具产出证据包：

- 关闭 CUDA Graphs（`--no-graphs`），避免 replay 掩盖逐 kernel 开销；
- 用 `--launch-skip 1500 --launch-count 200 --section SpeedOfLight` 跳过模型
  加载与 prefill，采样约 11 个 layer-step；
- 仓库只提交聚合 CSV 与 markdown（均值、SM/Memory/DRAM 利用率、限制），
  原始 `--csv` 报告留在仓库外；
- per-token 表按每层调用次数外推，并明确标注不是逐 token 实测总和。

nsys 不可用作为已知限制写入报告，不伪造时间线或 launch gap 归因。

## Alternatives considered

- **等 nsys importer 修好再做**：能获得端到端时间线，但当前无法产出；ncu
  的 kernel 占比已足以支撑“matmul 主导、LM head 带宽墙”的结论。
- **把原始 ncu CSV 入仓**：约 3,200 行、含 PID 与绝对路径，且是工具内部格式；
  汇总表足够复核结论，原始文件可按命令重放。
- **完整采集所有 decode kernel**：数据最全，但 replay 开销大且对结论边际收益
  低；200 个样本已覆盖所有 kernel 类别。
- **开启 CUDA Graphs 后剖析**：更接近默认路径，但 graph replay 让逐 kernel
  归因和 launch 计数失真；图的收益已有 2026-08-23 的 A/B 证据。

## Verification

- 采样窗口内 200 个 kernel launch 覆盖 11 类，无 load 期 transpose；
- w8a16 matmul 77 个样本、均值 34.33 µs、SM 48.1%、DRAM 16.2%；
- LM head 单样本 851.71 µs、DRAM 96.6%，与 272 MB / 约 336 GB/s 的理论下限
  （约 810 µs）一致；
- 报告与汇总 CSV 已入仓，命令可原样重跑；bench 同配置 TPOT 10.44 ms。

## Consequences

仓库首次有 ncu kernel 归因证据；代价是绝对耗时不等于正常 bench、缺端到端
时间线，且结论只适用于单卡、batch=1、关闭 CUDA Graphs 的 Qwen2.5-0.5B 配置。
nsys importer 修复后应补一份带 launch gap 的报告，而不是改写本报告。
