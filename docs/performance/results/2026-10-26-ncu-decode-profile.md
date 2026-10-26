# 2026-10-26 ncu decode kernel 剖析

## 1. 元信息

| 项 | 值 |
|----|-----|
| 日期 | 2026-10-26 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU，6144 MiB，驱动 610.88，sm_86 |
| CUDA Toolkit | 12.0.140 |
| tiny-llm commit | `cdff4e67d384049ba528d14ea139f8e82268c43c`（文档改动不影响被测二进制） |
| 模型 | Qwen2.5-0.5B-Instruct Q4_K_M，sha256 `74a4da8c…93d7a9db` |
| 工具 | Nsight Compute 2022.4.1（`ncu --section SpeedOfLight`） |
| CUDA Graphs | 关闭（`--no-graphs`），便于逐 kernel 归因 |
| 汇总数据 | [`data/2026-10-26-ncu-decode-kernels.csv`](data/2026-10-26-ncu-decode-kernels.csv) |

## 2. 复现命令

```bash
ncu --target-processes all --section SpeedOfLight \
  --launch-skip 1500 --launch-count 200 --csv \
  --log-file reports/ncu-decode-sol.csv \
  ./build/tiny_llm_bench /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "你好" --max-tokens 32 --warmup 1 --iters 1 --no-graphs
```

`--launch-skip 1500` 跳过模型加载与 prefill 的 kernel；采样窗口覆盖约 11 个
layer-step 与一个 LM head 边界。原始 `--csv` 报告留在仓库外，仓库只保存汇总表。

## 3. 结果

### 3.1 逐 kernel

| kernel | 样本 | 平均耗时 | SM 吞吐 | Memory 吞吐 | DRAM 吞吐 |
|---|---:|---:|---:|---:|---:|
| w8a16_matmul_m1_transposed | 77 | 34.33 µs | 48.1% | 35.7% | 16.2% |
| fp16_matmul_transposed_rows（LM head） | 1 | 851.71 µs | 82.8% | 96.6% | 96.6% |
| rmsnorm | 21 | 8.75 µs | 0.2% | 1.7% | 0.3% |
| attention_decode | 11 | 11.35 µs | 2.2% | 2.6% | 0.5% |
| add_bias | 33 | 4.19 µs | 0.2% | 1.3% | 0.4% |
| add_inplace | 22 | 4.10 µs | 0.3% | 1.5% | 0.9% |
| apply_rope_inplace | 11 | 5.12 µs | 0.4% | 2.4% | 1.2% |
| append_kv_at | 11 | 4.21 µs | 0.1% | 0.8% | 0.3% |
| silu_mul_inplace | 11 | 4.21 µs | 1.8% | 2.7% | 2.2% |
| rmsnorm_inplace | 1 | 8.00 µs | 0.2% | 0.6% | 0.2% |
| gather_embeddings | 1 | 4.64 µs | 0.5% | 0.9% | 0.1% |

### 3.2 每 token 估算（24 层）

按上表均值与每层调用次数（matmul 7、rmsnorm 2、add_bias 3、attention 1、
add_inplace 2、rope 1、append 1、silu 1）估算：

| 类别 | 每 token 耗时 | 占估算 kernel 时间 |
|---|---:|---:|
| w8a16 matmul（24×7） | 5.77 ms | 71.0% |
| LM head（1） | 0.85 ms | 10.5% |
| 其余逐层 kernel | 1.51 ms | 18.5% |
| 合计 kernel 时间 | ≈8.13 ms | 100% |

同一配置下 bench 报 TPOT 10.44 ms（无 CUDA Graphs）。差值约 2.3 ms 属于
launch 间隔、采样与未纳入窗口的 kernel；该差值不是精确测量，只用于说明
kernel 时间占 TPOT 的大部分。

## 4. 瓶颈结论

1. **M=1 量化 matmul 是主瓶颈**：占估算 kernel 时间 71%，SM 吞吐 48.1%、
   DRAM 仅 16.2%，说明不是带宽受限，而是 M=1 GEMV 形状下的并行度/指令效率
   限制；这与 C1 转置权重快路径的定位一致。
2. **LM head 是明确的带宽墙**：851.7 µs、DRAM 96.6%；每次 decode 读约
   272 MB FP16 权重（151936×896×2 B），按 3060 约 336 GB/s 峰值计算的下限
   约 810 µs，实测与理论一致，说明该路径已接近带宽上限。
3. **attention 与逐层 elementwise 不是瓶颈**：attention 仅约 0.27 ms/token，
   各 elementwise kernel 均低于 9 µs 且利用率低，优化优先级应排在 matmul
   和 LM head 之后。
4. **CUDA Graphs 的价值可由 launch 数解释**：每 token 约 400 次 kernel
   launch（24 层 × 约 17 次），与 2026-08-23 记录的 TPOT 8.322→5.225 ms
   下降方向一致；本报告不重复该项 A/B 数字。

## 5. 限制

- **nsys 仍不可用**：`nsys profile` 生成成功但 importer 二进制缺失，无法导出
  `.nsys-rep` 或 `nsys stats`；本报告仅使用 ncu，缺少端到端时间线与
  memcpy/launch gap 归因。
- ncu 会锁定并扰动时钟，逐 kernel 绝对耗时与正常 bench 不完全一致；本报告
  只用它做类别占比与利用率，正式 TPOT 仍引用 bench 输出。
- 采样窗口只有约 11 个 layer-step，注释中的 LM head 样本来自窗口边界；
  per-token 表是按均值外推，不是逐 token 实测总和。
- 单卡、单架构（sm_86）、单模型；不证明其他 GPU、batch>1 或 paged KV 路径。
- CUDA Graphs 关闭；开启后的 kernel 组合与 launch 开销不同，不能直接套用本表。
