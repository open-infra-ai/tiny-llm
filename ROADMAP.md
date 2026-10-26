# Tiny-LLM 路线图

> 目标：把 Tiny-LLM 做成**可在面试中完整讲述的端到端推理引擎作品**——
> 真实模型、真实硬件、可复现数字、有对比基线。
> 原则：每完成一项，更新 README「项目状态」表，把 ❌ 改成 ✅。

## 阶段 0：诚实基线（已完成）

- [x] 清理文档中无法复现的性能数字与不存在的配置项
- [x] README 明确标注已实现/未实现边界

## 阶段 1：端到端跑通真实模型（最高优先级）

- [x] 本机 CUDA 构建环境（CUDA 12.9 toolkit + gcc 13，`cmake --build` 全绿）
- [x] 用真实 GGUF 模型验证加载路径（Qwen2.5-0.5B-Instruct Q4_K_M，469MB，291 tensors）：
      架构感知配置提取、vocab 派生、Q5_0/Q4_K/Q6_K 反量化，与 Python gguf 参考实现一致
- [x] `--inspect` CPU-only 检查模式（无需 GPU 即可验证任意 GGUF 文件）
- [x] 修复加载路径暴露的真实 bug（架构前缀硬编码、vocab_size 来源错误、
      未知量化类型静默假设 FP16），均有回归测试
- [x] 实现 tokenizer（Qwen2.5 为 gpt2 风格 BPE，GGUF 内嵌 tokens+merges，从 GGUF 读取）
      ——手写 Qwen2 预分词正则 + GPT-2 字节编码 + BPE，与 HuggingFace tokenizers
      库 30 例 417 token 逐 id 差分对齐，decode 无损往返
- [x] GPU 环境验证权重上传与端到端生成（RTX 3060 实测：
      `tiny_llm_demo model.gguf --prompt "..."`，prefill ~170ms、decode ~10 tok/s，输出合理文本）
- [x] 与 llama.cpp 同模型同输入对比输出一致性（token 级 diff）：
      前 14 个 token 与 llama-server（同 prompt、greedy）完全一致；后续分歧源于
      W8A16 与 Q4_K_M 量化方案的精度差异（argmax 边界翻转），非架构错误

**完成证据**：一条命令从 GGUF 到文本输出，且与 llama.cpp 输出对齐。

## 阶段 2：可复现的性能基准

- [x] 编写 benchmark 驱动（固定预热/迭代/统计口径，CUDA Event + 宿主计时）
      —— `tiny_llm_bench` schema v2（同请求 TTFT / TPOT / tok/s / 常驻显存差值，
      warmup/iters/统计口径固定；峰值显存须由外部采样器或 profiler 另测）
- [x] 指标：TTFT、TPOT、decode 吞吐（tok/s）、常驻显存差值
- [ ] 对比基线：llama.cpp（同硬件、同模型、可解释的量化差异）
      —— `docs/performance/results/2026-08-18-rtx3060.md` 是已审计的 schema v1
      历史快照；tiny-llm 自身 schema v2 clean A/B 已完成，llama.cpp 同口径正式复测仍待完成
- [x] kernel 级瓶颈分析 —— ncu/nsys 本机不可用（`ERR_NVGPUCTRPERM` / importer
      缺失），改用仓库内微基准 `tiny_llm_kernel_bench`（C0）作为替代证据
- [x] 将实测数字回填 docs/performance/（附硬件/软件版本/复现命令）
      —— 见 `docs/performance/results/2026-08-18-decode-optimization.md`
- [x] ncu 逐 kernel 剖析（2026-10-26）—— ncu 恢复可用，产出 decode kernel
      占比与利用率；nsys importer 仍缺失。见
      `docs/performance/results/2026-10-26-ncu-decode-profile.md`

**当前证据**：clean commit `565da79` 的 schema v2 五组配对 A/B 中，CUDA Graph
使 TPOT 跨进程中位数 8.322 → **5.225 ms/token**（-37.2%），decode 吞吐
120.168 → **191.384 tok/s**（+59.3%）；10 个进程的原始 JSONL 与限制已归档。
历史 schema v1 仅用于说明优化沿革，不与正式数字混算。

## 阶段 3：选一个推理加速主题做深（二选一）

**选项 A：CUDA Graphs（已选，已完成）**
- [x] decode 阶段 graph capture/replay，配置开关
      —— 默认开启，`TLLM_CUDA_GRAPHS=0` opt-out（见 `cuda-graphs.md`）
- [x] before/after 的 launch 开销与 TPOT 对比数字
      —— schema v2 在 clean commit 上完成五组交错配对 A/B；TPOT 5/5 组下降，
      原始 JSONL、聚合、复现命令与限制见
      `docs/performance/results/2026-08-23-cuda-graphs-ab.md`

**完成证据**：一个有数字、有 profiling 证据、能讲清取舍的优化故事。
（C1 转置权重 M==1 快路径是本批另一个核心加速点，与 CUDA Graphs 叠加。）

**选项 B：Speculative Decoding（草稿模型 n-gram 或小型同构模型）**
- [ ] 接受率统计与端到端加速比
- [ ] 与标准 decode 的正确性差分测试

**完成证据**：一个有数字、有 profiling 证据、能讲清取舍的优化故事。

## 阶段 4：工程完整度（面试加分项）

- [x] GQA 真实模型验证：Qwen2.5-0.5B **14→2** 端到端；kernel 级另验 Llama-3.2 风格
      **32→8** 与 **MQA 16→1**（`tests/test_kernels.cu`）
- [x] 第二个真实 GGUF 端到端门控（`TLLM_GGUF_TEST_MODEL_2`，SmolLM2-360M-Instruct
      Q8_0，GQA 15→5，2026-10-26；见 `docs/performance/results/2026-10-26-gpu-gate.md`）
- [ ] 长上下文 KV Cache 显存实测曲线
- [ ] 失败路径审计：加载损坏文件、显存不足、超长输入的行为与测试
- [x] 导出 C ABI（`ffi.h`/`ffi.cpp`：`tinyllm_load` / `tinyllm_step` / `tinyllm_allocate_sequence`
      / `tinyllm_free_sequence` / `tinyllm_free`），供 paged-serving 调度层接入
- [x] 与 [paged-serving](https://github.com/open-infra-ai/paged-serving) 的调度层对接：Rust `tiny-llm`
      feature + build.rs 链接 + `TinyLlmExecutor`；分页 KV 策略 1 默认启用，3 并发 e2e
      与 llama.cpp greedy 对齐（2026-08-18，`phase-2-e`）
- [x] C ABI 正常 greedy 的批量末端后处理：`logprobs_k == 0` 时每序列完成 layer forward
      后立即把末层 hidden 写入 GPU batch buffer；循环结束后批量执行 final RMSNorm、LM head
      与 argmax，并在 `tinyllm_step` 末尾一次回传整批 token。真实模型与 host/logprobs
      路径对照、分页/连续 KV 差分与 paged-serving 三并发 e2e 均已复验。
- [x] Ragged RoPE 位置前置原语：内部 CUDA API 可接收 `[num_tokens]` device 绝对位置数组，
      非连续、非单调位置与 CPU half-split RoPE 参考逐元素对照通过；它没有改变 C ABI，也尚未
      接入 FFI 的 Transformer layer forward。
- [ ] 融合 batch compute：当前 Transformer layer forward 仍逐序列执行；需建立 ragged
      batch workspace 与逐 token oracle，并逐层批量化 layer compute，不能把已完成的末端
      输出阶段批量化称为 fused batch。

> **状态**：**active**。`phase-2-e` tag 记录 2026-08 面试就绪快照；其后的开发（正确性修复与功能打磨）在 CHANGELOG 中延续记录，未勾选项保留在「不做什么」清单供恢复开发时逐项评估。

## 面试讲述要点（完成后自查）

1. 能在 10 分钟内讲清：架构、量化方案、KV Cache 设计、瓶颈与优化过程
2. 每个性能数字都能回答：基线是什么、硬件是什么、怎么复现
3. 能回答"为什么不用 llama.cpp/vLLM"：学习目的、可控性、以及从它们学到的设计
