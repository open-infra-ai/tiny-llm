# Tiny-LLM 推理引擎

> 📚 Portfolio map: https://github.com/open-infra-ai/open-infra-ai

> 面向聚焦型 Transformer 工作负载的 CUDA 原生 C++ 推理引擎。

> **项目状态**：**active**（学习/演进中）；权威状态注册表见
> [open-infra-ai/open-infra-ai](https://github.com/open-infra-ai/open-infra-ai)。

[![CI](https://github.com/open-infra-ai/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/open-infra-ai/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/open-infra-ai/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://open-infra-ai.github.io/tiny-llm/)
[![Release](https://img.shields.io/github/v/release/open-infra-ai/tiny-llm?include_prereleases&label=version)](https://github.com/open-infra-ai/tiny-llm/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

[文档](https://open-infra-ai.github.io/tiny-llm/) • [架构说明](https://open-infra-ai.github.io/tiny-llm/architecture/) • [API](https://open-infra-ai.github.io/tiny-llm/api/) • [路线图](ROADMAP.md) • [更新日志](CHANGELOG.md)

---

## 项目概述

Tiny-LLM 将仓库表面保持得尽量小：CUDA/C++17 内核、W8A16 量化、显式 KV Cache 管理，以及一条更容易审计和维护的精简运行时路径。

在 GPU 学习主线中，本仓库只负责模型权重到 token 生成的运行时主线；CUDA/Triton kernel 学习和 Serving 调度保持在各自主仓。整体顺序见 [`LEARNING_PATH.md`](https://github.com/open-infra-ai/open-infra-ai/blob/master/LEARNING_PATH.md)（meta 仓）。

## 项目边界（IN / OUT）

**IN（本仓库负责）**：
- GGUF 加载与反量化（F16/F32/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K）
- W8A16 量化推理、KV Cache、tokenizer、采样、端到端生成
- 分页 KV（策略 1：block_tables + scatter/gather；经 C ABI 供 paged-serving 使用）
- 性能基准（TTFT / TPOT / tok/s / 常驻显存差值）与 CUDA Graphs 加速 decode

**OUT（明确不做，见对应仓库）**：
- 调度 / 批处理 / continuous batching → [paged-serving](https://github.com/open-infra-ai/paged-serving)
- FlashAttention 深挖与优化 → [cuflash](https://github.com/open-infra-ai/cuflash)
- Triton 算子 → [trifuse](https://github.com/open-infra-ai/trifuse)

## 项目状态（诚实声明）

| 能力 | 状态 |
|------|------|
| W8A16 量化 kernel 与运行时路径 | ✅ 已实现，有差分测试 |
| KV Cache 管理、采样（temperature/top-k/top-p） | ✅ 已实现 |
| GGUF 解析与反量化（F16/F32/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K） | ✅ 已实现，真实模型验证通过（见下） |
| 架构感知配置提取（qwen2/llama/...） | ✅ 已实现，真实模型验证通过 |
| tokenizer | ✅ 已实现，与 HuggingFace tokenizers 差分测试逐 id 对齐（30 例 417 token） |
| 真实模型端到端生成 | ✅ Qwen2.5-0.5B-Instruct（Q4_K_M）在 RTX 3060 上验证通过 |
| 分页 KV（策略 1） | ✅ block_tables + scatter/gather 池，与连续 KV 策略逐 token 差分一致，paged-serving 3 并发 e2e 与 llama.cpp 对齐 |
| C ABI 正常 greedy 输出 | ✅ `logprobs_k == 0` 时每序列末层 hidden 先写入 GPU batch buffer，再批量执行 final RMSNorm、LM head 与 argmax，`tinyllm_step` 一次回传整批 token；Transformer layer 仍逐序列，logprobs 保持主机路径 |
| Ragged RoPE 位置原语 | ✅ 内部 CUDA API 支持每个 token 独立 device 绝对位置，并与 CPU 参考逐元素对照；尚未接入 FFI 的逐层 batch compute |
| 端到端性能基准 | ✅ 已实现 `tiny_llm_bench`（同请求 TTFT / TPOT / tok/s / 常驻显存差值），见下方基准快照 |

当前开发重点见 [ROADMAP](ROADMAP.md)。性能相关的文档只描述方法与计划，不引用未实测的数字。

### 正式基准快照（2026-08-23，schema v2）

| 环境 | 值 |
|------|----|
| 硬件 | RTX 3060 Laptop 6GB，驱动 610.88，CUDA runtime 12.0 |
| 模型 | Qwen2.5-0.5B-Instruct GGUF Q4_K_M（加载后重量化为 W8A16） |
| 被测 commit | `565da79`，每个进程启动前工作区 clean |
| 协议 | 5 组独立进程对；每进程 3 warmup + 10 timed iterations；on/off 顺序交错 |

| 指标 | Graph off | Graph on | 变化 |
|------|-----------|----------|------|
| TTFT p50 的跨进程中位数 | 9.072 ms | 8.822 ms | -2.8%（配对中位数 +3.4%，视为噪声） |
| TPOT mean 的跨进程中位数 | 8.322 ms | 5.225 ms | **-37.2%** |
| decode 吞吐的跨进程中位数 | 120.168 tok/s | 191.384 tok/s | **+59.3%** |
| 常驻显存差值 | 3364 MB | 3368 MB | +4 MB（非峰值） |

> CUDA Graphs decode 默认开启；`TLLM_CUDA_GRAPHS=0` 显式关闭。
> 正式报告包含 10 个进程的原始 JSONL、机器可读聚合、模型哈希、完整命令和限制：
> [`2026-08-23-cuda-graphs-ab.md`](docs/performance/results/2026-08-23-cuda-graphs-ab.md)。
> 2026-08-18 schema v1 与 kernel microbench 只保留为优化沿革，不和 schema v2 混算；
> 见 [decode optimization](docs/performance/results/2026-08-18-decode-optimization.md)。

## 已实现能力

- **W8A16 推理路径**：INT8 权重 + FP16 激活
- **显式 KV Cache 管理**：面向自回归解码
- **CUDA 原生 Kernel**：共享内存与 warp 级优化模式
- **基于 `Result<T>` 的可失败 API**：宿主侧错误传播更直接
- **GoogleTest 测试覆盖**：kernel 数值差分、KV Cache 不变量、模型加载等核心路径

## 模型加载

- `InferenceEngine::load()` 支持 **GGUF** 和**二进制运行时格式**两种加载路径。
- GGUF 路径：`GGUFParser` 解析文件、提取模型配置，读取 tensor 数据并反量化（支持 F16/F32/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K），重量化为 W8A16 后上传 GPU。
- 二进制路径：`loadBin()` 直接读取预量化的 W8A16 权重，主要用于测试。

> GGUF 加载路径已用真实模型 **Qwen2.5-0.5B-Instruct（Q4_K_M，GGUF v3，291 tensors）** 验证：
> 配置提取（hidden_dim=896 / layers=24 / GQA 14→2 / vocab=151936）与 Q5_0/Q4_K/Q6_K
> 首块反量化均与 Python `gguf` 参考实现一致（见 `tests/test_quantization.cpp`，
> 设置环境变量 `TLLM_GGUF_TEST_MODEL` 可复现）。
> CLI 提供 CPU-only 的 `tiny_llm_demo --inspect model.gguf` 查看配置与 tensor 摘要；
> tokenizer 已实现并通过差分验证；端到端文本生成已在 RTX 3060 上验证（见上方基准快照与 [ROADMAP](ROADMAP.md)）。

> **GQA/MQA 映射验证**（任务 4.2）：
> - Qwen2.5-0.5B **GQA 14→2**（真实模型端到端）。
> - SmolLM2-360M-Instruct **GQA 15→5**（Q8_0，真实模型端到端；2026-10-26）。
> - Kernel 级已验证：Llama-3.2-1B **GQA 32→8**（group_size=4）与 **MQA 16→1**
>   （group_size=16）的 decode 与 CPU 参考逐元素对齐（`tests/test_kernels.cu`）。
> - 两个真实模型门控测试分别由 `TLLM_GGUF_TEST_MODEL` 与
>   `TLLM_GGUF_TEST_MODEL_2` 启用；`scripts/run-gpu-gate.sh` 要求两者都设置，
>   并把任何 skip 视为失败。当前证据见
>   [`docs/performance/results/2026-10-26-gpu-gate.md`](docs/performance/results/2026-10-26-gpu-gate.md)。

## 从源码构建

Tiny-LLM 需要可用的 CUDA 工具链（`nvcc` 在 `PATH` 中，或已正确配置 CUDA 安装）。

| 组件 | 最低要求 |
|---|---|
| NVIDIA GPU | 计算能力 7.0+ |
| CUDA Toolkit | 11.0+ |
| CMake | 3.18+ |
| C++ 编译器 | GCC 9+ 或 Clang 10+ |

```bash
git clone https://github.com/open-infra-ai/tiny-llm.git
cd tiny-llm

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure --timeout 300
```

### 真实 GPU 非 skip 门禁

无模型的 `ctest` 会跳过 GGUF 门控用例；发布前用以下脚本跑完整门禁：

```bash
TLLM_GGUF_TEST_MODEL=/path/to/qwen2.5-0.5b.gguf \
TLLM_GGUF_TEST_MODEL_2=/path/to/second-model.gguf \
  ./scripts/run-gpu-gate.sh
```

脚本要求两个模型变量都存在，构建并运行全部测试，任一测试失败或被跳过都会以
非零状态退出，并把环境、commit、模型哈希与完整输出写入
`docs/performance/results/data/<date>-gpu-gate.log`。

## 最小使用示例

```cpp
#include <iostream>
#include <tiny_llm/inference_engine.h>

int main() {
    using namespace tiny_llm;

    ModelConfig config;
    config.vocab_size = 32000;
    config.hidden_dim = 4096;
    config.num_layers = 32;

    auto engine_result = InferenceEngine::load("model.bin", config);
    if (engine_result.isErr()) {
        std::cerr << engine_result.error() << '\n';
        return 1;
    }

    GenerationConfig gen;
    gen.max_new_tokens = 64;
    gen.temperature = 0.7f;
    gen.top_p = 0.9f;
    gen.do_sample = true;

    auto engine = std::move(engine_result.value());
    auto output = engine->generate({1, 15043, 29892}, gen);
    if (output.isErr()) {
        std::cerr << output.error() << '\n';
        return 1;
    }

    return 0;
}
```

## 仓库结构

```text
include/tiny_llm/         公共头文件
src/                      主机端 C++ 实现
kernels/                  CUDA kernels
tests/                    单元测试与属性测试
docs/                     VitePress 文档站点
.github/workflows/        CI、Pages、release 自动化
CHANGELOG.md              唯一的已跟踪发布历史
```

## 参与贡献

欢迎提交 Issue 与 Pull Request。请保持改动聚焦，让文档与真实运行时边界一致，并避免重新引入重复的流程脚手架。详见[开发者指南](https://open-infra-ai.github.io/tiny-llm/contributing/)。

## 许可证

Tiny-LLM 采用 [MIT License](LICENSE)。
