# Changelog

All notable tracked releases of Tiny-LLM are recorded here.

## [Unreleased]

### Added

- `tests/paged_attention_oracle.h`（TLLM-P0-002）：不依赖外部 GGUF 的
  paged/contiguous synthetic correctness oracle。纯 host 参考实现，冻结 paged KV
  地址公式、块表长度 contract 与 GQA 映射，并独立计算 fp32 decode attention；
  **不调用** `paged_gather_blocks` / `paged_scatter_blocks` / `attention_decode`。
- `tests/test_paged_oracle.cpp`：上述 oracle 的 kernel 级与 layer 级差分门禁。
  kernel 级把生产 scatter + gather + attention_decode 对照独立参考；layer 级把
  `TransformerLayer::forwardPaged` 的 pool 内容按冻结公式读回，与连续 KV cache
  的可见 K/V 做逐层位级比较。

### Fixed

- `TransformerLayer::forwardPaged` 增加块表长度校验：`visible_blocks` 必须
  `>= ceil(visible_tokens / block_size)`，否则返回错误。此前过短的块表会让
  `paged_gather_blocks` 越界读 `block_table`（未定义行为）；该 contract 已由
  `src/ffi.cpp` 的 `need_blocks` 检查使用，现下移到层入口。
- `KVCacheManager::create` 显式把 `append_pos_` 清零（与 `memory_pool_` 的
  `cudaMemset` 约定一致）。此前依赖 `cudaMalloc` 返回清零内存：调用方未先
  `setAppendPos` 时，`appendKV` 会按未初始化值把 K/V 写到错误位置。

### Changed

- C ABI 正常 greedy 路径（`logprobs_k == 0`）现在在每个序列 layer forward 完成后把
  末层 hidden 写入独立 GPU batch buffer；循环结束后批量执行 final RMSNorm、LM head 与
  argmax，并由 `tinyllm_step` 一次性回传该 batch 的 token。C ABI 布局、输出顺序与
  `logprobs` 路径均未改变；请求 logprobs 时仍保留主机侧完整 logits / top-k 计算。
  Transformer layer forward 仍逐序列执行，因此这不是 fused batch，也尚无新的 serving
  性能结论。

### Added

- `kernels/sampling.cu`：多行 GPU greedy argmax，保持 CPU 顺序扫描的同分 token id 选择
  与首项 NaN 语义。
- FP16 转置 LM head 的每行 coalesced 路径，供批量末端输出阶段复用。
- RoPE 内部 CUDA API 新增 `[num_tokens]` device 绝对位置输入，支持 ragged sequence 的
  非连续、非单调位置；现有连续位置 API 与 C ABI 均不变，尚未接入逐层 batch compute。

### Tests

- TLLM-P0-002 oracle（8 项）：kernel 级与 layer 级 paged/contiguous 差分，覆盖
  block_size 1/16/32、跨块尾部、MHA/GQA/MQA、head_dim 32/64/128、绝对位置增量
  scatter、多 layer pool offset、非法块 id 与过短块表、随机 seed；oracle 已做变异
  检验（破坏 scatter 位置写入或 layer 步长会分别被对应门禁捕获），
  `compute-sanitizer --tool memcheck` 0 error。边界：被测路径仍是
  scatter → gather → continuous attention，**不是** direct PagedAttention；
  本变更不产生任何性能数字。
- CUDA kernel 直接对照 CPU greedy（跨 block scan、同分 token、首项 NaN 与双行 batch）；
  batch final RMSNorm / LM head 对照逐行单 token 路径，转置 FP16 M=4 对照 reference。
  RoPE 每 token 位置数组以非连续、非单调位置逐元素对照 CPU half-split 参考。
  真实 GGUF 门控对照 device 与 host/logprobs 路径连续 4 token、分页/连续 KV 差分；
  paged-serving `tiny-llm` feature 的真实后端、文本与三并发分页 e2e 复验通过。

## [2.0.2] - 2026-08-28

### Fixed
- 修复 `tiny_llm_bench` 指标口径：TTFT 与 TPOT 改为从同一次 `generate()` 请求取样，
  不再用独立 1-token 请求从另一条完整请求中相减；`GenerationStats` 新增
  `time_to_first_token_ms`，真实模型 Graph on/off 测试断言该字段有效。
- benchmark 不再把加载前/运行后的 `cudaMemGetInfo` 差值冒充“峰值显存”，统一改称
  常驻显存差值；删除从未赋值的 `GenerationStats::peak_memory_bytes`。
- 澄清并收紧 `tinyllm_step` 的 logprobs ABI：输出缓冲区至少为
  `num_sequences * logprobs_k * 2` 个 `float`；拒绝负 `logprobs_k`、超过词表大小
  或请求输出但传空缓冲区。新增双序列 stride 与首尾 canary 回归测试。
- Unicode 类别表生成器固定并校验 Unicode 15.0.0，生成文件标注不再错误声称 15.1；
  生成区间用 clang-format guard 保持确定性，格式 CI 明确豁免只读 tokenizer fixture。
- 按 CI 使用的 clang-format 18 统一 FFI、benchmark JSON 输出与模型权重结构体排版，
  消除本机缺少同版本格式器时未被发现的远端 Format 门禁失败。
- CUDA 安装 Action 更新至 `Jimver/cuda-toolkit@v0.2.23`，并把版本写成完整的
  `11.8.0`，避免 Node.js 24 执行环境把 `11.8` 判为无效语义版本而阻断构建。
- 补齐 `ExecutionCommon` 与超长输入回归测试的 GPU-less 门控；CUDA runtime 返回
  `cudaErrorNoDevice` 时按其他 GPU 测试的既有约定跳过，而不是误判为功能失败。
- GGUF 解析器健壮性加固（审计 llama.cpp#26366/#26978 同类问题时发现并修复）：
  - `readTensorInfoEntry`：`n_dims` 原无上限，文件可控的恶意值（如 0xFFFFFFFF）
    会使 `dimensions.resize` 尝试 ~32GB 分配，未捕获的 bad_alloc 直接 abort；
    现按 GGML_MAX_DIMS(4) 拒绝
  - `readTensorData`：`data_offset_ + tensor.offset` 无溢出检查，64 位回绕后
    seek 到错误偏移读垃圾数据（静默损坏）；现相加前拒绝
  - 两个字节级构造的回归测试（`test_gguf_parser.cpp`）；当前全量 193 测试通过

### Changed
- `tiny_llm_bench --json` 升级为 schema v2：stdout 只输出一个合法 JSON 对象，包含
  GPU、warmup/iterations、Graph 实际 enabled/captured 状态；单 token 场景的 TPOT/tok/s
  输出 `null`。`--graphs` 现显式启用，且与 `--no-graphs` 冲突时失败。
- 修正性能方法论：llama.cpp `-t` 是 CPU 线程数而非采样开关；`llama-bench pp1`
  不再标为 TTFT；合成 decode 吞吐与同 prompt `llama-cli --temp 0` 行为核验分开报告。
- 新增 2026-08-23 CUDA Graph schema v2 clean-commit 正式 A/B：5 组交错配对、
  10 个独立进程均归档原始 JSONL；TPOT 跨进程中位数 8.322→5.225 ms（-37.2%），
  decode 吞吐 120.168→191.384 tok/s（+59.3%）。同时提供机器可读聚合、模型哈希、
  常驻显存口径与 ncu/nsys 限制；TTFT 因配对波动不作改善声明。
- 更新 VitePress 文档依赖锁文件到兼容范围内的安全补丁版本，`npm audit` 从 12 项降至
  4 项；剩余项均来自 VitePress 1.6.4 的 Vite 5/esbuild 链，当前无兼容修复，未强制
  升级到 VitePress 2 alpha。
- 面向用户的 GitHub 链接统一为 `github.com/open-infra-ai/...`（tokenizer 差分夹具原文不改）
- 默认 CUDA 架构加入 sm_70（`CMAKE_CUDA_ARCHITECTURES` 非新版本路径下为 `70 75 80 86 89`）
- `model_loader` 中 `data_t` 显式 `* sizeof(int8_t)`（显式以字节为单位的语义，避免换量化元素类型时踩坑）
- 消除 `quantizeF16ToW8A16` 中冗余的 `static_cast<half>(scale)`

### Added

- gpt2 风格字节级 BPE tokenizer：从 GGUF 读取 tokens/merges/token_type，
  手写 Qwen2 预分词正则（Unicode 感知）+ GPT-2 字节编码 + BPE 合并，
  支持 CONTROL/USER_DEFINED 特殊 token 精确隔离与字节级无损 decode
- `loadTokenizerData`：从 GGUF 元数据提取 TokenizerData
- 测试：tokenizer 差分测试（对照 HuggingFace tokenizers 库，30 例 417 token
  逐 id 对齐 + decode 无损往返），门控于 TLLM_GGUF_TEST_MODEL

- Q5_0 / Q4_K / Q6_K GGUF 反量化（Q4_K_M 文件的实际量化类型）
- 架构感知的 GGUF 配置提取：按 general.architecture 前缀读取（qwen2/llama/...），
  vocab_size 从 tokenizer.ggml.tokens 数组长度派生
- `tiny_llm_demo --inspect model.gguf`：CPU-only 的 GGUF 配置/tensor 摘要
- 测试：合成块反量化单元测试（期望值来自 Python gguf 参考实现）；
  真实模型门控测试（TLLM_GGUF_TEST_MODEL）

### Fixed

- **各投影使用自身 group_size 反量化**：attention/attentionPaged/feedForward 不再复用
  `wq`/`w1` 的 group_size，K/V/输出与 gate/up/down 各按各自张量的 group_size 索引 scale
  （异构/重量化场景下原实现 scale 行号整体错位）
- **softmax 改为 O(1) 共享内存**：旧实现按 `(seq_len+32)*4B` 缓存 exp 值，
  seq_len ≈ 12K 即超 48KB 动态共享内存上限导致 launch 失败；现三遍法（max → sum → 重算 exp），
  任意 seq_len 正确
- **paged 块 id 值域防护**：`paged_scatter/gather_blocks` 增加 `max_num_blocks` 参数，
  越界块 id 跳过写入（scatter）/写 0（gather），坏块表不再造成越界访存毒化 CUDA 上下文
- **GGUF 计数上界校验**：`tensor_count`/`metadata_kv_count` 超 `1<<20` 直接报错，
  损坏/恶意文件的巨大计数不再穿透 `parse()` 抛 `length_error`/`bad_alloc`
- **UTF-8 continuation 校验**：多字节序列后续字节必须为 0x80–0xBF，否则 leader 按单字节
  回退（与 HF GPT-2 byte-level 语义一致）；残缺尾部逐字节处理；新增 `decodeUtf8Codepoints`
  诊断接口
- **Qwen2 attention bias 缺失（GPU 端到端乱码根因）**：加载并应用 attn_q/k/v.bias，
  补齐 Qwen2 系 q/k/v 投影的 bias 项；修复后输出与 llama.cpp 前 14 token 完全一致
- **共享层工作区（OOM 修复）**：中间激活缓冲改为所有层复用（LayerWorkspace），
  修复 24 层每层独立分配导致的显存爆炸（0.5B 模型在 6GB 卡无法加载）
- **attention O 投影非就地（未初始化内存/不确定输出）**：注意力输出改用独立 attn_buf，
  修复就地 matmul 输入被覆盖导致的数据竞争与不确定生成
- **lm_head 支持 FP16**：output 层不量化，保持 logits 精度（W8A16 作为后备）
- calculateSize 不再对未知量化类型按 FP16 估算（会导致静默错位读取），改为显式失败
- 移除断言旧行为（"GGUF 运行时加载不支持"）的过时测试，改为验证真实的加载错误路径

### Fixed（2026-08-21 bug 专项修复）

- **add_bias_inplace kernel 越界写**：`add_bias_kernel` 增加 `idx >= rows*cols` 边界
  检查（grid 按 ceil(total/256) 启动，尾块线程在非 256 倍数尺寸下会越界读写；
  Qwen2 系 hidden=896 的 decode/奇数 token prefill 会触发）
- **CUDA Graph H2D host 源指针固化（未定义行为）**：decodeStep 的 token_id /
  decode_len / rope_pos 与 setAppendPos 的 H2D memcpy 源指针由栈/临时变量改为
  引擎/KVCache 成员变量——graph capture 会固化 host 指针并在重放时读取当前值，
  此前正确性依赖栈地址复用（未定义行为）
- **CUDA Graph 捕获异常路径 stream 卡死**：capture 中抛 CudaException 时补充
  `cudaStreamEndCapture` 清理，避免 stream 永久停在 capture 状态
- **repetition_penalty 静默失效**：实现 llama.cpp 语义的重复惩罚（负 logit ×
  penalty、正 logit ÷ penalty，作用于 prompt + 已生成 token），greedy 与各
  采样策略统一生效；新增 `applyRepetitionPenalty` 公共静态辅助（供测试）
- **重复 seq_id 分配泄漏**：`KVCacheManager::allocateSequence(seq_id, ...)` 与
  FFI `tinyllm_allocate_sequence` 显式拒绝已存在的 seq_id，避免旧 slot 永久泄漏
- **FFI 越界校验缺失**：prefill 长度 / decode 绝对位置增加 `max_seq_len` 边界
  校验（hidden_buf 与 RoPE 表越界防护）；`logprobs_k` 增加 vocab_size 上限
- **异常路径 GPU 资源泄漏**：`InferenceEngine` 构造函数与 `ModelLoader::loadGGUF` /
  `loadBin` 增加 try/catch 清理（CUDA_CHECK 抛出时释放裸指针与已上传权重）
- **GGUF head_count_kv 缺失静默错配**：MHA 老 GGUF 缺该键时显式回退 num_heads，
  不再保持默认 32
- **GQA/head_dim 整除校验**：`validateModelConfig` 校验 `num_heads % num_kv_heads`
  与 `hidden_dim % num_heads`，畸形配置显式报错而非静默截断
- **权重 tensor 维度防御**：`load_quantized` / lm_head 校验 `dimensions.size() >= 2`
- **CLI `--max-tokens` 非法输入**：`std::stoi` 异常捕获 + 正值校验，不再直接 abort
- 清理编译警告：loadGGUF 未用变量、attention() 未用 position、kernel_bench 未用 name

### Added

- C ABI 执行后端（`include/tiny_llm/ffi.h` + `src/ffi.cpp`）：`tinyllm_load` /
  `tinyllm_step` / `tinyllm_allocate_sequence` / `tinyllm_free_sequence` / `tinyllm_free`，
  契约与 paged-serving `src/tiny_llm_ffi.rs` 逐字段对齐（策略 2：连续 KV，位置引擎内部跟踪）。
  真实模型端到端验证：prefill/decode 步进生成与 demo CLI 输出一致。

### Tests

- **repetition_penalty 单元测试**：llama.cpp 语义（正/负 logit、no-op、越界 id 忽略、
  greedy 避开重复 token）
- **重复 seq_id 拒绝测试**：KVCache 二次分配同一 id 返回错误且不消耗新 slot，
  释放后可复用
- **add_bias 非对齐尺寸测试**：rows*cols 非 256 倍数时结果正确且不越界
- C ABI 端到端测试（TLLM_GGUF_TEST_MODEL 门控）：load/allocate/step/free 全流程 +
  非法参数错误处理
- W8A16 大矩阵差分测试（M*N >= 4096 走 tiled 分支，与 reference 对齐）
- Attention GQA decode 与 CPU 参考逐元素对比（此前仅验证"不 crash/非零"）
- 真实模型权重量化往返测试（反量化 -> 转置 -> W8A16 量化 -> 重建误差受控）
- demo CLI 支持 `--prompt` / `--max-tokens` / `--show-tokens` / `--use-reference`（GPU 端到端生成入口）

### Verified

- tokenizer：C++ encode 与 HuggingFace tokenizers 权威实现逐 id 一致
  （151936 词表，含 CJK/emoji/缩写/空白/特殊 token 等 30 例）

- Qwen2.5-0.5B-Instruct Q4_K_M（GGUF v3，291 tensors，469MB）：
  配置提取与 Q5_0/Q4_K/Q6_K 首块反量化同 Python gguf 参考实现一致

## Releases

| Version | Date | Summary |
|---|---|---|
| v2.0.2 | 2026-04-27 | Code quality improvements, GGUF quantization utilities, and repository cleanup |
| v2.0.1 | 2026-04-16 | Bug-fix release for scale-dimension handling and loader cleanup |
| v2.0.0 | 2026-03-09 | Core engine milestone with KV cache API redesign |

## Policy

- This file is the only tracked changelog in the repository.
- Keep entries short and focused on meaningful user-facing or maintainer-relevant milestones.
- Do not duplicate release history in the documentation site.
