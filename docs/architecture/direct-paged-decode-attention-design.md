# Direct Paged Decode Attention 设计包（TLLM-P0-004）

> **Status: APPROVED（2026-09-14）—— 设计已定，可实现。**
> 本文是 `ai-infra-interview-prep/L3_L4_DESIGN_REVIEW_PACKAGES.md` §3 模板的填充版，
> 对应 §4（TLLM-DPA）与任务卡 `P0_P1_AGENT_BACKLOG.md` → TLLM-P0-004。
> 本包只包含设计，不含实现；实现按 §10 的 PR-1…PR-6 推进。
>
> **独立性缺陷（不得忽略）**：本包由作者编写、也由作者汇总决议，**不满足**
> 「实现 Agent 不应成为唯一 reviewer」的要求。§12 的独立性声明列出残余风险——
> **PR-1 改动现有热路径 kernel，合并前应有第二方复核其 diff 与性能数据。**

## 1. Decision summary

- **Chosen design**：在 `kernels/attention.cuh` 新增内部函数 `attention_decode_paged`，
  直接以「物理 pool + block table」寻址读取 K/V，不再把可见窗口 gather 成连续
  scratch。地址寻址通过一个可复用的 **tile loop + 寻址策略** 模板在
  `kernels/attention.cu` 内实现，与连续版共用 online-softmax 归约循环。
  kernel 只支持 batch=1、query_len=1；上层 `ffi.cpp` 的 per-sequence 循环不变。
- **Why**：
  1. 消除每 step 每 layer 一次整段可见窗口的 gather（读 + 写 scratch），这是当前
     legacy 路径相对 direct 的唯一额外工作量；
  2. 与 legacy 语义完全等价（同样的逻辑 K/V），因此可以做**逐元素相等**的差分门禁，
     而不是靠容差掩盖差异；
  3. 保持既有约束：device 端 `visible_tokens`、caller stream、无内部同步 → 仍可被
     CUDA Graph 捕获。
- **Rejected alternatives**：
  - *在 `kernels/attention.cu` 内复制一份 online-softmax 循环*：两份 90 行级联代码
    会漂移，且「direct 与 legacy 数值一致」这一门禁会因为两份实现而变弱。
  - *把 kernel 完全隐藏在 `TransformerLayer` 内、不暴露 kernel API*（§4.2 选项 3）：
    `attention.cu` 拿不到 `TransformerWeights`/`ModelConfig`，无法承担 dispatch；且
    §4.5 的 benchmark PR 需要直接调用 kernel，隐藏后不可测。
  - *按值传 POD view struct*（§4.2 选项 2）：C++17 下逐字段命名赋值确实比 14 个位置
    参数更抗"传错顺序"，但它只服务一个调用点，且 `kernels/*.cuh` 无传 struct 先例。
    **决定性理由是门禁已覆盖该失败模式**：`block_size` / `max_num_blocks` 传反时，
    host 侧 `table_len >= ceil(visible/block_size)` 校验或块 id 值域防护会让输出偏离，
    §8 的逐元素相等门禁立刻失败。既然失败模式已被覆盖，就不为一致性之外的收益引入
    新类型。（reviewer 若更看重"少一个位置参数陷阱"，此决定可低成本翻转。）
  - *host 端把 block table 拷回并校验块 id*：会引入 D2H 同步，直接破坏 CUDA Graph
    捕获。见 §7 的「非法块 id」决策。
- **Explicit non-goals**：
  - 不实现 direct paged **prefill**（本任务只做 decode，query_len=1）；
  - 不实现 batched / ragged decode（batch=1，per-sequence 循环仍在上层）；
  - 不输出 logsumexp，不引入 FP32/BF16 pool（仅 FP16 pool + FP32 累加 + FP16 输出）；
  - 不改 C ABI（`include/tiny_llm/ffi.h`）与 `paged-serving` 侧契约；
  - 不省掉 K/V scratch 的分配：direct 路径不再需要 scratch，但移除它要改 `ffi.cpp`
    的分配逻辑，超出本任务范围；显存收益记为 follow-up（见 §5）；
  - 不产生任何 TTFT/TPOT 或端到端 serving 声明。

## 2. Base evidence

- **Repository**：`open-infra-ai/tiny-llm`
- **Base branch**：`master`
- **Commit**：`2b15fb2b6e16671a91c648a5e1b3cf0666099b3f`（= backlog 审查参考 commit）
- **Dirty state**：干净。本设计的直接前置 TLLM-P0-002 已开 PR
  `open-infra-ai/tiny-llm#4`（分支 `tllm-p0-002-paged-oracle`，未合并）；
  TLLM-P0-004 的实现必须 rebase 到该 PR 合并后的 master。
- **Existing code anchors**：
  | 位置 | 事实 |
  |------|------|
  | `kernels/attention.cu::attention_decode_kernel` | grid = `num_q_heads`，block = 128 线程，动态 smem，`ATTEND_TILE=128` 的 online softmax |
  | 同上 L134 | 连续寻址 `k_pos = k + pos * kv_stride`，`kv_stride = num_kv_heads * head_dim` |
  | 同上 L97 | `visible_len` 从 device int 读取（CUDA Graph 重放前置条件） |
  | `kernels/attention.cuh::AttentionSmemLayout` | smem 字节数 = `(ATTEND_TILE + 8 + head_dim) * 4 + head_dim * 2` |
  | `kernels/paged_kv.cu` | 现有分页原语只有 scatter/gather，均以「本层 pool 指针 + block_table + max_num_blocks」为参数，索引统一 `size_t` |
  | `src/transformer.cpp::attentionPaged` | 当前 decode 走 `scatter → gather → attention_decode`；层偏移 = `layer_idx_ * max_num_blocks * block_size * kv_dim` |
  | `src/ffi.cpp` L404–497 | 策略 1 的 per-sequence 循环；`max_visible_tokens = max_num_blocks * block_size`；`decode_len` device int = `position + 1` |
- **Existing tests**：
  - `tests/test_kernels.cu`：`PagedKvTest.*`（scatter/gather 原语、非法块 id guard、层不重叠）；
    `RoPETest.*`；注意 `PagedKvTest.InvalidBlockIdIsGuardedNotDereferenced` 已冻结
    「非法块 id → gather 写 0」的语义。
  - `tests/test_transformer.cu`：attention decode/prefill 与 KV cache 行为。
  - `tests/test_ffi.cpp::PagedKVStrategyMatchesContiguous`：**需要真实 GGUF，默认 skip**。
  - TLLM-P0-002（PR #4）：`tests/paged_attention_oracle.h` 独立 host 参考 +
    `tests/test_paged_oracle.cpp` kernel/layer 级差分，8 项，已过变异检验与
    Compute Sanitizer。
- **Existing GPU/performance evidence**：`docs/performance/results/` 中的 CUDA Graph
  A/B（2026-08-23）与 `kernel_bench` 的连续 attention 计时。**没有任何 direct paged
  的数据**；CUDA Graph 的历史数字不能用来回答本任务的收益。
- **Unknown / 必须在实现前关闭**：已由 §12 的 Reviewer 决议全部关闭。
  - 已决定：逐元素相等（§8）、抽取共享 tile loop 且附性能不回归检查（§10 PR-1）、
    支持几何上界（§3）、flat 参数（§3）、零行语义冻结（§4.3）、fallback 以开关
    表达（§7）、scratch 保留到 follow-up（§5）。
  - 仍待实测（不阻塞设计）：本机 profiler（ncu/nsys）可用性——历史记录为不可用；
    若仍不可用则 §9 的 profiler 问题标 `not_run`。

## 3. API / ABI

- **Public or internal**：**internal**。声明在 `kernels/attention.cuh`，与现有
  `attention_decode` 同层；`kernels/` 不依赖 `include/tiny_llm/`，不进入 C ABI，
  不进入 `paged-serving`。无 versioning 需求；唯一调用方是 `src/transformer.cpp`。
- **Signature（冻结草案）**：

```cpp
// 直接以物理 pool + block table 寻址的 decode attention。
// Q:    [1, Hq, D]（FP16）
// K/V pool: 本层指针，[max_num_blocks, block_size, Hkv, D] 连续（FP16）
// block_table: device int[table_len]，元素为物理块 id
// O:    [1, Hq, D]（FP16）
// device_visible_tokens: device int，可见 KV token 数（= position + 1）
void attention_decode_paged(const half *__restrict__ query,
                            const half *__restrict__ k_pool_layer,
                            const half *__restrict__ v_pool_layer,
                            const int  *__restrict__ block_table,
                            half       *__restrict__ output,
                            float scale,
                            int   num_q_heads,
                            int   num_kv_heads,
                            int   head_dim,
                            const int *device_visible_tokens,
                            int   block_size,
                            int   max_num_blocks,
                            int   table_len,
                            cudaStream_t stream = 0);
```

- **Parameter order and types**：与 `attention_decode` 保持同样的「指针 →
  `scale` → 头数 → `head_dim` → device int → 几何 int」次序，把 paged 特有的
  `block_size / max_num_blocks / table_len` 追加在 device 指针之后，使两者在
  diff 中可直接对照。整数一律 `int`（与既有 kernel 一致），**指针运算一律
  `size_t`**（见 §4）。
- **Shape/dtype/stride/device**：见 §4。dtype 仅 FP16 + FP32 累加；所有指针均为
  device 端；无 stride 参数（布局由 §4 公式唯一确定）。
- **Size/alignment/field order**：不适用（无 struct、无 C ABI 变更）。
- **Compatibility/versioning**：内部 API，无版本化；`TransformerLayer` 的 host 结构
  `PagedKVCacheView` **不新增字段**——`visible_blocks` 即 block table 长度
  （`src/ffi.cpp` 已如此赋值，PR #4 已为它加上长度校验）。
- **调用方迁移**：无。legacy 路径保留，见 §11。
- **G1 未决项**：无（参数语义见 §4；`table_len` 与 `visible_tokens` 的关系已在
  PR #4 冻结为 `table_len >= ceil(visible_tokens / block_size)`）。

## 4. Data layout and numerics

### 4.1 逻辑 shape 与线性地址公式（冻结）

```text
kv_dim       = num_kv_heads * head_dim                // 元素（half）数
head_stride  = head_dim
row_stride   = kv_dim                                 // 一个 token 的 K（或 V）跨度
block_stride = block_size * kv_dim                    // 一个物理块的跨度
layer_stride = max_num_blocks * block_size * kv_dim   // 一个 layer 的跨度

第 ℓ 层由 caller 预先偏移（本包规定 layer 偏移在 caller 计算，见 4.3）：
  k_layer = k_pool + ℓ * layer_stride

可见 token t ∈ [0, visible_tokens)：
  b = t / block_size            // 逻辑块
  r = t - b * block_size        // 块内偏移
  p = block_table[b]            // 物理块 id
  K[ℓ, t, kh, d] 在 k_layer 内的 half 元素偏移 =
      (size_t)p * block_stride + (size_t)r * row_stride + (size_t)kh * head_stride + (size_t)d
```

GQA 映射（与 `kernels/attention.cuh` 既有 contract 一致）：

```text
group_size = num_q_heads / num_kv_heads        // 必须整除
kv_head(q_head) = q_head / group_size
```

### 4.2 数值

| 项 | 冻结值 |
|----|--------|
| pool / Q / O dtype | FP16（`half`） |
| 累加 dtype | FP32（`float`，与连续版一致：`scores` / `out_acc` 为 FP32） |
| score | `scale * dot(q, k)`，`scale = 1 / sqrt(head_dim)`，FP32 |
| softmax | online softmax，tile = `ATTEND_TILE = 128`；`exp` 用 `__expf`（与连续版一致） |
| 最终归一化 | `inv_sum = 1 / (running_sum + 1e-9)`，然后 `__float2half` |
| mask | 无（decode，query_len=1，全部可见 token 参与） |
| 容差 | 见 §8：direct 与 legacy 要求**逐元素相等**（同一 tile loop、同一输入） |

**不使用**：FP32/BF16 pool、logsumexp 输出、split-KV、flash-decoding。

### 4.3 边界、非法输入与溢出

- **`visible_tokens` 恒等于 `position + 1`**（decode）。这是当前 `attentionPaged`
  与 `ffi.cpp` 的隐式不变量，本包将其**显式冻结**：`*device_visible_tokens` 必须
  等于 host 侧用于校验的 `kv.position + 1`。
- **`visible_tokens = 0`**：tile loop 不执行 → `running_sum = 0` →
  `inv_sum = 1/(0+1e-9)`，`out_acc = 0` → 输出全 0（非 NaN）。与连续版行为一致。
- **非法 `block_table[b]`（`p < 0` 或 `p >= max_num_blocks`）**：**冻结为「该 token 的
  K 行与 V 行按 0 处理，但仍参与 softmax 归一化」**，即
  `score = 0`（跳过点积）但**仍计入** tile max / sum，V 贡献 0。
  理由：`PagedKvTest.InvalidBlockIdIsGuardedNotDereferenced` 已冻结 gather 写 0 的
  语义，而「K 行 = 0」在 softmax 中并非无贡献条目；direct 若跳过该条目，softmax
  归一化会与 legacy 不同，逐元素相等门禁立刻失败。这是**有意**保持 legacy 等价，
  不是可选的优化。
- **整数宽度**：kernel 内所有 pool 偏移用 `size_t`（与 `paged_kv.cu` 现有做法一致）。
  `(size_t)p * block_stride + r * row_stride + kh * head_stride + d` 的上界是
  `layer_stride`，而 `layer_stride` 由 host 分配时已验证不溢出，故 kernel 内无回绕。
  不要为省寄存器把 `p * block_stride` 写成 `int`。
- **loop 边界**：`tile_start < visible_tokens`，`tile_size = min(ATTEND_TILE, visible_tokens - tile_start)`。
  与 block 边界无关（token 级寻址），因此 block_size=1 与 `visible_tokens = k*block_size ± 1`
  都不需要特殊分支。
- **NaN/Inf**：输入 pool 含 NaN/Inf 时与连续版行为一致（不额外拦截）。

## 5. Ownership and lifecycle

- **Allocator**：kernel **不分配任何内存**。K/V pool、block table、Q/O buffer、
  `device_visible_tokens` 全部由调用方持有：
  - 策略 1（paged）生产路径：`src/ffi.cpp` 的 handle（`paged_k_pool` / `paged_v_pool` /
    `d_block_tables` / `decode_len`），生命周期与 handle 一致；
  - 测试：`DeviceBuffer`。
- **Owner / reuse scope**：pool 与 table 为 request/sequence 级，由上层管理；
  kernel 视角是只读输入。新增 **零** 有状态资源 → 无新的 reallocation、
  destruction、partial-init 路径。
- **scratch 在 direct 路径下成为冗余（有意保留）**：`k_scratch` / `v_scratch` 只服务
  legacy gather，direct 路径不读它们。本任务仍要求二者非空（入口校验不变），理由是
  去掉它们必须改 `ffi.cpp` 的分配，会与「PR-2/PR-3 不碰 FFI」的拆分冲突。
  代价是每 handle 白占两块 `max_visible_tokens * kv_dim` 的显存；
  **follow-up**：direct 稳定后，按 `TLLM_PAGED_ATTENTION` 取值决定是否分配 scratch，
  并记录实测显存差异。
- **Reallocation**：不存在（pool 在 load 时按 `max_num_blocks * block_size * kv_dim`
  一次分配，`max_visible_tokens` 即该容量）。
- **Success / cancel / timeout / error cleanup**：kernel 无资源，无需清理；调用失败
  （launch error）由 §7 处理，不改变 pool 内容。
- **G3 拒绝条件对照**：无 function-level static/raw pointer（相比 host-int 版
  `attention_decode` 的 `static thread_local DeviceBuffer` 包装，本 API **只接受 device
  指针**，天然可并发，见 §6）；owner 销毁时不会有 in-flight work 的额外风险
  （与现有 kernel 相同：调用方负责 stream 顺序与同步）。

## 6. Stream and concurrency

- **Caller stream**：唯一参数 `cudaStream_t stream`，默认 0；kernel 内不创建 stream、
  不创建 event、**不调用 `cudaDeviceSynchronize`**。
- **Internal stream/event**：无。
- **Supported concurrency**：
  - 同 stream 顺序调用：支持（无内部状态）。
  - 不同 stream 并发、不同 handle 并发：支持，前提是调用方不并发写同一 block table
    或同一 pool 区域（与现有 kernel 的契约相同）。
  - **不提供** host-int 便捷重载：`attention_decode` 的 host-int 版本使用
    `static thread_local` 缓冲并在注释中声明「不适用于并发多线程」，paged 版本不复制
    这个陷阱；测试若需要 host int，必须在测试侧自备 `DeviceBuffer<int>`。
- **CUDA Graph 可捕获性（硬要求）**：launch 路径上不得有 D2H 拷贝、不得有分配、
  不得有 host 侧读取 device 数据。`visible_tokens` 必须走 device int；因此
  **block table 的合法性只能在 device 上按 §4.3 的 guard 处理，不能在 host 上校验**。
- **Unsafe combinations**：在同一个 step 内对同一 sequence 并发 launch 两个
  direct/legacy decode；在 graph capture 期间修改 block table 内容（table 指针固定，
  内容由调用方在 capture 外的同一 stream 上更新）。

## 7. Error and fallback

分层策略：**host 侧校验 + 返回 `Result`；kernel 侧只保留文档化的前置条件**（与仓库
现有分层一致——`kernels/` 不依赖 `include/tiny_llm/result.h`）。

- **Validation errors（在 `TransformerLayer::forwardPaged` 内，返回
  `Result<void>::err`）**：
  | 条件 | 现状 | 本设计要求 |
  |------|------|-----------|
  | `k_pool/v_pool/block_table/k_scratch/v_scratch == nullptr` | 已校验 | 保留（direct 路径不再需要 scratch，但保留校验以保持两条路径同一入口契约） |
  | `block_size <= 0`、`max_num_blocks <= 0`、`max_visible_tokens <= 0` | 已校验 | 保留 |
  | `position < 0` 或 `position + num_tokens > max_visible_tokens` | 已校验 | 保留 |
  | `visible_blocks < ceil(visible_tokens / block_size)` | PR #4 已加 | 保留（这就是 `table_len` 校验） |
  | `num_q_heads % num_kv_heads != 0` | **不在本任务修，见 §7.1** | **不在 kernel / `forwardPaged` 重复校验**：唯一有效的修复位置是 C ABI 载入边界 |
  | `head_dim <= 0` | `forwardPaged` 依赖 config，未显式校验 | 要求 `head_dim > 0` 且 `AttentionSmemLayout{head_dim}.total_bytes() <=` device 动态 smem 上限 |

### 7.1 模型几何校验的位置（本设计的 Q5 结论）

设计初期主张在 `forwardPaged` 里加 `num_q_heads % num_kv_heads != 0` 校验，理由是
"连续版 `attention_decode` 静默取整，属既有缺口"。**查证后该描述不准确，位置也不对**：

- `Validator::validateModelConfig` **已经**实现了这组校验（含 `hidden_dim % num_heads`
  与 `head_dim` 偶数），注释还写明"静默截断会导致 kv_head 映射错位"；
- 但它全仓只在 `inference_engine.cpp` 被调用一次，C ABI 路径 `tinyllm_load` **不调用**；
- 而 `GGUFParser::extractModelConfig` 对这类元数据是"补默认值"而非报错，且
  `head_dim = hidden_dim / num_heads` 同样是不校验整除性的截断除法。

后果不是"映射错位"这么温和：`kv_head = q_head / (num_heads / num_kv_heads)` 越界后，
最后一个 token 的 K/V 读越过缓冲末尾。已用最小复现证实（`Hq=14, Hkv=3`）：

```text
Invalid __global__ read of size 2 bytes
  at tiny_llm::kernels::attention_decode_kernel
  Access to 0x718000000 is out of bounds，位于分配末尾之后 1 字节
cudaDeviceSynchronize -> unknown error       // CUDA 上下文被毒化
ERROR SUMMARY: 13 errors
```

**结论**：修复落在外部边界 `tinyllm_load`（`extractModelConfig()` 之后、`loadGGUF()`
之前调用 `Validator::validateModelConfig`），一处覆盖整条 C ABI 路径（attention
prefill/decode、RoPE、FFI 缓冲尺寸），对合法模型零行为变化。
**不在 kernel 层加**——`kernels/` 没有 `Result` 通道，在那里"校验"只能静默 return，
比现状更糟；**也不只修 `forwardPaged`**——那样 prefill 分支与策略 2 仍然暴露。
该修复已作为独立 PR 提交（`open-infra-ai/tiny-llm#6`），与本任务解耦。

- **Fallback = 开关，不是"几何不支持"**：本设计**不引入**按几何自动回落的路径。
  理由是那条分支不可达也不可测：smem 需求为 `(ATTEN_TILE + 8 + head_dim) * 4 +
  head_dim * 2` 字节，`head_dim = 128` 时才 1.3 KB，要碰到 48 KB 上限需要
  `head_dim > 7800`。真正的 fallback 是显式开关 `TLLM_PAGED_ATTENTION=auto|legacy|direct`
  （§11），它可以直接被测。若将来出现**真实可达**的不支持几何，再引入自动回落，
  并按 §4.6 加计数器与 `TLLM_WARN`；benchmark 不得把回落运行计入 direct 数字。
- **OOM**：本 kernel 不分配，无新 OOM 路径。
- **Launch/runtime errors**：沿用仓库现状（kernel launch 后由 `cudaGetLastError` /
  测试侧 `cudaDeviceSynchronize()` 断言捕获）；本设计**不引入**「kernel 内静默 return」
  作为错误处理——参数非法必须由 host 校验拦下，而不是让 kernel 悄悄不做事。
- **Partial success**：不适用（单 sequence、单 token）。
- **Observability**：`TLLM_ERROR`（校验失败）；当 `TLLM_PAGED_ATTENTION` 显式把 direct
  降级为 legacy 时打 `TLLM_WARN` 一次，使 benchmark / 结果包能区分"跑的到底是哪条路"；
  无静默失败。
- **G5 拒绝条件对照**：不吞错误（校验在返回 `Result` 的 host 层）、fallback 可观测且
  不被记为目标 fast path。

## 8. Correctness matrix

独立参考 = `tests/paged_attention_oracle.h`（TLLM-P0-002）。**它不调用任何生产 kernel，
不复用生产寻址/归约**，因此满足 G6/G2 对 reference 独立性的要求。

三层门禁，逐层定位失败：

1. **kernel 级 direct vs 独立 oracle**：与 TLLM-P0-002 的 kernel 级矩阵同集合，
   把 `scatter + gather + attention_decode` 换成 `attention_decode_paged`；
2. **kernel 级 direct vs legacy（逐元素相等）**：同一 pool、同一 table、同一 Q，
   两条路径输出**必须逐元素相等**（同一 tile loop + 同一输入 ⇒ 无浮点差异）。
   这一条是定位「寻址错误」的主门禁，且比容差比较强得多；
3. **layer 级 direct vs legacy vs contiguous（逐层 K/V + 最终 hidden）**：
   复用 TLLM-P0-002 的 layer 级方法（按冻结公式读回 pool），把 decode 分支切到
   direct 后再比一次。

| Case | Reference | Expected | GPU required | Sanitizer |
|------|-----------|----------|--------------|-----------|
| block_size = 1 / 16 / 32 | oracle + legacy | 逐元素相等 | 是 | 是 |
| visible = 1 / block−1 / block / block+1 / 2·block+tail | oracle + legacy | 逐元素相等 | 是 | 是 |
| MHA(4/4) / GQA(8/2) / MQA(8/1) | oracle + legacy | 逐元素相等 | 是 | 是 |
| head_dim = 32 / 64 / 128 | oracle + legacy | 逐元素相等 | 是 | 是 |
| 非连续 / 复用顺序物理块表 | oracle + legacy | 逐元素相等 | 是 | 是 |
| 非法块 id（负值、`== max_num_blocks`） | oracle（零行语义） | 逐元素相等，且不 fault | 是 | 是 |
| `visible_tokens = 0` | 全 0 输出 | 逐元素相等 | 是 | 是 |
| `table_len` 恰好 = required（off-by-one） | host 校验 | `forwardPaged` 返回 err（少 1 块必须拒绝） | 否（host） | — |
| `num_q_heads % num_kv_heads != 0` | C ABI 载入边界校验（§7.1 / PR #6） | `tinyllm_load` 返回错误，不进入 kernel | 否（host） | — |
| 随机 seed × 多几何 | oracle | 逐元素相等 | 是 | 是 |
| 多 layer pool offset（2–3 层） | contiguous cache | 逐层 K/V 逐元素相等 | 是 | 是 |
| allocate→prefill→多次 decode→free→reuse | legacy 差分 | 全程一致，free 后可复用 | 是 | 是 |
| non-default stream | legacy 差分 | 与 default stream 结果一致 | 是 | 是 |
| 端到端 token/logit canary（真实 GGUF） | legacy | token 一致 | 是（且需模型） | 是 |

- **CPU/build 与 GPU correctness 分开报告**：host 校验类用例无 GPU 也运行；
  所有 kernel 用例在无 device 时 **skip 且计数可见**，不得把 skip 记为 pass。
- **Sanitizer**：`compute-sanitizer --tool memcheck` 跑新增用例，要求 0 error；
  这是与 TLLM-P0-002 同一门禁。
- **变异检验（沿用 TLLM-P0-002 的方法）**：至少验证门禁能捕获
  ① 块内偏移丢 `r`；② 层偏移丢 `max_num_blocks`；③ 非法块 id 被跳过而非计零。
- **G6 拒绝条件对照**：correctness 早于 benchmark；三层差分可定位 kernel/layer/FFI；
  skip 可见；oracle 不复用生产寻址。

## 9. Benchmark plan

- **Measurement layer**：**kernel 级**，扩展 `src/kernel_bench.cpp`。**不写 TTFT/TPOT**。
- **Baseline（语义等价，因此允许算 speedup，但必须给出等价证据）**：
  1. `legacy` = `paged_gather_blocks` ×2 + `attention_decode`；
  2. `contiguous` = `attention_decode`（连续 scratch，仅作上界参考）；
  3. `direct` = `attention_decode_paged`。
  三者输入同一逻辑 K/V，输出已在 §8 证明逐元素相等（先正确性、后计时）。
  另单独报告两条 gather 的耗时，以便区分「省下的 gather」与「direct 自身的开销」。
- **Shapes/workload**：Qwen2.5-0.5B decode 形状 `Hq=14, Hkv=2, D=64`；
  `visible_tokens ∈ {8, 32, 64, 128, 512, 1024, 2048}` × `block_size ∈ {16, 32}`；
  再加 block 边界形状 `visible = k·block_size ± 1`。
- **Warmup/repetitions**：warmup 20、reps 200（沿用 `kernel_bench` 现有约定）；
  A/B 交替（interleaved）执行以避免时钟漂移；报告 median / p10 / p90 与 CV。
- **Raw artifact**：每次运行输出原始样本（JSON 或 CSV），并记录
  GPU、driver、CUDA/nvcc 版本、`CMAKE_BUILD_TYPE`、tiny-llm commit + dirty、
  输入 seed、warmup/reps、限制说明。CV > 10% 的 shape 必须标 `not_converged`，
  不得删除失败/OOM shape。
- **Convergence**：同一 shape 重复 3 次运行，median 差异 < 10% 才认为收敛。
- **Profiler question**：若 ncu 可用，回答「direct 路径是否被 block table 的随机
  访问延迟主导，gather 省下的 DRAM 流量中有多少转化为 wall-clock」；若不可用
  （本机历史记录为不可用），标 `not_run` 并保留该问题。
- **可证伪假设（不得写成结论）**：legacy 每 step 每 layer 的 K/V 读放大为
  direct 的约 2 倍（gather 读一遍 + attention 读一遍），另有 scratch 写入；
  因此 direct 在小 `Hq`/`Hkv` 形状下可能受低 occupancy（grid = `Hq` 个 block）
  限制而看不到等比收益。
- **G7 拒绝条件对照**：correctness 已过才计时；多次运行；有 raw data；
  CV 超限标注；等价性有证据所以允许 speedup；**不把 kernel 结果写成 TTFT/TPOT**。

## 10. PR and ownership plan

| PR | 内容 | 允许文件 | 门禁 |
|----|------|----------|------|
| PR-1 重构（行为不变） | ~~把 decode 的 tile loop 抽成共享模板~~ **执行时被门禁否决（+1.4~2.3% 回归，§10.1）；后按 issue #8 翻转为接受回归并改回共享循环（§10.2），已并入 PR-2** | `kernels/attention.cu` | 连续路径数值逐位不变 + 现有测试全过 + 回归已测量并显式记录 |
| PR-2 kernel | 新增 `attention_decode_paged`（与连续版共用 `decode_online_softmax`，各自一份取址策略）+ 头声明 + kernel 级差分测试 + sanitizer | `kernels/attention.{cu,cuh}`、`tests/**` | §8 的 kernel 级 1/2 层；**不改 FFI、不改 Transformer** |
| PR-3 runtime dispatch | `attentionPaged` decode 分支按 `TLLM_PAGED_ATTENTION` 分发（`auto`/`legacy`/`direct`；默认 `legacy`）；direct 时跳过 gather；非法取值显式报错；显式 legacy 打一次 `TLLM_WARN` | `src/transformer.cpp`、`tests/**` | **已提交：#9**。layer 级逐位比对 + 开关三态 + prefill 不受影响的测试；变异检验 3 项 |
| PR-4 ABI/integration | **计划为空**：本设计不改 C ABI。若实施中发现必须改，则与 `paged-serving` 成对提交并按 §5 的 ABI 包评审 | — | 未发生则不开 PR |
| PR-5 benchmark | 三路 kernel benchmark + 结果归档（须绑定 PR-2/3 的 correctness commit）；**通过后把 `TLLM_PAGED_ATTENTION` 默认值由 `legacy` 改为 `auto`** | `src/kernel_bench.cpp`、`docs/performance/**` | §9；raw data 与 provenance |
| PR-6 docs | 更新能力边界（仅在证据完成后） | `docs/architecture/**`、`CHANGELOG.md` | 只能引用已归档证据 |

- **一个 PR 只做一层**：PR-1 只重构、PR-2 只加 kernel、PR-3 只接线；不得把
  contract + 实现 + benchmark + 结论塞进同一个 PR。benchmark PR 不改 kernel。
- **Shared-file owner**：`kernels/attention.cu` 在 PR-1 与 PR-2 之间**串行**，
  由同一 owner 完成，避免并行改动冲突。
- **跨仓顺序**：本设计不触碰 C ABI，故与 `paged-serving` 无强制顺序；但 PR-3 之后
  的 PSRV-P1-002/004 若要使用 direct 收益，必须引用 PR-5 的结果包。

### 10.1 执行结果：PR-1 被门禁否决（2026-09-14）

PR-1 计划把 decode 的 tile loop 抽成 `__device__` 模板 + 取址策略，让连续 KV 与分页
KV 共用同一份 online softmax。**实测该抽取给生产 kernel 带来可复现的回归，因此按
§10 的门禁回退为复制实现，PR-1 未提交。**

测量过程中先后排除了三个会把结论带偏的因素，记在此处以免后人重蹈：

1. 本机 GPU 空闲时 SM 时钟停在 **900/3090 MHz**，小 kernel 推不动 boost，逐次运行
   差异可达 50%；
2. 临时 scratch 程序默认编译到 **sm_75**，而生产构建用 `native`（sm_120）；
3. 现成 harness 的 host-int 重载每次调用附带一次 4 字节 H2D memcpy。

修正后的方法：时钟预热 4s + `-arch=native` + device-int 重载 + 大 S（512…2048）+
顺序平衡交替 + **新旧 kernel 编入同一进程交替调用**（消除跨二进制代码布局这一最后的
混淆）。

结果（同进程 A/B，8 轮顺序平衡，两组独立重复）：

| 几何 | Δ |
|------|-----|
| S=128 / 512 / 1024 / 2048（D=64, Hq=14） | +2.3% / −2.6% / +2.0% / +0.9% |
| D=128, S=512 | **+4.9%** |
| D=128, S=1024 / 2048 | +1.6% / +0.7% |
| Hq=8, Hkv=1, S=1024 | +1.6% |

两轮分别 7/8 与 6/8 几何为正，均值 +1.4% / +2.3%。SASS 对比：指令数相同（1168）、
寄存器相同（56）、无 spill，但**指令调度与选择确有变化**——不是测量假象。

期间尝试过两种规避写法，均未改变结论：策略按值 / 按引用传递；以及把"无效行"从
`nullptr + 分支`改为返回**零行**（使共享循环完全无分支）。后者本身是实现上的更优形态
（语义与 gather 写 0 构造性一致），仍测到同样回归——说明扰动来自抽取本身，而非某个
具体写法。

**结论**：PR-2 改为自持一份寻址实现（`attention_decode_paged_kernel` 复制归约循环，
只替换 K/V 取址）。代价是两份实现必须保持数值一致；**缓解手段是 PR-2 新增的
direct vs legacy 逐元素相等门禁**——任何漂移立即失败。该门禁比原计划的"纯寻址差分"
更强：它要求两份**独立实现**逐位一致。

若将来重新评估这一取舍，需要的是一个**不改变 `attention_decode` 代码生成的抽取方式**，
而不是重测同一方案。

### 10.2 后续：该取舍已按 issue #8 翻转（2026-09-14）

**结论已改为「接受回归、改回共享循环」。** 见 `open-infra-ai/tiny-llm#8`。

翻转的理由与本节的原始记录并不矛盾——它质疑的是门禁**回退方案的收益**，而非测量本身：

1. 复制方案唯一的风险是"两份实现漂移"，而这个风险已由 §8 的 direct vs legacy
   **逐位相同**门禁自动覆盖。也就是说：**安全来自门禁，不来自副本数量**；复制换来的
   不是额外保障，而是双份维护成本。
2. 代价量级：+1.4~2.3% 落在单个 kernel，换算端到端约 0.1~0.3%，不划算。
3. 反向收益：共享后 direct 与 legacy 的差分退化为**纯寻址测试**，定位"寻址错"与
   "归约错"的能力更强。

落地形态（PR-2 的第二个 commit）：抽出 `decode_online_softmax` 模板 + 取址策略契约，
`ContiguousRows` / `PagedRows` 两份策略。无效行返回**共享内存零行**而非 `nullptr`，
因此循环内无任何有效性分支。

新增的验证（比原计划更强）：

- 连续路径数值**逐位不变**（10 组几何的 fp16 位模式指纹，抽取前后完全一致）；
- 变异检验第三项：**在共享循环里丢掉 online rescale**（两条路径同等出错）→ 逐位门禁
  通过、**独立 oracle 捕获**。这正面验证了"共享归约 + 独立参考"分层门禁的必要性：
  若只有逐位差分而没有独立参考，这一类错误会整体漏过。

§10.1 中记录的测量方法与回归量级仍然有效，保留作为该决策的事实依据。

## 11. Rollback

- **Preserved legacy path**：`scatter → gather → attention_decode` **不删除**，
  至少在 direct 路径经过固定的观察期与结果矩阵（PR-5 完成）之后才讨论删除。
- **Feature flag / fallback**：dispatch 由 `TLLM_PAGED_ATTENTION` 控制，取值
  `auto | legacy | direct`（大小写不敏感），**默认（未设置）= `legacy`**；PR-5 通过后
  改为 `auto`。**本设计不提供"按几何自动回落"**：那条分支不可达也不可测（§7），
  开关本身就是 fallback，三态都可直接被测。
  - 已实现的语义（PR-3 / #9）：`auto` **当前等价于 `direct`**（没有可达的不支持几何，
    保留该取值是为了将来不必改调用方）；`legacy` 显式选择时打一次 `TLLM_WARN`，
    便于结果包区分实际路径；**非法取值显式返回错误**，不静默回退（G5）。
  - 开关**不做进程级缓存**：每次调用解析（约 20 ns、无堆分配），使 `setenv` 在测试中
    即时生效，因而不需要为测试暴露 reset seam。该解析只在分页路径上执行。
  - 只影响 strategy 1（分页 KV）的 **decode**；prefill 与 strategy 2（连续 KV）不变。
- **Trigger（任一命中即回滚到 `legacy`）**：
  1. direct 与 legacy 的逐元素相等门禁在任何 shape 上失败；
  2. Compute Sanitizer 报错；
  3. 真实模型 canary（`TLLM_GGUF_TEST_MODEL`）token 不一致；
  4. `paged-serving` 集成路径出现回归。
- **Procedure**：把默认值从 `auto` 改回 `legacy`（单点改动），保留 direct 代码与其
  失败用例；在中性提交上复现问题后再决定修复或撤销。
- **Artifact 保留**：失败 shape、OOM、`not_converged` 结果全部保留在
  `docs/performance/results/` 对应目录，不删除。
- **G8 拒绝条件对照**：PR 已拆分；有 legacy fallback 与显式 flag；benchmark Agent
  不优化算法；不改 C ABI，因此不会单侧破坏 `paged-serving`。

## 12. Approval

### 门禁自检

| 门禁 | 自检 | 说明 |
|------|------|------|
| G0 事实与范围 | 通过 | §2 绑定 exact commit 与 dirty；显式区分「已实现 / 证据不足 / 未来目标」；明确不把 paged bookkeeping 写成 direct PagedAttention |
| G1 API/ABI | 通过 | §3 冻结内部签名（flat 参数）；无未决参数 |
| G2 布局与数值 | 通过 | §4 给出完整线性地址公式与各 stride；非法块 id 语义与 legacy 显式对齐；整数宽度与上界已说明 |
| G3 所有权与生命周期 | 通过 | kernel 零分配、零状态；无 static/raw pointer；scratch 冗余已显式记录并给出 follow-up（§5） |
| G4 stream 与并发 | 通过 | caller stream、无内部同步、graph 可捕获；不提供 thread_local 包装 |
| G5 错误语义 | 通过 | §7 分层校验；fallback 以开关表达且三态可测；不静默 return |
| G6 correctness | 通过 | §8 三层门禁 + sanitizer + 变异检验；oracle 独立 |
| G7 性能 baseline | 通过 | §9 baseline 语义等价且有证据；raw data、收敛标准、profiler 问题齐备 |
| G8 合并与回滚 | 通过 | §10 六个 PR 分层；§11 flag + trigger + procedure；PR-1 附性能不回归检查 |

### 决议（2026-09-14）

| # | 议题 | 决议 |
|---|------|------|
| Q1 | flat 参数 vs POD view | **flat**（§1 已补理由：逐元素相等门禁覆盖了传参顺序这一失败模式） |
| Q2 | 共享 tile loop vs 复制 | **抽取共享 loop**；PR-1 必须附 `attention_decode` 前后性能对比，出现可测回归则退回复制 → 执行时门禁触发（+1.4~2.3%，§10.1），短暂回退为复制；随后按 issue #8 判定**接受回归、改回共享循环**（§10.2）。最终形态为共享 |
| Q3 | 非法块 id = 零行且参与 softmax | **冻结为稳定契约**；"设备端违规计数"仅作为可观测性 follow-up（需改 FFI，超出本任务） |
| Q4 | direct vs legacy 逐元素相等 | **要求严格相等**（已复核包括非法块 id 在内的每个分支都逐位一致） |
| Q5 | `num_q_heads % num_kv_heads` 校验位置 | **改在 C ABI 载入边界**（§7.1），不落在 kernel / `forwardPaged`；已作为独立 PR #6 提交 |
| Q6 | 收益表述边界 | **接受**：kernel 级结果不得外推为 TTFT/TPOT；serving 级结论须另开实验与结果包 |
| Q7 | direct 路径下的 scratch | **本任务保留**（避免改 `ffi.cpp` 分配）；显存收益记为 follow-up（§5） |
| Q8 | 按几何自动回落 | **取消**：该分支不可达且不可测；fallback 只由 `TLLM_PAGED_ATTENTION` 开关表达（§7 / §11） |

- **Reviewer**：仓库 owner（本轮将决议委托给作者的分析与证据）
- **Decision**：`approved` —— 设计可行，PR-1…PR-6 可按 §10 推进

### 独立性声明（重要，不得省略）

本包由作者编写、也由作者汇总决议，**不满足「实现 Agent 不应成为唯一 reviewer」的
独立性要求**（`NEXT_AGENT_START_HERE.md` §12）。上述决议中：

- Q1 / Q2 / Q3 / Q7 / Q8 是设计取舍，且均有门禁或不可达性论证兜底；
- **Q2 是唯一有真实爆炸半径的决定**（改现有热路径 kernel）。在 PR-1 合并前，
  建议由第二方复核 PR-1 的 diff 与前后性能数据；
- Q5 不依赖本决议：它由最小复现 + Compute Sanitizer 证据驱动，并已拆成独立 PR #6。

### 本轮修正的设计缺陷（记录在案）

1. **Q5 原表述错误**：原文写"连续版 `attention_decode` 静默取整，属既有缺口"，
   并提议在 `forwardPaged` 新增校验。实际 `Validator::validateModelConfig` 早已实现
   该校验，只是不在 C ABI 路径上；正确修复位置是 `tinyllm_load`（§7.1）。
2. **Q8 原含死代码**："几何不被 direct 路径支持则回落"的分支需要 `head_dim > 7800`
   才可达，既不可测试也不构成真实 fallback，已删除。
3. **Q7 原未记录**：direct 路径令 scratch 冗余，原文只说"保留校验"而未记录显存代价
   与 follow-up，已在 §5 补充。
