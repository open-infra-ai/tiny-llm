# Direct Paged Decode Attention 设计包（TLLM-P0-004）

> **Status: DRAFT — 待评审，未批准。**
> 本文是 `ai-infra-interview-prep/L3_L4_DESIGN_REVIEW_PACKAGES.md` §3 模板的填充版，
> 对应 §4（TLLM-DPA）与任务卡 `P0_P1_AGENT_BACKLOG.md` → TLLM-P0-004。
> 在 §12 的 Decision 变为 `approved` 之前，**不得修改生产 kernel 的算法**；本包本身
> 只提交设计，不包含实现。
>
> 作者自检 ≠ 批准。§12 的每项门禁由 reviewer 独立判定，作者不代签。

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
  - *按值传 POD view struct*（§4.2 选项 2）：字段顺序会成为跨 TU 的事实契约，为一个
    单请求 kernel 引入不必要的 ABI 冻结面；`kernels/*.cuh` 目前无传 struct 的先例。
  - *host 端把 block table 拷回并校验块 id*：会引入 D2H 同步，直接破坏 CUDA Graph
    捕获。见 §7 的「非法块 id」决策。
- **Explicit non-goals**：
  - 不实现 direct paged **prefill**（本任务只做 decode，query_len=1）；
  - 不实现 batched / ragged decode（batch=1，per-sequence 循环仍在上层）；
  - 不输出 logsumexp，不引入 FP32/BF16 pool（仅 FP16 pool + FP32 累加 + FP16 输出）；
  - 不改 C ABI（`include/tiny_llm/ffi.h`）与 `paged-serving` 侧契约；
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
- **Unknown / 必须在实现前关闭**：
  1. reviewer 是否接受「direct 与 legacy 逐元素相等」而不是容差比较（§8 主张相等）；
  2. tile loop 抽取重构（§10 PR-1）是否被接受，还是要求复制实现；
  3. 本机 profiler（ncu/nsys）可用性——历史记录为不可用，若仍不可用则 §9 的
     profiler 问题标 `not_run`；
  4. 支持几何的上界（§3 取「任何正 `head_dim` + smem 容量校验」）是否被接受。

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
  | `num_q_heads % num_kv_heads != 0` | **未校验** | **新增**：否则 `q_head / group_size` 静默取整，GQA 映射错误 |
  | `head_dim <= 0` | `forwardPaged` 依赖 config，未显式校验 | 新增：`head_dim > 0` 且 `AttentionSmemLayout{head_dim}.total_bytes() <=` device 动态 smem 上限 |
- **Unsupported → fallback**：当几何不被 direct 路径支持（如 `head_dim` 超 smem、
  未来扩展的 `block_size` 集合外取值）时，dispatch 回落到 legacy gather 路径，
  并**必须可观测**：每次回落递增一个计数器并打一条 `TLLM_WARN`（§4.6 要求）。
  benchmark 不得把回落运行计入 direct 数字。
- **OOM**：本 kernel 不分配，无新 OOM 路径。
- **Launch/runtime errors**：沿用仓库现状（kernel launch 后由 `cudaGetLastError` /
  测试侧 `cudaDeviceSynchronize()` 断言捕获）；本设计**不引入**「kernel 内静默 return」
  作为错误处理——参数非法必须由 host 校验拦下，而不是让 kernel 悄悄不做事。
- **Partial success**：不适用（单 sequence、单 token）。
- **Observability**：`TLLM_ERROR`（校验失败）、`TLLM_WARN`（回落）、
  以及 benchmark/结果包中的 `fallback_count`；无静默失败。
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
| `num_q_heads % num_kv_heads != 0` | host 校验 | 返回 err | 否（host） | — |
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
| PR-1 重构（行为不变） | 把 decode 的 tile loop 抽成 `__device__` 模板 + 寻址策略；连续版改用它 | `kernels/attention.cu` | TLLM-P0-002 全部 + 既有 attention 测试**逐元素不变**；sanitizer 0 error |
| PR-2 kernel | 新增 `attention_decode_paged` + 头声明 + kernel 级差分测试 + sanitizer | `kernels/attention.{cu,cuh}`、`tests/**` | §8 的 kernel 级 1/2 层；**不改 FFI、不改 Transformer** |
| PR-3 runtime dispatch | `attentionPaged` decode 分支切 direct，支持几何外回落 legacy + 计数器/日志 | `src/transformer.cpp`、`include/tiny_llm/transformer.h`、`tests/**` | §8 的 layer 级 + §7 校验用例 |
| PR-4 ABI/integration | **计划为空**：本设计不改 C ABI。若实施中发现必须改，则与 `paged-serving` 成对提交并按 §5 的 ABI 包评审 | — | 未发生则不开 PR |
| PR-5 benchmark | 三路 kernel benchmark + 结果归档（须绑定 PR-2/3 的 correctness commit） | `src/kernel_bench.cpp`、`docs/performance/**` | §9；raw data 与 provenance |
| PR-6 docs | 更新能力边界（仅在证据完成后） | `docs/architecture/**`、`CHANGELOG.md` | 只能引用已归档证据 |

- **一个 PR 只做一层**：PR-1 只重构、PR-2 只加 kernel、PR-3 只接线；不得把
  contract + 实现 + benchmark + 结论塞进同一个 PR。benchmark PR 不改 kernel。
- **Shared-file owner**：`kernels/attention.cu` 在 PR-1 与 PR-2 之间**串行**，
  由同一 owner 完成，避免并行改动冲突。
- **跨仓顺序**：本设计不触碰 C ABI，故与 `paged-serving` 无强制顺序；但 PR-3 之后
  的 PSRV-P1-002/004 若要使用 direct 收益，必须引用 PR-5 的结果包。

## 11. Rollback

- **Preserved legacy path**：`scatter → gather → attention_decode` **不删除**，
  至少在 direct 路径经过固定的观察期与结果矩阵（PR-5 完成）之后才讨论删除。
- **Feature flag / fallback**：dispatch 由开关控制，取值 `auto | legacy | direct`，
  默认先 `legacy`，PR-3 合并且 PR-5 通过后改 `auto`；`auto` 在支持几何内走 direct，
  否则回落并计数（§7）。
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

作者自检（**不构成批准**）：

| 门禁 | 作者自检 | 说明 |
|------|----------|------|
| G0 事实与范围 | 自检通过 | §2 绑定 exact commit 与 dirty；显式区分「已实现 / 证据不足 / 未来目标」；明确不把 paged bookkeeping 写成 direct PagedAttention |
| G1 API/ABI | 自检通过，待确认 | §3 冻结内部签名；**唯一开放决策**是 reviewer 是否接受「flat 参数」而非 POD view（§1 rejected alternatives 已给理由） |
| G2 布局与数值 | 自检通过 | §4 给出完整线性地址公式与各 stride；非法块 id 语义与 legacy 显式对齐；整数宽度与上界已说明 |
| G3 所有权与生命周期 | 自检通过 | kernel 零分配、零状态；无 static/raw pointer |
| G4 stream 与并发 | 自检通过 | caller stream、无内部同步、graph 可捕获；不提供 thread_local 包装 |
| G5 错误语义 | 自检通过 | §7 分层校验 + 可观测 fallback；不静默 return |
| G6 correctness | 自检通过 | §8 三层门禁 + sanitizer + 变异检验；oracle 独立 |
| G7 性能 baseline | 自检通过 | §9 baseline 语义等价且有证据；raw data、收敛标准、profiler 问题齐备 |
| G8 合并与回滚 | 自检通过 | §10 六个 PR 分层；§11 flag + trigger + procedure |

**Reviewer**：*待指派（不得由本设计作者担任；按 `NEXT_AGENT_START_HERE.md` §12，
实现 Agent 不应成为唯一 reviewer）*

**Decision**：`pending` —— 尚未批准。

### Reviewer 需要明确回答的问题

1. §3 的 flat 参数签名是否接受？（替代方案：POD view struct）
2. §10 的 PR-1「抽取共享 tile loop」是否接受？（替代方案：复制一份循环）
3. §4.3 冻结的「非法块 id = 零行但参与 softmax」是否接受为**稳定契约**？
   （替代方案：把非法块 id 视为调用方 bug，在 host 侧校验——**会破坏 CUDA Graph
   捕获**，见 §6）
4. §8 要求 direct 与 legacy **逐元素相等**是否接受？（替代方案：容差比较）
5. §7 新增的 `num_q_heads % num_kv_heads != 0` 校验是否同时补到连续版
   `attention_decode`？（当前连续版静默取整，属既有缺口）
6. §9 是否接受「kernel 级收益不得外推为 TTFT/TPOT 改善」的表述边界？
