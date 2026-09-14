# Split-KV decode attention 设计包（TLLM-ATTN-SPLITKV）

> **Status: PROPOSED（2026-09-14）—— 待 review，尚未批准。**
> 本文按 `L3_L4_DESIGN_REVIEW_PACKAGES.md` §3 模板填写。实现按 §10 的 PR-A…PR-D 推进；
> **设计未获批准前，不得修改 production algorithm**（本包即为该批准所要求的载体）。
>
> **独立性缺陷（不得忽略）**：本包由实现者编写。TLLM-P0-004 的
> direct-paged decode attention 设计包 §12（`direct-paged-decode-attention-design.md`，见
> PR #5，尚未合入本分支）已记录同类
> 缺陷，且本任务**改动现有热路径 kernel**（两条生产 attention 路径共用同一份归约循环），
> 与 PR-1 是同一类爆炸半径。**合并前必须由第二方复核 diff 与前后性能数据。**
> 本包先于实现提交，就是为了让该复核有对象。

## 1. Decision summary

- **Chosen design**：把 decode 的**可见 KV 逻辑窗口**切成 `num_splits` 段，每段一个 block
  独立跑现有的 online-softmax 归约，产出局部的 `(m_i, l_i, acc_i)` 三元组；再用第二个轻量
  kernel 把这 `num_splits` 组三元组按 flash-decoding 的标准公式合并成最终输出。
  - `partial kernel`：`grid = (num_q_heads, num_splits)`，`block = 128`（今天的 block 尺寸不变）；
  - `combine kernel`：`grid = (num_q_heads, 1)`，`block = 128`。
  - 段范围**在 device 端**由 `*device_visible_tokens` 派生（`chunk = ceil(visible / num_splits)`，
    `base = blockIdx.y * chunk`），`num_splits` 是 **host 参数**。因此 grid 在捕获时固定、
    可见长度仍可随 replay 变化 —— **不破坏 CUDA Graph 可捕获性**（本任务最硬的约束）。
  - 归约循环**不复制**：把现有的 `decode_online_softmax` 泛化为
    `decode_online_softmax_range<Rows, kFinalize>`，连续 KV 与分页 KV 继续共用同一份实现。

- **Why**：
  1. 现有 kernel 的并行度只有 `num_q_heads` 这一个轴，实测**落在 8.33% occupancy**
     （§2 证据），而 `sm__throughput` 仅 0.72%、DRAM 仅 0.44% —— **没有任何资源饱和**，
     瓶颈是"没有足够多的 warp 去隐藏访存延迟"；
  2. 单 query decode 唯一可扩展的并行轴就是 KV 窗口，因此必须跨 block 切分；
  3. 切分点选在**逻辑 token** 上，所以既有的地址公式、`ContiguousRows` / `PagedRows`
     取址策略、非法块 id 的零行语义**全部不需要改**；
  4. `num_splits = 1` 时合并式退化为恒等（`exp(0) == 1.0f`，且沿用同一个 `1/(l + 1e-9)`），
     输出与今天**逐位相同** —— 这给了一个比容差强得多的回归锚点（§4.4）。

- **Rejected alternatives**：
  - *只重写归约第 4 步的 V 访问*（作者在 PR-5 收尾时的初步判断）：**治标**。第 4 步的低效是
    "只有 4 个 warp、无法隐藏延迟"的症状，不是原因；grid 仍是 14 个 block，SM 仍空闲。
    该判断已被 §2 的 profiler 数据推翻，记录在此以免后人重走。
  - *在 block 内加线程、不跨 block 切分*：`grid = 14` 意味着约 70 个 SM 里有 56 个完全没有 block，
    加大 block 只会让少数 SM 更忙，**无法提升 occupancy**。
  - *单 kernel + 原子/`threadfence` 信号量做 split-K 归约*：省一次 launch，但需要一个全局同步
    协议、更难审计，且在 CUDA Graph 下 launch 开销本就已被摊销 —— 复杂度换不到收益。
  - *kernel 内部自行分配 partial 缓冲（`static` / `thread_local`）*：违反 G3
    （kernel 零分配、零状态）；`cuflash` 的 `src/forward/flash_decoding.cu` 函数级 static scratch
    正是该仓最高风险项，不在此处重演。
  - *把 partial 塞进 C ABI 或 `paged-serving` 契约*：本设计完全落在内部 kernel 头，**ABI 不变**。

- **Explicit non-goals**：
  - 不改 **prefill**（`attention_prefill` 的并行轴是 query 位置，问题不同，见 §10 非目标）；
  - 不做 batched / ragged decode（仍 `batch = 1`，per-sequence 循环留在上层）；
  - **不改 C ABI**（`include/tiny_llm/ffi.h`）与 `paged-serving` 侧契约；
  - 不引入 FP32/BF16 pool；不输出 logsumexp；
  - **不移除** legacy `scatter → gather → attention_decode` 路径（保留为 fallback）；
  - 不产生任何 TTFT/TPOT 或端到端 serving 声明（kernel 级证据，G7）。

## 2. Base evidence

- **Repository**：`open-infra-ai/tiny-llm`
- **Base branch**：`tllm-dpa-pr5-benchmark`（= PR #10 head，堆叠链 #4 → #7 → #9 → #10）
- **Commit**：`7df9ec28d1b4dacceb68139965237302b05ada94`
- **Dirty state**：干净（仅新增本设计文档的分支）
- **Existing code anchors**：

  | 位置 | 事实 |
  |------|------|
  | `kernels/attention.cu:130` `decode_online_softmax` | 两条 decode 路径**共用**的 online-softmax 归约循环（PR-2 抽取，见 issue #8） |
  | `kernels/attention.cu:115` `ContiguousRows` / `:283` `PagedRows` | 取址策略；无效行返回**共享内存零行**，循环内无有效性分支 |
  | `kernels/attention.cu:200` `attention_decode_kernel` + `:245` launch | grid = `num_q_heads`，block = 128，动态 smem 928 B，39 registers/thread |
  | `kernels/attention.cu:322` `attention_decode_paged_kernel` + `:396` launch | 同上，另加 `row_elem[ATTEND_TILE]` + 零行 |
  | `kernels/attention.cu:34` `ATTEND_TILE = 128` | tile 粒度 |
  | `kernels/attention.cu:41` `AttentionSmemLayout` | smem = `(ATTEND_TILE+8+head_dim)*4 + head_dim*2` |
  | `src/transformer.cpp:388` / `:477` / `:489` | 三处生产调用点（连续 prefill/decode、direct decode、legacy decode） |
  | `include/tiny_llm/transformer.h:15` `LayerWorkspace` | 跨层复用的共享工作区（层间串行，故可复用） |

- **Existing tests**：
  - `tests/test_paged_direct.cpp`：direct vs legacy **逐位相同** + 独立 oracle（12 几何 × 3 seed），
    含非法块 id、`visible = 0`、短块表、多 layer offset；
  - `tests/test_paged_oracle.cpp`：kernel 级 / layer 级差分，oracle 不复用生产寻址；
  - `tests/test_paged_dispatch.cpp`：开关三态 + 用 scratch 哨兵**直接观测走了哪条路径**；
  - `tests/test_kernels.cu`、`tests/test_transformer.cu`：连续 attention 的既有覆盖。
- **Existing GPU/performance evidence**：
  - `docs/performance/results/2026-09-14-rtx5070ti-dpa.md`（PR-5）：`contiguous` 在 S=2048 为
    **0.1434 ms**，且 `legacy ≈ gather_k + gather_v + contiguous`（差 3–9%），两条 gather 仅占
    legacy 的 8.3% —— 即**收益天花板由 attention kernel 本身决定**；
  - 同机 ncu（2026-09-14，`visible=2048 / block_size=16`，单次 launch）：

    | kernel | 时长 | occupancy | SM | DRAM | L2 | L1 | long-scoreboard | barrier |
    |--------|------|-----------|----|------|----|----|------------------|---------|
    | `attention_decode_kernel` | 282.3 µs | **8.33%** | 0.72% | 0.44% | 0.96% | 4.00% | 45.8% | **38.6%** |
    | `attention_decode_paged_kernel` | 316.0 µs | — | 0.86% | 1.16 MB | 7.88 MB | 62.4 MB | 40.2% | — |
    | `paged_gather_blocks_kernel` | 4.42 µs | — | 26.6% | — | — | — | 66.7% | — |

    结论：**延迟受限、无资源饱和、并行度不足**，且约 39% 的 issue 槽位丢在 tile 之间的
    `__syncthreads()` 上。
- **Unknown / 必须在实现前关闭**（或由 reviewer 接受为受限范围）：
  1. `num_splits` 的最优取值（倾向 4/8/16 中实测选一，不预判）；
  2. `num_splits > 1` 是否在小 S 上引入可测回归（额外 block + 一次 combine launch）；
  3. combine 的额外 launch 在**非 graph** 路径上的开销占比（graph 下预期可忽略）。

## 3. API / ABI

- **Public or internal**：**internal**。声明在 `kernels/attention.cuh`，与现有
  `attention_decode` / `attention_decode_paged` 同层；`kernels/` 不依赖 `include/tiny_llm/`，
  不进入 C ABI，不进入 `paged-serving`。
- **新增签名（冻结草案）**：

```cpp
// split-KV decode：可见窗口切 num_splits 段，partial 写 partial_workspace，再由 combine kernel 合并。
// partial_workspace 布局见 §4.3；调用方负责分配（零分配 kernel，G3）。
void attention_decode_splitkv(const half *query, const half *k_cache, const half *v_cache,
                              half *output, float scale, int num_q_heads, int num_kv_heads,
                              const int *device_visible_len, int head_dim,
                              float *partial_workspace, int num_splits, cudaStream_t stream = 0);

void attention_decode_paged_splitkv(const half *query, const half *k_pool_layer,
                                    const half *v_pool_layer, const int *block_table,
                                    half *output, float scale, int num_q_heads, int num_kv_heads,
                                    int head_dim, const int *device_visible_tokens, int block_size,
                                    int max_num_blocks, int table_len, float *partial_workspace,
                                    int num_splits, cudaStream_t stream = 0);
```

- **既有签名保持不变（刻意的）**：`attention_decode`（两个重载）与 `attention_decode_paged`
  继续存在，语义等价于 `num_splits = 1`，即**今天的行为**。理由：
  1. 现有 **~18 个调用点**（`src/transformer.cpp`、`src/kernel_bench.cpp`、以及 4 个测试文件）
     无需改动，diff 面收敛；
  2. 单遍入口成为 §8 的**活体回归锚点**，而不是历史产物；
  3. 两个新入口与旧入口调用**同一个** `decode_online_softmax_range`，不存在第二份归约实现。
- **Parameter order and types**：新参数 `float *partial_workspace, int num_splits` 追加在
  `head_dim` / device int **之后**、`stream` 之前，使新旧签名在 diff 中可直接对照。
  整数一律 `int`（与既有 kernel 一致），指针运算一律 `size_t`（见 §4）。
- **Shape/dtype/stride/device**：partial 缓冲为 **device fp32**，布局见 §4.3；Q/K/V/O 的
  dtype、stride、device 语义与现有 kernel 完全一致（FP16 pool + FP32 累加 + FP16 输出）。
- **Size/alignment/field order**：`sizeof(float)` 对齐即可；布局是连续的
  `[head][split][2 + head_dim]`，无 padding、无结构体（沿用 flat 参数约定）。
- **Compatibility/versioning**：无 versioning 需求。**C ABI 未变**，故与 `paged-serving`
  无强制联合顺序（对比设计包 §10 的 PR-4/A）。唯一调用方仍是 `src/transformer.cpp`。

## 4. Data layout and numerics

### 4.1 Logical shape 与段范围（冻结）

```
visible = *device_visible_tokens                     // 仍由 device int 提供（graph 重放前置条件）
chunk   = ceil(visible / num_splits)                 // 在 kernel 内计算
base_s  = blockIdx.y * chunk                         // 第 s 段的逻辑起点
count_s = max(0, min(chunk, visible - base_s))        // base_s >= visible 时为 0（空段）
loop    = for (tile = base_s; tile < base_s + count_s; tile += ATTEND_TILE)
```

段边界落在**逻辑 token** 上（不是物理块、也不要求 tile 对齐），因此 `PagedRows::begin_tile(tile, …)`
仍按 `b = tile / block_size; r = tile % block_size; p = block_table[b]` 寻址，**地址公式不变**。

### 4.2 归约与累加

- 归约循环与今天**同一份**（`decode_online_softmax_range`），仅新增 `begin`/`end` 与一个
  编译期 `kFinalize`：
  - `kFinalize = true` → 归一化后写 `output`（今天的行为逐字保留）；
  - `kFinalize = false` → 不归一化，写 `(m, l, acc[head_dim])` 到 partial。
- **Accumulation dtype**：`m`、`l`、`acc` 全部 **fp32**；输出 fp16。与今天一致。

### 4.3 Partial 布局（冻结）

连续 fp32 数组，`(2 + head_dim)` 个元素一组，同一 head 的 `num_splits` 组相邻（便于 combine
按 head 顺序读）：

```
offset(head, split) = (head * num_splits + split) * (2 + head_dim)
  [0]          = m        (该段最大 score；空段为 -INFINITY)
  [1]          = l        (该段 Σ exp；空段为 0)
  [2 .. 1+D]   = acc[d]   (该段 Σ exp * v；空段为 0)

total floats = num_q_heads * num_splits * (2 + head_dim)
Qwen2.5-0.5B / num_splits=8：14 * 8 * 66 = 7392 floats ≈ 29.6 KB
```

**每个 block 都必须写自己的槽位**（含空段），否则 combine 会读到未初始化内存。

### 4.4 Combine 公式与 `num_splits = 1` 的逐位锚点

```
M   = max_i m_i
L   = Σ_i l_i * exp(m_i - M)
ACC = Σ_i acc_i * exp(m_i - M)
out[d] = __float2half(ACC[d] * 1/(L + 1e-9))        // +1e-9 与乘法形式与今天相同
```

- **空段**贡献 `l=0`、`acc=0`，且 `exp(-INF - M) = 0`（M 有限时），故为中性；
- **全空**（`visible = 0`）必须特判：`M == -INFINITY` 时直接写 0。否则 `exp(-INF - -INF)`
  是 NaN。这保持了 `test_paged_direct.cpp::EmptyVisibleWindowMatchesLegacy` 冻结的
  「`visible = 0` → 全 0 输出」语义；
- **`num_splits = 1` 时**：`M = m_0`、`exp(0) == 1.0f` ⇒ `L = l_0`、`ACC = acc_0`，
  输出为 `acc_0 * 1/(l_0 + 1e-9)` —— 与今天单遍路径**逐位相同**。
  这就是 §8 的第一条门禁，也是本改动唯一能隔离"重构 vs 数值"的手段。

### 4.5 Boundary 与 overflow

- `num_splits >= 1`；`num_splits = 1` 为恒等路径；
- `chunk` 用 `int` 计算，`visible <= max_visible_tokens` 由上层保证；`base_s + count_s <= visible`，
  循环不会越界；
- 指针运算沿用既有 `size_t`（`(p * block_size + r) * kv_dim`），**本设计不新增任何溢出面**；
- 非法块 id / `b >= table_len` 的零行语义**不变**（`PagedRows` 未改）。

## 5. Ownership and lifecycle

- **Allocator**：**host 侧**，`LayerWorkspace`（`include/tiny_llm/transformer.h:15`）。kernel 本身
  **零分配、零状态**（G3）。
- **Owner**：`LayerWorkspace`（每实例一个 `float *attn_partial`）；
- **Reuse scope**：所有 `TransformerLayer` 复用同一份（层间串行，今天所有 workspace 缓冲同理）；
- **Reallocation**：跟随既有 `LayerWorkspace::allocate(config)` / `free()` 语义；缓冲大小只依赖
  `num_q_heads`、`num_splits`、`head_dim`（都由 config + 常数决定），**不依赖运行时序列长度**；
- **Destruction**：随 `LayerWorkspace::free()`（析构自动调用），与既有指针同一路径；
- **Success / cancel / error cleanup**：不新增任何需要清理的中间态；缓冲常驻，无 per-call 分配；
  失败路径与今天一致（`Result<void>::err`，不半写状态）。

## 6. Stream and concurrency

- **Caller stream**：两个 kernel 都在调用方传入的 stream 上，顺序 partial → combine；**无内部
  stream / event / 同步**，保持 CUDA Graph 可捕获（与 `attention_decode` 一致）；
- **Supported concurrency**：同一 `LayerWorkspace` 不可被多 stream 并发使用 —— 这是**既有**约束
  （所有 workspace 缓冲都是如此），本设计不引入新的不安全组合；
- **Synchronization**：无内部同步；
- **CUDA Graph 前置条件**（必须全部成立）：
  1. grid 在捕获时固定：`num_splits` 是 host 常量，捕获后 replay 复用同一 grid；
  2. 可见长度仍走 **device int**，replay 时只更新该值；
  3. partial 缓冲**预先分配**，捕获期间零分配、零 `cudaMalloc`、零 D2H；
- **Unsafe combinations**：与今天相同 —— 跨 stream 共享同一 workspace、或捕获期间改变
  `num_splits`（后者会导致 grid 与缓冲大小不匹配，由调用方保证）。

## 7. Error and fallback

- **Validation errors**：`num_splits < 1`、指针为 `nullptr` → 与既有 kernel 一致地**静默 no-op**
  （廉价防御性检查，真正的错误契约在 host 侧）。host 侧 `TLLM_ATTN_SPLITKV` 的**非法取值显式
  返回错误**，不静默回退（G5，与 `TLLM_PAGED_ATTENTION` 同一处理）；
- **OOM**：缓冲约 29.6 KB，随 `LayerWorkspace::allocate` 走既有分配路径；分配失败按既有
  `LayerWorkspace` 错误处理（不新增 API）；
- **Launch/runtime errors**：不新增错误契约；不引入需要检查的异步错误点；
- **Partial success**：不适用 —— 两个 kernel 同 stream 顺序执行，combine 在 partial 之后；
- **Fallback**：`TLLM_ATTN_SPLITKV=0`（或 `num_splits = 1`）→ 走**逐位相同**的旧路径；
- **Observability**：开关状态与 `num_splits` 进入 benchmark 的 provenance 记录，使结果包能区分
  实际路径（与 PR-3 的 `TLLM_WARN` 同一目的）。

## 8. Correctness matrix

独立参考 = `tests/paged_attention_oracle.h`（TLLM-P0-002，不调用任何生产 kernel）。

| Case | Reference | Expected | GPU required | Sanitizer |
|------|-----------|----------|--------------|-----------|
| **`num_splits = 1`（锚点）** | 改动前的单遍输出，10 组几何的 fp16 位模式指纹 | **逐位相同**（连续 + 分页各一次） | 是 | 是 |
| `num_splits ∈ {2,4,8,16}` | 独立 oracle（容差 2e-3） | 容差内 | 是 | 是 |
| split 连续 vs split 分页 | 彼此 | **逐位相同**（同一归约循环） | 是 | 是 |
| `visible = 0` | 全 0 | 逐位相同（含 combine 的 `M == -INF` 分支） | 是 | 是 |
| `visible < num_splits`（多空段） | 独立 oracle | 容差内，**不得 NaN** | 是 | 是 |
| `visible` 非 `num_splits` 整数倍 | 独立 oracle | 容差内（尾段 `count` 截断正确） | 是 | 是 |
| block_size = 1 / 16 / 32；跨块尾部 | oracle + legacy | 容差内 | 是 | 是 |
| MHA(4/4) / GQA(8/2) / MQA(8/1)；head_dim 32/64/128 | oracle | 容差内 | 是 | 是 |
| 非连续 / 随机物理块表 | oracle | 容差内 | 是 | 是 |
| 非法块 id（负值、`== max_num_blocks`） | oracle（零行语义） | 容差内，不 fault | 是 | 是 |
| `table_len` 不足 | oracle（零行语义） | 容差内，不越界读块表 | 是 | 是 |
| 多 layer pool offset（2–3 层） | 连续 cache | 逐层一致 | 是 | 是 |
| **CUDA Graph：捕获后可见长度增长并 replay** | 非 graph 同输入 | 逐位相同 | 是 | 是 |
| non-default stream | default stream | 一致 | 是 | 是 |

- **变异检验**（沿用 TLLM-P0-002/004 的方法，至少 4 项）：
  1. combine 丢掉 `l_i` 权重（等权求和）→ oracle 捕获；
  2. 段范围 off-by-one（`count_s` 少 1 / `base_s` 偏移）→ 边界用例 + oracle 捕获；
  3. 空段写垃圾而非中性值 → `visible < num_splits` 用例出现 NaN/偏差 → 捕获；
  4. combine 的 `M` 不取 max（如取 `m_0`）→ 大动态范围 score 的用例捕获；
  5. **删掉在线 rescale**（两条路径同等出错）→ 逐位门禁通过、**独立 oracle 捕获**
     （与 PR-2 第三项同一用意，验证分层门禁的必要性）。
- **CPU/build 与 GPU correctness 分开报告**：host 校验类用例无 GPU 也运行；kernel 用例在无
  device 时 **skip 且计数可见**，不得把 skip 记为 pass。
- **Sanitizer**：`compute-sanitizer --tool memcheck` 跑新增用例，要求 **0 error**；并且对
  **benchmark harness 自身**（新增的 workspace 分配/传入路径）也跑一遍。

## 9. Benchmark plan

- **Measurement layer**：**kernel 级**，扩展现有 `tiny_llm_kernel_bench --dpa-bench`（PR-5 已建立
  三路对比与 raw JSONL）。**不写 TTFT/TPOT**。
- **Baseline（语义等价，允许算 speedup，但必须给出等价证据 = §8 的锚点 + oracle）**：
  1. 旧单遍：`legacy`（gather×2 + `attention_decode`）、`contiguous`、`direct` —— 同 PR-5；
  2. 新 split：`contiguous-splitkv`、`direct-splitkv`、以及 `legacy-splitkv`（gather×2 + split）；
  3. 同进程新旧交替（order-balanced），消除跨二进制代码布局混淆（§10.1 教训）。
- **Shapes/workload**：沿用 PR-5 的 32 个 shape（Hq=14, Hkv=2, D=64；
  `visible ∈ {8,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,512,1024,2048}` ×
  `block_size ∈ {16,32}`），另加 `num_splits ∈ {2,4,8,16}` 的扫描以定最优值。
- **Warmup/repetitions**：沿用 PR-5 实测有效的配置（每样本 100 次调用均值、每 repeat 1000 次、
  3 repeat、4 s 时钟预热）；理由与反例见 PR-5 报告 §1.1。
- **Raw artifact**：JSONL（provenance / equivalence / sample / path_stats / shape_summary），记录
  GPU / driver / CUDA / `CMAKE_BUILD_TYPE` / `-arch=native` / commit + dirty / seed / warmup / reps /
  `num_splits`；`CV > 10%` 或重复间中位数偏差 `> 10%` 标 `not_converged`，**不删除**。
- **Convergence**：沿用 PR-5 的门禁（只约束被比较的路径；gather 类辅助诊断单独报告 CV）。
- **Profiler question**：`num_splits > 1` 后 **occupancy 是否真的从 8.33% 抬起来**、
  barrier stall 是否从 38.6% 显著下降 —— 这是"改动是否命中了诊断"的判定，不只是速度。
  若 ncu 不可用则标 `not_run`（本机 2026-09-14 实测可用）。
- **可证伪假设（不得写成结论）**：`num_splits > 1` 的收益在小 S 会被"额外 block + 一次 combine
  launch"吃掉，交叉点预计在 `visible ≈ 128–512`（与 PR-5 观察到的 launch 主导区一致）；
  且 combine 引入的 fp32 重排可能使**超长 S 的误差略微增大**（仍在 oracle 容差内）。
- **G7 拒绝条件对照**：correctness 先于计时；多次运行；有 raw data；CV 超限标注；
  不把 kernel 结果写成 TTFT/TPOT。

## 10. PR and ownership plan

| PR | 内容 | 允许文件 | 门禁 |
|----|------|----------|------|
| **PR-A 设计（本包）** | 诊断证据 + 设计 + 数值策略 + 验收矩阵；**不改任何代码** | `docs/architecture/**`、`docs/.vitepress/config.mts` | 本包获 reviewer 批准 |
| **PR-B kernel** | `decode_online_softmax_range<Rows, kFinalize>` + partial/combine kernel + 两个 `_splitkv` 入口点 + 头声明 + kernel 级测试 | `kernels/attention.{cu,cuh}`、`tests/**` | §8 的锚点 + oracle + 边界 + 图重放；sanitizer 0 error；**既有入口行为不变** |
| **PR-C 接线** | `LayerWorkspace` 分配 partial + `TLLM_ATTN_SPLITKV` 开关 + `transformer.cpp` 路由（direct 与 legacy decode 两处） | `include/tiny_llm/transformer.h`、`src/transformer.cpp`、`tests/**` | 全量测试不减；graph 捕获/重放一致；非法开关值显式报错 |
| **PR-D benchmark** | 扩 `--dpa-bench` 测新旧对比、扫描 `num_splits`、归档 raw + 结果报告 | `src/kernel_bench.cpp`、`docs/performance/**` | §9；只有此时才讨论开关默认值 |
| **PR-E docs**（如需） | 能力边界与 `CHANGELOG` | `docs/architecture/**`、`CHANGELOG.md` | 只能引用已归档证据 |

- **一个 PR 只做一层**：不得把 contract + 实现 + benchmark + 结论塞进同一个 PR；benchmark PR 不改 kernel。
- **Shared-file owner**：`kernels/attention.{cu,cuh}` 由**同一 owner 串行**完成 PR-B，避免与
  PR-2/PR-3 并行冲突；`src/transformer.cpp` 在 PR-C 单独改动。
- **跨仓顺序**：不改 C ABI，故与 `paged-serving` 无强制顺序；但若 `paged-serving` 后续要引用
  收益，必须引用 PR-D 的结果包。

## 11. Rollback

- **Preserved legacy path**：单遍 `decode_online_softmax_range<…, kFinalize=true>` 与
  `scatter → gather → attention_decode` **都不删除**，至少在 split-KV 经过结果矩阵（PR-D）之前。
- **Feature flag / fallback**：`TLLM_ATTN_SPLITKV`（`0 | 1`，或直接给 `num_splits ≥ 1`），
  **默认关闭 = `num_splits = 1` = 逐位等于今天**。默认值只在 PR-D 通过后才考虑打开。
  不做进程级缓存（使 `setenv` 在测试中即时生效，避免为测试暴露 reset seam，同 PR-3 的决定）。
- **Trigger（任一命中即回滚到单遍）**：
  1. `num_splits = 1` 的逐位锚点失败；
  2. Compute Sanitizer 报错；
  3. `num_splits > 1` 超出 oracle 容差，或 `visible = 0` / 空段出现 NaN；
  4. 真实模型 canary（`TLLM_GGUF_TEST_MODEL`）token 不一致；
  5. 收敛 shape 上出现可复现回归而 `num_splits` 无法规避。
- **Procedure**：把 `TLLM_ATTN_SPLITKV` 关掉（单点改动）；保留 split-KV 代码与其失败用例；
  在中性提交上复现问题后再决定修复或撤销。
- **Artifact 保留**：失败 shape、`not_converged`、退化配置的选择结果全部保留在
  `docs/performance/results/`，不删除。
- **G8 拒绝条件对照**：PR 已拆分；有逐位等价的 legacy fallback；benchmark Agent 不优化算法；
  不改 C ABI，不会单侧破坏 `paged-serving`。

## 12. Approval

### 门禁自检

| 门禁 | 自检 | 说明 |
|------|------|------|
| G0 事实与范围 | 通过 | §2 绑定 exact commit 与 dirty；「已实现 / 证据不足 / 未来目标」分开；未把 occupancy 诊断写成已知结论之外的推断 |
| G1 API/ABI | 通过 | §3 冻结内部签名；**C ABI 不变**；既有签名刻意保留并说明理由 |
| G2 布局与数值 | 通过 | §4 给出段范围公式、partial 布局、combine 式与 `num_splits = 1` 的逐位论证；空段/全空语义冻结 |
| G3 所有权与生命周期 | 通过 | kernel 零分配零状态；缓冲由 `LayerWorkspace` 持有，复用与释放走既有路径 |
| G4 stream 与并发 | 通过 | caller stream、无内部同步；列出 graph 捕获的三个前置条件与不安全组合 |
| G5 错误语义 | 通过 | 开关非法取值显式报错；kernel 侧沿用廉价防御检查；无静默 return |
| G6 correctness | 通过 | §8 含**逐位锚点**（比容差强）、独立 oracle、空段/边界/graph，以及 5 项变异检验 |
| G7 性能 baseline | 通过 | §9 baseline 语义等价且有 §8 证据；raw data、收敛标准、profiler 问题与可证伪假设齐备 |
| G8 合并与回滚 | 通过 | §10 五个 PR 分层；§11 flag + 触发条件 + 程序；legacy 单遍路径保留 |

### 决议（待 reviewer 填写）

| # | 议题 | 候选决议 |
|---|------|----------|
| Q1 | combine 用独立 kernel（本设计）还是单 kernel 信号量归约 | 建议**独立 kernel**：graph 下 launch 已摊销，复杂度更低 |
| Q2 | `num_splits` 是 host 参数（本设计）还是编译期常数 | 建议 **host 参数**：捕获时固定即可，且允许调用方按上下文长度选择 |
| Q3 | `num_splits > 1` 破坏「contiguous 逐位不变」门禁是否可接受 | **需要明确批准**：这是本任务唯一有真实爆炸半径的数值决定；建议接受，因为 §8 的 `num_splits = 1` 锚点覆盖了重构风险本身 |
| Q4 | partial 缓冲放 `LayerWorkspace`（本设计）还是复用 direct 路径已冗余的 `k_scratch` | 建议**新缓冲**：两者 dtype/语义不同，复用会把耦合藏起来；与设计包 §5 的 scratch follow-up 一并再议 |
| Q5 | 小 S 是否允许回落到 `num_splits = 1` | 建议**允许由调用方选择**，不做内核内分支（内分支不可测） |

- **Reviewer**：仓库 owner（本轮将决议委托给作者的分析与证据）
- **Decision**：`changes_requested`（未批准前不得进入 PR-B 改 kernel）
