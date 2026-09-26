# Agent Note: paged attention 用独立 oracle + 逐位门禁，不靠容差

Status: implemented

## Problem

分页 KV（策略 1）与连续 KV（策略 2）语义必须等价，但等价性 bug 并不表现为
崩溃：`attn_partial` 分配曾被留着禁用标记，kernel 对 `nullptr` 防御性静默
返回，`wo` 消费陈旧 buffer——C ABI 层面只表现为 logit 漂移
（top-1 0.051 vs 0.009）。5% 容差的冒烟测试吸收了这个差异；比较两条同样
空转路径的测试也全绿。容差型比较在原理上就抓不住这类 bug。

## Decision

paged attention 的正确性门禁建立在**独立参考 + 逐位相等**上：

- `tests/paged_attention_oracle.h`：纯 host 参考实现，冻结 paged KV 地址
  公式、块表长度契约与 GQA 映射，独立计算 fp32 decode attention；
  **不调用**生产 `paged_gather_blocks` / `paged_scatter_blocks` /
  `attention_decode`。
- `tests/test_paged_oracle.cpp`：kernel 级（生产 scatter+gather+attention
  vs oracle）与 layer 级（`forwardPaged` pool 按冻结公式读回 vs 连续 KV
  可见 K/V 逐层位级）差分。
- `tests/test_paged_direct.cpp`：`attention_decode_paged` 与 legacy
  gather+连续路径输出**逐位相同**（12 组几何 × 3 seed），覆盖非法块 id、
  `visible_tokens=0`、块表不足、多 layer offset；另对照独立 oracle。
- `tests/test_ffi_paged_dispatch.cpp`：FFI 级差分——正式抓获
  `attn_partial` 事故的那道门。

## Alternatives considered

- **容差比较（rtol/atol）** — 写起来最省；实测已漏掉「零行参与归一化」
  级别的语义分歧，容差吸收的就是要抓的东西。
- **只跑端到端 GGUF 模型对比** — 最贴近真实；但重依赖、慢，且端到端分歧
  无法定位到地址公式还是 kernel，oracle 把契约钉在可单测的层面。

## Consequences

- **收益**：两条等价路径（含 `decode_online_softmax` 抽取共享循环后的
  ContiguousRows/PagedRows）互为护栏，复制实现的漂移风险由逐位门禁覆盖；
  共享抽取带来的 +1.4~2.3% kernel 回归也因此敢接受。
- **代价**：oracle 本身是第二份实现，地址公式变更要同批改 oracle+生产码；
  逐位等价也意味着有意发散必须先改契约再谈优化。

## Verification

`tests/test_paged_oracle.cpp`、`test_paged_direct.cpp`、
`test_ffi_paged_dispatch.cpp` 在 CI 常驻；`attn_partial` 事故的完整复盘
见 CHANGELOG Unreleased Fixed 段。
