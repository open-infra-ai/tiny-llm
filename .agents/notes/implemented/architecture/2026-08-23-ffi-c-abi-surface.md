# Agent Note: `include/tiny_llm/ffi.h` 是双源 C ABI 的 C++ 侧

Status: implemented

## Problem

paged-serving（Rust 控制面）需要同进程调用本仓 C++ 引擎。跨语言边界若没有
明确的窄契约，结构体布局或缓冲区语义的微小分歧都会变成内存损坏。

## Decision

`include/tiny_llm/ffi.h` 与 `paged-serving/src/tiny_llm_ffi.rs` 构成
**同进程 C ABI 的代码双源**（meta 仓 `docs/cross-repo-contracts.md` §10.1）：

- `TinyLlmConfig` 为 9 个 int 的 repr(C) 布局，Rust 侧 `size_of == 9*4`
  守卫测试锁定一致性。
- KV 生命周期由后端管理（`tinyllm_allocate_sequence` / `tinyllm_free_sequence`），
  调度侧驱动。
- `tinyllm_step`：`next_tokens` 至少容纳 `num_sequences` 个 int；
  `logprobs_k > 0` 时 `logprobs` 缓冲区至少 `num_sequences * logprobs_k * 2`
  个 float，按 `(token_id, logprob)` 交错；`logprobs_k < 0`、超词表、
  或请求输出但缓冲区为空均为参数错误（2026-08-23 澄清）。
- 正常 greedy 路径把各序列末层 hidden 写入 GPU batch buffer，step 末批量执行
  final RMSNorm + LM head + argmax，一次回传整批 token。

## Alternatives considered

- **cxx / bindgen 生成绑定** — 消除手写双源最强；但为 ~9 字段 ABI 引入代码
  生成链不值得，守卫测试已廉价覆盖漂移。
- **更宽的 C++ API 直出** — 表达力更强；但每多一个暴露符号就是一处跨语言
  一致性负担，窄 ABI 是刻意的。

## Consequences

- **收益**：ABI 变化走 breaking-change 流程（契约文档先行、双仓同批、
  两仓 CHANGELOG 各记一条），漂移可测。
- **代价**：签名扩展必须双仓同批；本仓单侧改动会破坏 paged-serving 守卫。

## Verification

`tests/test_ffi.cpp`（策略 1 vs 2 逐 token 差分）、
`tests/test_ffi_paged_dispatch.cpp`；paged-serving 侧布局守卫与
`tiny_llm_text_e2e.rs`。
