# Agent Note: generate 路径自包含，不链接 cuflash

Status: implemented

## Problem

组织内 `cuflash` 有更深的 FlashAttention 实现（WMMA、FlashDecoding）。给
tiny-llm 的 decode 直接接入 cuflash kernel 看似是性能捷径，但这会把旗舰
运行时的正确性审计面耦合到第二个仓库，并打破「一个仓一条责任主线」的
作品集分工。

## Decision

`tiny-llm` 的 generate/FFI 路径保持自包含：attention、KV、采样、量化运行时
都在本仓演进（如 `attention_decode_paged`、`TLLM_PAGED_ATTENTION` 路由）。
`cuflash` 是独立的 kernel 深度作品，**不进入**本仓 generate 路径；
README 的 IN/OUT 边界与工作区 `AGENTS.md` 规则 5 是同一约束。

## Alternatives considered

- **链接 cuflash 作为 attention 后端** — 直接拿到更快的 kernel 最强；但旗舰
  路径的每个正确性声明都要跨仓审计，「哪份代码负责哪条断言」变模糊。
- **把 cuflash kernel 拷进本仓改造** — 无外部依赖；但产生 fork 漂移，两份
  实现各自修复，等价于更糟的耦合。

## Consequences

- **收益**：本仓的正确性与性能声明可独立审计、独立复现；边界清晰。
- **代价**：decode attention 不是组织内最快实现；需要更强 kernel 时只能
  在本仓内演进（如 paged direct kernel），不能走捷径。

## Verification

`git grep -i cuflash` 在 `src/`、`kernels/`、`include/` 无命中；
构建系统（CMakeLists.txt）无 cuflash 依赖。
