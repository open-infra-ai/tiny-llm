# Agent Note: pre-commit 门禁真正生效——hooksPath + clang-format-18

Status: implemented

## Problem

`.githooks/pre-commit` 一直存在于仓内但 `core.hooksPath` 从未设置，本地门禁
从未生效；且脚本找不到 `clang-format-18` 时提示后 `exit 0` **静默放行**。
后果已在 2026-09-07-09-10 间实际发生：`src/ffi.cpp` 格式违规溜进 master，
`Format` 任务红 → `build-test` 因 `needs:` 被跳过，两个 commit 从未编译验证。

## Decision

- 本地 `git config core.hooksPath .githooks`，钩子真实生效。
- 安装 `clang-format-18`（18.1.3，与 CI 版本一致）；格式修复以 CI 同版本
  工具产出为准（commit `2b15fb2`，62 个源文件 0 违规）。
- 钩子脚本逻辑不改：缺二进制时仍为跳过而非报错——本地是便利层，
  CI 的 Format 门才是权威。

## Alternatives considered

- **缺二进制时硬失败** — 拦截最强；但非 Ubuntu 环境没有 `clang-format-18`
  包名会卡住所有提交，把环境差异变成提交事故。
- **删掉本地钩子只靠 CI** — 最简单；但红 CI 才发现格式的反馈环太长，
  本次事故正是这么来的。

## Consequences

- **收益**：格式问题在本地提交前被拦截，不再污染 CI 链路。
- **代价**：`core.hooksPath` 是本地配置不进仓库，新克隆环境要记得重设；
  **已知缺口**：仓内无分支保护，CI 红灯不阻塞合并——制度化防线仍缺一层。

## Verification

`git config core.hooksPath` 输出 `.githooks`；`clang-format-18 --version`
报 18.1.3；复盘记录见根 `changelog/2026-09-10-public-surface-fixes.md` §1-2。
