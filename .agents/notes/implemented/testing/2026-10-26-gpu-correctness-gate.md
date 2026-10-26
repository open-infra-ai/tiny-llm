# Agent Note: 真实 GPU 正确性门禁把 skip 视为失败

Status: implemented

## Problem

`ctest` 在无模型环境下仍显示 100% 通过，因为 9 个真实 GGUF 用例用 `GTEST_SKIP`
退出；2026-08-22 的 184/184 快照同样含 9 个门控跳过。第二个模型门控
（`TLLM_GGUF_TEST_MODEL_2`）虽已写好，但一直没有模型资产，长期处于“就绪但
未验证”。同时仓库 CI 只有 CPU runner，GPU 用例不可能在托管 CI 执行。缺一个
可重复、能明确失败的本地门禁。

## Decision

新增 `scripts/run-gpu-gate.sh`：

- 强制要求 `TLLM_GGUF_TEST_MODEL` 与 `TLLM_GGUF_TEST_MODEL_2` 同时存在，
  缺任一变量直接失败，不进入“跳过即绿色”的状态；
- 每次先重新 configure + Release 构建，再运行完整 `ctest`；解析 ctest 的
  “did not run” 列表，任何 skip 都非零退出；
- 把日期、commit、被跟踪文件脏计数、GPU/驱动/CUDA、模型大小与 sha256、
  完整构建与测试输出写入 `docs/performance/results/data/<date>-gpu-gate.log`，
  落盘前把仓库绝对路径与 `$HOME` 脱敏；`.gitignore` 只放行该证据目录的日志，
  避免全局 `*.log` 规则让公开报告链接到未入库文件。

第二个模型选 **SmolLM2-360M-Instruct Q8_0**：官方 GGUF 资产、体积小（369 MiB）、
GQA 15→5 与 Qwen 的 14→2 明显不同，且 Q8_0 在 `model_loader` 的支持列表内。

门禁定位为本地/自托管运行，不新增 GitHub Actions job：组织没有自托管 GPU
runner，托管 runner 无 GPU，加 workflow 只会得到永远排队的死配置。

## Alternatives considered

- **继续用裸 `ctest` + 人工看 skip 数**：零成本，但默认输出是“100% passed”，
  极易把未执行当通过，正是本门禁要消除的失败模式。
- **只要求第一个模型**：能消除大部分 skip，但第二模型门控继续空转，无法证明
  GQA 映射不依赖单一配置。
- **选 Llama-3.2-1B**：与 kernel 级 32→8 验证呼应更强，但 HF 资产需登录授权、
  体积约 2 倍；SmolLM2 已能提供不同的真实配置。
- **在 gate 里允许一份 expected-skip 白名单**：能立刻全绿，但白名单会随时间
  腐化，等价于把 skip 重新合法化。
- **新增 `workflow_dispatch` GPU job**：没有 runner 时不可执行，且容易被误读为
  “CI 已覆盖 GPU”，故不加。

## Verification

2026-10-26 在 RTX 3060 Laptop（驱动 610.88、CUDA 12.0.140）执行 gate：

- Qwen2.5-0.5B-Instruct Q4_K_M 与 SmolLM2-360M-Instruct Q8_0 同时加载；
- 重新 configure 后注册 252 项测试（旧 build 目录停留在 198 项），252 项全部
  执行、252 通过、0 失败、0 跳过，ctest 墙钟 224.0 s；
- `SecondModelTest.LoadsAndGeneratesWithDistinctGQA` 单独运行为 PASS（10.5 s，
  配置 960/32 层/15 heads/5 kv heads/49,152 vocab）；
- 原始日志 `docs/performance/results/data/2026-10-26-gpu-gate.log` 无
  `/home/<user>` 路径；
- 证据摘要 `docs/performance/results/2026-10-26-gpu-gate.md` 记录 commit
  `cdff4e6`、模型哈希与限制。

## Consequences

本地有一条可失败的严格门禁，发布前不再需要人工数 skip；代价是每次运行需要
两份模型资产（约 838 MiB，均不入仓）和约 4 分钟构建+测试。门禁不改变远端
CI，也不产生性能或跨架构结论；GPU 机型变化后需重跑并新建日期证据。
