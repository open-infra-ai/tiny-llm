# Agent Note: DPA 计时证据由入仓工具从 raw 重算

Status: implemented

## Problem

split-KV 报告引用未入仓的临时汇总脚本，读者无法复核统计生成方法。raw 中的
path_stats 和人工摘要不能独立证明样本覆盖完整、正确性通过或统计收敛。

## Decision

提供标准库 CLI 读取 DPA v1/v2，从 timed sample 重算分位数、样本标准差 CV 和
repeat median spread。工具验证声明的全部 shape/path/repeat、正确性和聚合记录；
失败时拒绝输出。未收敛 shape 的速度比值为 null，保留计时供诊断。

## Alternatives considered

复制 raw 的 path_stats 即可得到摘要且与原 C++ 数字精确相同，但不能独立复核
样本完整性和统计模型，采用重新聚合并允许输入六位小数造成的舍入误差。

重新生成并覆盖历史 summary 可保持一个 schema，但旧摘要含手工 headline、日期与
profiler 解释，raw 不包含全部来源。选择独立 schema，保持历史结果不变。

## Verification

`python3 -m unittest discover -s tests -p test_dpa_summary.py -v`：12 个测试通过，
覆盖两份真实归档、缺样本、错误输出、重复记录、未知 schema、非法时延、聚合/
repeat 冲突与拒绝覆盖。v2 重算 19,200 timed samples、480 split 等价记录、
32 shape 中 4 个收敛；报告复现命令指向入仓脚本。CI 配置独立 CPU job，无 CUDA
或第三方 Python 运行依赖；远端 CI 未在本轮触发。

## Consequences

计时统计有稳定入仓入口，不依赖临时脚本；代价是维护独立重算 schema。
重算精度受 raw 的六位小数限制，不能还原逐次 kernel 时延；工具验证数据内部一致性，
不证明采集真实性或 profiler 归因，也不构成新的 GPU 实测。
