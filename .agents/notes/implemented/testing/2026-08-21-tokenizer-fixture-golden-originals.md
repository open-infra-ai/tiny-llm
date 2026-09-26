# Agent Note: tokenizer 差分夹具三件套是冻结的黄金原文

Status: implemented

## Problem

本仓 tokenizer 要与 HuggingFace tokenizers 逐 id 对齐。对齐证据依赖一组
从 HF 侧录制的夹具；如果夹具本身被顺手「整理」（重命名、格式化、重新生成、
修正看似奇怪的 case），外部参照物就失效了——测的是自己和自己的副本。

## Decision

三个文件是**冻结的黄金原文**，列入工作区 `AGENTS.md` 豁免清单：

- `tests/tokenizer_fixture_cases.h`
- `tests/data/tokenizer_fixture.json`
- `scripts/gen_tokenizer_fixture.py`

不改写、不挪位、不就地重新生成。夹具覆盖 byte fallback、跨 token merge、
中文、emoji 等；差分结论（30 例 417 token 逐 id 对齐）绑定这份原文。

## Alternatives considered

- **每次 CI 重新生成夹具** — 数据最新最强；但生成器或上游 tokenizer 变化会
  静默改写「参考答案」，回归测试失去锚点，字节级历史也无法追溯。
- **不用外部夹具，构造 mock token** — 最轻；但没有外部权威参照，无法声明
  与 HF 对齐。

## Consequences

- **收益**：tokenizer 对齐声明有可复算的外部锚点；夹具语义独立于本仓实现演进。
- **代价**：夹具可能显得「丑」（保留原始格式）；新增 case 须新录制而非
  就地修补。

## Verification

差分测试以夹具为输入逐 id 比对 HF tokenizers 输出；夹具三件套在
`git log` 中除生成提交外无改写记录。
