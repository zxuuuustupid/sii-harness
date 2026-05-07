# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 任务概述

这是一个 **Harness Engineering 考核**，要求设计一个基于 LLM 的文本分类 Harness。

**核心约束**：
- 只能修改 `solution.py` 中的 `MyHarness` 类
- 只能 import Python 标准库 + numpy + harness_base，**禁止 openai/sklearn/torch**
- 禁止读写任何文件
- `call_llm` 单次 prompt 超过 `max_prompt_tokens`（默认 2048）会被截断

## 运行命令

> ⚠️ 必须使用 **sii** conda 环境运行：`conda run -n sii python run.py`
> （sii 环境 transformers 5.8.0 与 tokenizer 兼容；base 环境会报 `'list' object has no attribute 'keys'`）

```bash
# 安装依赖（sii 环境）
conda run -n sii pip install -r requirements.txt

# 本地调试（默认 4 轮取均值）— 由用户在终端运行
conda run -n sii python run.py

# 快速单轮测试
conda run -n sii python run.py --runs 1

# 调整并发数
python run.py --workers 50

# 指定 max_prompt_tokens
python run.py --max-prompt-tokens 2048
```

## 架构说明

```
solution.py          ← 考生唯一要提交的代码，实现 MyHarness
harness_base.py      ← 基类，提供 call_llm / count_tokens / memory 等注入接口
llm_client.py        ← LLM API 配置（顶部三行）；含 call_llm / count_tokens / truncate_to_tokens
run.py               ← 评测脚本：加载 train_dev.jsonl → 调用 update → 调用 predict → 计算准确率
data/                ← DEV 集（231 训练 + 539 验证，77 类客服意图分类）
tokenizer/           ← 本地 tokenizer（token.json），用于精确 token 计数
```

**基类 `Harness` 提供的接口**：

| 属性/方法 | 类型 | 说明 |
|---|---|---|
| `self.call_llm(messages)` | `(list[dict]) -> str` | 调用 LLM，prompt 超限自动截断 |
| `self.count_tokens(text)` | `(str) -> int` | 计算单段文本 token 数 |
| `self.count_messages_tokens(messages)` | `(list[dict]) -> int` | 计算 messages 总 token 数 |
| `self.max_prompt_tokens` | `int` | prompt 上限（默认 2048） |
| `self.memory` | `list[tuple[str, str]]` | 存储 (text, label) 样本 |

**MyHarness 必须实现的接口**：

- `update(text, label)`：接收一条带标签样本，更新内部记忆
- `predict(text) -> str`：对文本预测标签，返回标签字符串（**必须 exact match**）

## 提高准确率的关键

当前 baseline 的 `predict()` 只是简单调用 LLM，没有任何记忆利用。要提升准确率：

1. **利用 `update()` 积累 few-shot 示例**：将训练样本存入 memory，predict 时动态组装到 prompt 中
2. **预算控制**：用 `count_messages_tokens` 预先检查 prompt 长度，超限时优先截断/减少示例
3. **OOD 泛化**：不要针对客服意图分类设计过于特殊化的方案（如传统 ML），评测包含选择题任务
4. **Prompt Injection 安全**：评测中会混入恶意样本，prompt 设计需有防御意识

## 评分标准

- 客观分 80%：私有测试集加权平均准确率（同类分类 + OOD 领域分类 + 选择题）
- 主观分 20%：专家评审 Harness 设计的创新性、合理性