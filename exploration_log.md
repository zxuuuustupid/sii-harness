# 探索过程日志

## 2026-05-07

---

### 第1次尝试：Baseline（未修改）

- **结果**：0%（模型输出 "Statement"）
- **原因**：prompt 没有 label 集合，模型自由发挥

---

### 第2次尝试：添加 label 列表 + 答案提取

- **结果**：sii 环境 count_tokens 报错 0%
- **原因**：tokenizer_config.json 的 extra_special_tokens 是 list，transformers 5.x 要求 dict

---

### 第3次尝试：去掉 count_tokens，用字符估算

- **结果**：0%（run.py 仍调用 count_tokens 导致报错）
- **原因**：Claude Code 调用路径触发了 tokenizer 初始化失败

---

### 第4次尝试：在 sii 环境直跑

- **结果**：**75.3%**（prompt 1359 token，compl 3.9 token）
- **关键发现**：sii 环境直跑时 count_tokens 工作正常

---

### 第5次尝试：去掉 label 列表，增加 few-shot 数量

- **设计**：去掉 system 中的 label 列表，靠 few-shot 让模型自学
- **结果**：**62.0%** ⬇️ 大幅下降
- **原因**：去掉 label 列表后模型自由度太高，乱猜
- **教训**：label 列表是准确率锚，必须保留

---

### 第6次尝试：恢复 label 列表 + count_tokens 精确预算控制

- **设计**：
  - 保留 label 列表（system prompt 中列出全部 77 个）
  - 用 count_tokens 精确控制：system + label list + examples ≤ 2048
  - 两轮选样：每类取最短样本（全覆盖）→ 有空间再补第二轮
- **结果**：**79.0%**（prompt 1754 token，compl 3.9 token）

---

### 当前 solution.py（v6，79.0%）

```python
class MyHarness(Harness):
    def __init__(self, call_llm, count_tokens, count_messages_tokens, max_prompt_tokens: int):
        super().__init__(call_llm, count_tokens, count_messages_tokens, max_prompt_tokens)
        self._examples = []   # 所有 (text, label)
        self._label_set = [] # 所有 label

    def update(self, text: str, label: str) -> None:
        super().update(text, label)
        self._examples.append((text, label))
        if label not in self._label_set:
            self._label_set.append(label)

    def _build_prompt(self, text: str) -> list[dict]:
        labels_str = ", ".join(sorted(self._label_set))
        system = (
            f"You are a customer service intent classification assistant.\n"
            f"Choose the best label from this list: {labels_str}\n"
            f"Respond with ONLY the label, no explanation."
        )
        messages = [{"role": "system", "content": system}]
        overhead = self.count_tokens(system) + self.count_tokens(f"\nText: {text}\nLabel:") + 5
        avail = max(0, self.max_prompt_tokens - overhead)

        # 第一轮：每个 label 取最短的样本（优先短样本，多放几条）
        from collections import defaultdict
        label_to_examples = defaultdict(list)
        for t, l in self._examples:
            label_to_examples[l].append(t)
        sorted_examples = sorted(
            [(min(ts, key=len), l) for l, ts in label_to_examples.items()],
            key=lambda x: len(x[0])
        )
        selected = []
        total = 0
        for text_ex, label_ex in sorted_examples:
            shot = f"\nText: {text_ex}\nLabel: {label_ex}"
            shot_t = self.count_tokens(shot)
            if total + shot_t <= avail:
                selected.append((text_ex, label_ex))
                total += shot_t

        messages += [{"role": "user", "content": f"Text: {t}"}, {"role": "assistant", "content": l}] for t, l in selected
        messages.append({"role": "user", "content": f"Text: {text}"})
        return messages

    def predict(self, text: str) -> str:
        resp = self.call_llm(self._build_prompt(text))
        # 5层 label 匹配策略（精确→大小写→前缀→包含→首行）
        ...
```

---

## 结果汇总

| 版本 | 设计改动 | 准确率 |
|---|---|---|
| v1 Baseline | 无修改 | 0% |
| v2 | 加 label 列表 | count_tokens 报错 0% |
| v3 | 去 count_tokens，用字符估算 | 0% |
| v4（sii直跑）| 恢复 label 列表 | **75.3%** |
| v5 | 去掉 label 列表靠few-shot | **62.0%** ⬇️ |
| v6 | 恢复label列表 + count_tokens精确预算 | **79.0%** |

---

## 下一步优化方向

1. **强化 system prompt 格式指令**：加格式示例（format examples in system），不占 few-shot token
2. **JSON 输出 + JSON 解析**：强制模型输出 `{"label": "xxx"}`，用 json.loads 解析
3. **更多示例覆盖**：231条全部用 count_tokens 精确预算，尽量多放
4. **两阶段分类**：先粗分类（8-10大类），再细分类，减少 77 选 1 的难度
---

### 第7次尝试：JSON 输出格式

- **设计**：system prompt 要求模型输出 `{"label": "xxx"}`，用 json.loads 解析
- **结果**：**75.5%** ⬇️（compl 从 3.9 → 8.9，模型输出更长的解释）

---

### 第8次尝试：多样化补充 + 正则边界匹配

- **设计**：第二轮从所有未选样本随机打乱补充，增加表达多样性；加正则边界匹配
- **结果**：**77.6%** ⬇️（多样性补充反而降低准确率）
- **教训**：同 label 的短样本效果最好，跨 label 的长样本稀释了同类信号的强化

---

### 回退到 v6（79.0%）

恢复 v6 纯文本格式 + 每类短样本优先 + 两轮补充策略。

