# 探索过程日志

## 2026-05-07

---

### 第1次尝试：Baseline（未修改）

`predict()` 内容：
```python
response = self.call_llm([{"role": "user", "content": f"Classify the following text. Respond with only the label, nothing else.\n\nText: {text}\nLabel:"}])
return response.strip()
```

- **问题**：模型输出 `"Statement"`，和 77 类真实标签完全不匹配，0% 准确率
- **原因**：prompt 没有列出 label 集合，模型自由发挥

---

### 第2次尝试：添加 label 列表 + 答案提取

核心修改：
- system prompt 中列出所有 77 个 label
- 5 层 label 匹配（精确 → 大小写 → 前缀 → 包含 → 首行）

`predict()` 部分逻辑：
```python
system = (
    f"You are a text classification assistant. "
    f"Classify the user text into exactly one of the following labels.\n"
    f"Labels: {', '.join(sorted(self._label_set))}\n"
    f"Respond with ONLY the label string, nothing else."
)
```

**问题**：`count_tokens` 在 sii 环境报错 `'list' object has no attribute 'keys'`
**原因**：tokenizer_config.json 中 `extra_special_tokens` 是 list，transformers 5.x 要求 dict

---

### 第3次尝试：去掉 count_tokens，用字符估算

`solution.py` 修改：
```python
def _char_to_tokens(self, text: str) -> int:
    """字符数转 token 粗估（英文约 4 char ≈ 1 token）。"""
    return max(1, len(text) // 4 + len(text.split()) // 2)
```

`_build_prompt()` 中用 `_char_to_tokens()` 替代 `count_tokens()`。

**问题**：Claude Code 调用时报错，用户终端直跑 0%（count_tokens 依然被调用）
**原因**：run.py 中 `make_controlled_llm` 仍然调用 `count_tokens`，且 sii 环境 `count_tokens` 报错会向外传播

---

### 第4次尝试：在 sii 环境直跑（解决 count_tokens 问题）

**关键发现**：sii 环境（transformers 5.8.0）下 `count_tokens` 直接报 `'list' object has no attribute 'keys'`，原因是 tokenizer_config.json 的 extra_special_tokens 格式。但用户终端直跑时，`count_tokens` 没被触发（因为没有调用路径），API 正常工作。

用户终端直跑结果：**75.3% 准确率**（1 run，539条，prompt 1359 token/条，compl 3.9 token/条）

---

### 当前 solution.py 最终代码

```python
class MyHarness(Harness):
    def __init__(self, call_llm, count_tokens, count_messages_tokens, max_prompt_tokens: int):
        super().__init__(call_llm, count_tokens, count_messages_tokens, max_prompt_tokens)
        self._label_set = []   # 动态累积见过的所有 label
        self._examples = []    # 用于 few-shot 的示例

    def update(self, text: str, label: str) -> None:
        super().update(text, label)
        if label not in self._label_set:
            self._label_set.append(label)
        count = sum(1 for t, l in self._examples if l == label)
        if count < 2:
            self._examples.append((text, label))

    def _char_to_tokens(self, text: str) -> int:
        """字符数转 token 粗估（英文约 4 char ≈ 1 token）。"""
        return max(1, len(text) // 4 + len(text.split()) // 2)

    def _build_prompt(self, text: str) -> list[dict]:
        system = (
            f"You are a text classification assistant. "
            f"Classify the user text into exactly one of the following labels.\n"
            f"Labels: {', '.join(sorted(self._label_set))}\n"
            f"Respond with ONLY the label string, nothing else."
        )
        messages = [{"role": "system", "content": system}]

        if self._examples:
            overhead = self._char_to_tokens(system) + self._char_to_tokens(f"\nText: {text}\nLabel: ") + 20
            avail = max(0, self.max_prompt_tokens - overhead)

            for text_ex, label_ex in self._examples:
                shot = f"\nText: {text_ex}\nLabel: {label_ex}"
                shot_t = self._char_to_tokens(shot)
                if avail >= shot_t:
                    messages.append({"role": "user", "content": f"Text: {text_ex}"})
                    messages.append({"role": "assistant", "content": label_ex})
                    avail -= shot_t
                else:
                    break

        messages.append({"role": "user", "content": f"Text: {text}"})
        return messages

    def predict(self, text: str) -> str:
        messages = self._build_prompt(text)
        resp = self.call_llm(messages)

        if not self._label_set:
            return resp.strip()

        resp_lower = resp.strip()

        for lbl in sorted(self._label_set, key=len, reverse=True):
            if resp_lower == lbl:
                return lbl
        for lbl in self._label_set:
            if resp_lower == lbl.lower():
                return lbl
        for lbl in sorted(self._label_set, key=len, reverse=True):
            if resp_lower.startswith(lbl.lower()):
                return lbl
        for lbl in self._label_set:
            if lbl.lower() in resp_lower:
                return lbl
        first = resp.strip().split('\n')[0].strip()
        for lbl in sorted(self._label_set, key=len, reverse=True):
            if lbl.lower() in first.lower():
                return lbl

        return resp.strip()
```

---

### 结果汇总

| 尝试 | 修改内容 | 结果 |
|---|---|---|
| Baseline | 未修改 | 0%（模型输出 "Statement"） |
| 添加 label 列表 | system prompt 列出所有 label | sii 环境 count_tokens 报错 |
| 去掉 count_tokens | 用字符估算替代 | 0%（run.py 仍调用 count_tokens） |
| sii 环境直跑 | 绕开 Claude Code 调用 | **75.3%** |

---

### 下一步优化方向

1. **精确控制 few-shot 数量**：用 count_tokens 替代字符估算
2. **增加示例数量**：当前每类 2 条，可尝试 3-4 条
3. **优化 prompt 模板**：减少模型自由发挥空间
4. **答案后处理增强**：正则提取标签