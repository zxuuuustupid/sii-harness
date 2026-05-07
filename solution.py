"""
solution.py — 考生唯一需要提交的文件

规则
----
1. 只能修改 MyHarness 类内部；其余部分不可改动。考生可以先行查看 harness_base.py 以了解可用接口和调用约定。
2. 只允许 import Python 标准库（re, math, random, json, collections 等）、numpy
   以及 harness_base（已提供）。
3. 禁止 import 其他第三方库（openai, sklearn, torch …）。
4. 禁止通过任何途径读写磁盘文件。
5. call_llm 每次调用的 prompt token 数若超过 max_prompt_tokens，
   会被自动截断至预算上限后再发送，
   可用 count_tokens（计算单条消息的 token 数） 和 count_messages_tokens（计算消息列表的总 token 数）预先控制 prompt 长度。
6. predict() 只接收 text，任何绕过接口获取 label 的行为将导致得分归零。
"""

from harness_base import Harness

# ============================================================
# 考生实现区（考生只能修改 MyHarness 类里的内容）
# ============================================================
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

        # few-shot budget 控制（不使用 count_tokens，用字符估算）
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

        # 策略1: 精确匹配
        for lbl in sorted(self._label_set, key=len, reverse=True):
            if resp_lower == lbl:
                return lbl

        # 策略2: 大小写不敏感匹配
        for lbl in self._label_set:
            if resp_lower == lbl.lower():
                return lbl

        # 策略3: resp 以某个 label 开头
        for lbl in sorted(self._label_set, key=len, reverse=True):
            if resp_lower.startswith(lbl.lower()):
                return lbl

        # 策略4: resp 包含某个 label
        for lbl in self._label_set:
            if lbl.lower() in resp_lower:
                return lbl

        # 兜底: 找第一行中匹配的 label
        first = resp.strip().split('\n')[0].strip()
        for lbl in sorted(self._label_set, key=len, reverse=True):
            if lbl.lower() in first.lower():
                return lbl

        return resp.strip()