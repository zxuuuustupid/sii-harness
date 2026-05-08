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
   会自动截断至预算上限后再发送，
   可用 count_tokens（计算单条消息的 token 数） 和 count_messages_tokens（计算消息列表的总 token 数）预先控制 prompt 长度。
6. predict() 只接收 text，任何绕过接口获取 label 的行为将导致得分归零。
"""

from collections import defaultdict
from harness_base import Harness

# ============================================================
# 考生实现区（考生只能修改 MyHarness 类里的内容）
# ============================================================
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

        if not self._examples:
            messages.append({"role": "user", "content": f"Text: {text}"})
            return messages

        overhead = (
            self.count_tokens(system)
            + self.count_tokens(f"\nText: {text}\nLabel:")
            + 5
        )
        avail = max(0, self.max_prompt_tokens - overhead)

        # 每个 label 取最短样本（短样本优先，多放几条）
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
            else:
                break

        # 如果还有空间，补充第二轮（每个 label 的次短样本）
        if len(self._examples) > len(label_to_examples):
            used_texts = {t for t, l in selected}
            second_round = []
            for t, l in self._examples:
                if t not in used_texts and l in {e[1] for e in selected}:
                    second_round.append((t, l))
            second_round.sort(key=lambda x: len(x[0]))
            for text_ex, label_ex in second_round:
                shot = f"\nText: {text_ex}\nLabel: {label_ex}"
                shot_t = self.count_tokens(shot)
                if total + shot_t <= avail:
                    selected.append((text_ex, label_ex))
                    total += shot_t

        for text_ex, label_ex in selected:
            messages.append({"role": "user", "content": f"Text: {text_ex}"})
            messages.append({"role": "assistant", "content": label_ex})

        messages.append({"role": "user", "content": f"Text: {text}"})
        return messages

    def predict(self, text: str) -> str:
        messages = self._build_prompt(text)
        resp = self.call_llm(messages)

        if not self._label_set:
            return resp.strip()

        resp_stripped = resp.strip()

        # 策略1: 精确匹配
        for lbl in sorted(self._label_set, key=len, reverse=True):
            if resp_stripped == lbl:
                return lbl

        # 策略2: 大小写不敏感精确匹配
        resp_lower = resp_stripped.lower()
        for lbl in self._label_set:
            if resp_lower == lbl.lower():
                return lbl

        # 策略3: resp 以某 label 开头
        for lbl in sorted(self._label_set, key=len, reverse=True):
            if resp_lower.startswith(lbl.lower()):
                return lbl

        # 策略4: resp 包含某 label
        for lbl in sorted(self._label_set, key=len, reverse=True):
            if lbl.lower() in resp_lower:
                return lbl

        # 策略5: 第一行包含某 label
        first = resp.strip().split('\n')[0].strip()
        for lbl in sorted(self._label_set, key=len, reverse=True):
            if lbl.lower() in first.lower():
                return lbl

        return resp_stripped
