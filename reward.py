"""Reward functions for the arithmetic task.

TRL GRPOTrainer calls each reward fn with (prompts, completions, **kwargs).
**kwargs contains every other column from the dataset (here: `answer`).
Returns a list[float] with one score per (prompt, completion).
"""
from __future__ import annotations

import re

ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>", re.IGNORECASE | re.DOTALL)
FORMAT_RE = re.compile(r"<answer>.*?</answer>", re.IGNORECASE | re.DOTALL)


def _extract_text(completion) -> str:
    """Completion may be a string or a list[dict] of chat messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return "".join(m.get("content", "") for m in completion)
    return str(completion)


def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    out = []
    for comp, gold in zip(completions, answer):
        text = _extract_text(comp)
        m = ANSWER_RE.search(text)
        if m and int(m.group(1)) == int(gold):
            out.append(1.0)
        else:
            out.append(0.0)
    return out


def format_reward(prompts, completions, **kwargs) -> list[float]:
    out = []
    for comp in completions:
        text = _extract_text(comp)
        out.append(0.1 if FORMAT_RE.search(text) else 0.0)
    return out


if __name__ == "__main__":
    fake_completions = [
        "Let me think... 23 + 45 = 68. <answer>68</answer>",
        "The answer is 70.",
        "<answer>67</answer>",
        "<answer>68</answer> done",
    ]
    fake_answers = [68, 68, 68, 68]
    print("correctness:", correctness_reward(None, fake_completions, fake_answers))
    print("format:    ", format_reward(None, fake_completions))
