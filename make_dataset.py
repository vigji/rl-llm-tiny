"""Synthetic 2-digit arithmetic prompt set."""
from __future__ import annotations

import random

from datasets import Dataset

SYSTEM = (
    "You are a math assistant. Solve the problem step by step, "
    "then put the final integer answer inside <answer></answer> tags."
)


def make_prompt(a: int, b: int) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Compute {a} + {b}."},
    ]


def build(n: int = 2000, seed: int = 0) -> Dataset:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        rows.append({"prompt": make_prompt(a, b), "answer": a + b})
    return Dataset.from_list(rows)


if __name__ == "__main__":
    ds = build(2000)
    print(ds)
    print("first row:", ds[0])
