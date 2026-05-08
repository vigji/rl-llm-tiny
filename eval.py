"""Eval base vs LoRA-adapted model on a held-out 200-example arithmetic set.

Run (after training):
    uv run python eval.py
    uv run python eval.py --adapter runs/grpo-qwen-add/final
"""
from __future__ import annotations

import argparse
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_dataset import build
from reward import ANSWER_RE, FORMAT_RE

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def generate_batch(model, tokenizer, prompts, max_new_tokens=128, temperature=0.0):
    texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def score(completions, answers):
    correct = 0
    formatted = 0
    for comp, gold in zip(completions, answers):
        if FORMAT_RE.search(comp):
            formatted += 1
        m = ANSWER_RE.search(comp)
        if m and int(m.group(1)) == int(gold):
            correct += 1
    n = len(completions)
    return correct / n, formatted / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="Path to LoRA adapter dir; omit for base model")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--show", type=int, default=5, help="Print this many example outputs")
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # generation needs left-pad

    print(f"Loading base model ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    if args.adapter:
        print(f"Applying LoRA adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)
    if torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()

    ds = build(args.n, seed=12345)  # different seed than train
    prompts = [r["prompt"] for r in ds]
    answers = [r["answer"] for r in ds]

    completions = []
    for i in range(0, len(prompts), args.batch):
        batch = prompts[i:i + args.batch]
        completions.extend(generate_batch(model, tokenizer, batch))
        if i == 0:
            for j in range(min(args.show, len(batch))):
                print("---")
                print("Q:", batch[j][-1]["content"])
                print("A:", completions[j])
                print(f"(gold={answers[j]})")

    acc, fmt = score(completions, answers)
    label = f"adapter={args.adapter}" if args.adapter else "base"
    print(f"\n[{label}] accuracy: {acc:.3f}  format-rate: {fmt:.3f}  (n={args.n})")


if __name__ == "__main__":
    main()
