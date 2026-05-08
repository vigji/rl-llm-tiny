"""Verification step 2: confirm rewards have variance on un-trained model.

Generates 20 outputs from the base model, scores each with both reward fns,
and prints the distribution. If everything is 0 the reward is too sparse;
if everything is 1 the task is too easy.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_dataset import build
from reward import correctness_reward, format_reward

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N = 20


def main():
    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    if torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()

    ds = build(N, seed=99)
    prompts = [r["prompt"] for r in ds]
    answers = [r["answer"] for r in ds]

    texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    print(f"Generating {N} completions (sampled, T=0.9)...")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    completions = tokenizer.batch_decode(gen, skip_special_tokens=True)

    corr = correctness_reward(prompts, completions, answers)
    fmt = format_reward(prompts, completions)

    print(f"\n{'#':>3}  {'gold':>4}  {'corr':>4}  {'fmt':>4}  output (truncated)")
    for i, (a, c, f, comp) in enumerate(zip(answers, corr, fmt, completions)):
        snippet = comp.replace("\n", " ")[:80]
        print(f"{i:>3}  {a:>4}  {c:>4.1f}  {f:>4.1f}  {snippet}")

    n = len(corr)
    print()
    print(f"correctness: mean={sum(corr)/n:.3f}  nonzero={sum(1 for x in corr if x>0)}/{n}")
    print(f"format:      mean={sum(fmt)/n:.3f}   nonzero={sum(1 for x in fmt if x>0)}/{n}")
    if 0 < sum(corr) < n:
        print("\nOK: correctness reward has variance, RL has signal to learn from.")
    elif sum(corr) == 0:
        print("\nWARN: all correctness rewards are 0. Task may be too hard or reward parser may be wrong.")
    else:
        print("\nWARN: all correctness rewards are 1. Task is trivial; pick something harder.")


if __name__ == "__main__":
    main()
