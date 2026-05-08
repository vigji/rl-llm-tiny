"""Verification step 1: load model on MPS and run a single generate()."""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_dataset import make_prompt

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    print(f"MPS available: {torch.backends.mps.is_available()}")
    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
    print(f"Dtype: {dtype}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    if torch.backends.mps.is_available():
        model = model.to("mps")
    print(f"Loaded in {time.time() - t0:.1f}s. device={model.device}, dtype={model.dtype}")

    msgs = make_prompt(23, 45)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    dt = time.time() - t1
    gen = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nGeneration ({dt:.1f}s):")
    print("-" * 60)
    print(gen)
    print("-" * 60)
    print("(gold answer: 68)")


if __name__ == "__main__":
    main()
