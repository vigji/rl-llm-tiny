"""SmolLM-135M fallback: GRPO + LoRA on a pure pattern-learning task.

Task: "Write exactly N words about <topic>." Reward = 1 if the model's
output (after the chat prefix) contains exactly N whitespace-separated
words; 0 otherwise. Pure format/pattern reward, no world knowledge needed.

Self-contained: dataset, reward, train, smoke, eval all in this file.

Usage:
    uv run python smollm_run.py smoke      # ~30s: load + generate
    uv run python smollm_run.py sanity     # ~30s: 20 base completions + reward
    uv run python smollm_run.py train      # ~10-20 min: 200 steps
    uv run python smollm_run.py eval       # base vs trained accuracy
    uv run python smollm_run.py eval --adapter runs/smollm-words/final
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
RUN_DIR = Path("runs/smollm-words")

TOPICS = [
    "cats", "rivers", "mountains", "music", "books", "trees", "rain",
    "stars", "bread", "winter", "coffee", "bicycles", "the ocean",
    "old houses", "long walks", "grandparents", "small towns",
    "morning light", "thunderstorms", "open windows", "the sun",
    "gardens", "lighthouses", "wooden boats",
]

SYSTEM = "You are a concise assistant. Follow the user's instructions exactly."


def make_prompt(n: int, topic: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Write exactly {n} words about {topic}. Output only those words, no preface."},
    ]


def build_dataset(n_examples: int = 1000, seed: int = 0) -> Dataset:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_examples):
        n = rng.randint(3, 8)
        topic = rng.choice(TOPICS)
        rows.append({"prompt": make_prompt(n, topic), "target_n": n})
    return Dataset.from_list(rows)


def _extract_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return "".join(m.get("content", "") for m in completion)
    return str(completion)


def word_count(text: str) -> int:
    return len(text.strip().split())


def exact_count_reward(prompts, completions, target_n, **kwargs) -> list[float]:
    out = []
    for comp, n in zip(completions, target_n):
        wc = word_count(_extract_text(comp))
        out.append(1.0 if wc == int(n) else 0.0)
    return out


def proximity_reward(prompts, completions, target_n, **kwargs) -> list[float]:
    """Always-nonzero shaped reward: exp(-|diff|/5). 1.0 at exact, 0.82 at diff=1,
    ~0.05 at diff=15. Provides a smooth gradient toward the target even when
    initial outputs are far off. Capped weight (set in reward_weights) keeps it
    smaller than the exact reward."""
    import math
    out = []
    for comp, n in zip(completions, target_n):
        wc = word_count(_extract_text(comp))
        diff = abs(wc - int(n))
        out.append(math.exp(-diff / 5.0))
    return out


def get_model(dtype, adapter: str | None = None):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    if torch.backends.mps.is_available():
        model = model.to("mps")
    return model, tok


def cmd_smoke(_):
    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
    print(f"Loading {MODEL_ID} ({dtype})...")
    t0 = time.time()
    model, tok = get_model(dtype)
    print(f"Loaded in {time.time() - t0:.1f}s. device={model.device}")
    msgs = make_prompt(5, "cats")
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
    dt = time.time() - t1
    gen = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nGen ({dt:.1f}s):\n{'-'*50}\n{gen}\n{'-'*50}\nword_count={word_count(gen)} (asked for 5)")


def cmd_sanity(_):
    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
    model, tok = get_model(dtype)
    tok.padding_side = "left"
    model.eval()
    ds = build_dataset(20, seed=99)
    prompts = [r["prompt"] for r in ds]
    targets = [r["target_n"] for r in ds]
    texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.9, pad_token_id=tok.pad_token_id)
    gen = out[:, inputs["input_ids"].shape[1]:]
    completions = tok.batch_decode(gen, skip_special_tokens=True)
    exact = exact_count_reward(None, completions, targets)
    prox = proximity_reward(None, completions, targets)
    print(f"\n{'#':>3} {'asked':>5} {'got':>4} {'exact':>5} {'prox':>5}  output (truncated)")
    for i, (n, c, e, p, comp) in enumerate(zip(targets, completions, exact, prox, completions)):
        wc = word_count(c)
        snippet = comp.replace("\n", " ").strip()[:70]
        print(f"{i:>3} {n:>5} {wc:>4} {e:>5.1f} {p:>5.3f}  {snippet}")
    n = len(exact)
    print(f"\nexact: mean={sum(exact)/n:.3f} nonzero={sum(1 for x in exact if x>0)}/{n}")
    print(f"prox:  mean={sum(prox)/n:.3f} std={(sum((x - sum(prox)/n)**2 for x in prox)/n)**0.5:.3f}")


def cmd_train(args):
    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
    print(f"Loading {MODEL_ID} ({dtype})...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    train_ds = build_dataset(args.n_train, seed=0)
    cfg = GRPOConfig(
        output_dir=str(RUN_DIR),
        run_name="smollm-words",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=dtype == torch.bfloat16,
        logging_steps=2,
        save_steps=100,
        save_total_limit=2,
        report_to=["none"],
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        top_p=1.0,
        beta=args.beta,
        num_iterations=1,
        scale_rewards=True,
        reward_weights=[1.0, 0.5],  # exact (sparse 1.0), proximity (smooth max 0.5)
        use_vllm=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[exact_count_reward, proximity_reward],
        args=cfg,
        train_dataset=train_ds,
        processing_class=tok,
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()
    print(f"Starting training: {args.max_steps} steps, batch={args.per_device_batch}, K={args.num_generations}")
    trainer.train()
    final = RUN_DIR / "final"
    trainer.save_model(str(final))
    print(f"Saved to {final}")


def cmd_eval(args):
    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
    print(f"Loading{' base' if not args.adapter else f' adapter={args.adapter}'}...")
    model, tok = get_model(dtype, adapter=args.adapter)
    tok.padding_side = "left"
    model.eval()
    ds = build_dataset(args.n, seed=12345)
    prompts = [r["prompt"] for r in ds]
    targets = [r["target_n"] for r in ds]
    completions = []
    for i in range(0, len(prompts), args.batch):
        batch_p = prompts[i:i + args.batch]
        texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in batch_p]
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = out[:, inputs["input_ids"].shape[1]:]
        completions.extend(tok.batch_decode(gen, skip_special_tokens=True))
        if i == 0:
            for j in range(min(args.show, len(batch_p))):
                print("---")
                print("Q:", batch_p[j][-1]["content"])
                print("A:", completions[j])
                print(f"(target_n={targets[j]}, got={word_count(completions[j])})")
    exact = exact_count_reward(None, completions, targets)
    label = f"adapter={args.adapter}" if args.adapter else "base"
    print(f"\n[{label}] exact-count accuracy: {sum(exact)/len(exact):.3f} (n={args.n})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    sub.add_parser("sanity")
    p_train = sub.add_parser("train")
    p_train.add_argument("--max_steps", type=int, default=200)
    p_train.add_argument("--per_device_batch", type=int, default=16)  # 2 prompts × 8 gens
    p_train.add_argument("--num_generations", type=int, default=8)
    p_train.add_argument("--max_completion_length", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=5e-6)
    p_train.add_argument("--beta", type=float, default=0.04)
    p_train.add_argument("--n_train", type=int, default=2000)
    p_train.add_argument("--lora_r", type=int, default=16)
    p_train.add_argument("--lora_alpha", type=int, default=32)
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--adapter", default=None)
    p_eval.add_argument("--n", type=int, default=200)
    p_eval.add_argument("--batch", type=int, default=8)
    p_eval.add_argument("--show", type=int, default=5)
    args = ap.parse_args()
    {"smoke": cmd_smoke, "sanity": cmd_sanity, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
