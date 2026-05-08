"""GRPO + LoRA fine-tune of Qwen2.5-0.5B-Instruct on synthetic 2-digit addition.

Run:
    uv run python train.py
or with overrides:
    uv run python train.py --max_steps 50

Logs to ./runs/<run_name>/ and prints generations periodically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from make_dataset import build
from reward import correctness_reward, format_reward

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--per_device_batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--max_completion_length", type=int, default=128)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--run_name", default="grpo-qwen-add")
    args = ap.parse_args()

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device_dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32

    print(f"Loading {MODEL_ID} (dtype={device_dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=device_dtype,
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    print("Building dataset...")
    train_ds = build(args.n_train, seed=0)

    cfg = GRPOConfig(
        output_dir=str(out_dir),
        run_name=args.run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        warmup_steps=10,
        max_grad_norm=1.0,
        bf16=device_dtype == torch.bfloat16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=["none"],  # set to ["wandb"] if you have wandb configured
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.9,
        top_p=1.0,
        beta=args.beta,
        num_iterations=1,
        scale_rewards=True,
        reward_weights=[1.0, 1.0],  # correctness, format
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward],
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )

    trainer.model.print_trainable_parameters()
    print("Starting training...")
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    print(f"Saved to {out_dir / 'final'}")


if __name__ == "__main__":
    main()
