# rl-llm-tiny

Tiny GRPO + LoRA experiment: fine-tune `Qwen/Qwen2.5-0.5B-Instruct` on
synthetic 2-digit addition with a verifiable reward (correct answer
inside `<answer>...</answer>` tags). Designed to run on a single
Apple Silicon Mac (tested on M1 Pro / 32 GB).

## Setup

```bash
uv sync
```

## Workflow

```bash
# 1. Smoke test: load model on MPS, generate on one prompt
uv run python smoke_test.py

# 2. Reward sanity: 20 base-model generations + reward distribution
uv run python reward_sanity.py

# 3. Train (GRPO + LoRA, ~1-2h on M1 Pro for 300 steps)
uv run python train.py --max_steps 300

# 4. Evaluate base vs trained
uv run python eval.py                                   # base
uv run python eval.py --adapter runs/grpo-qwen-add/final
```

## Files

- `make_dataset.py` — synthetic 2-digit add prompts (chat format)
- `reward.py` — `correctness_reward` + `format_reward` for `GRPOTrainer`
- `train.py` — `GRPOTrainer` setup with LoRA
- `eval.py` — accuracy / format-rate on held-out set
- `smoke_test.py`, `reward_sanity.py` — quick verification before long runs

## Stack

`torch` (MPS) · `transformers` · `trl` (`GRPOTrainer`) · `peft` (LoRA) · `datasets` · `accelerate`
