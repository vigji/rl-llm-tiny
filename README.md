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

## SmolLM-135M fallback (`smollm_run.py`)

Self-contained variant for fast iteration / lower memory: GRPO+LoRA on
`HuggingFaceTB/SmolLM-135M-Instruct` with a "write exactly N words about
{topic}" task. Two reward funcs: `exact_count_reward` (sparse 0/1) and
`proximity_reward` (smooth `exp(-|diff|/5)` so GRPO has gradient even far
from target).

```bash
uv run python smollm_run.py smoke        # ~10s
uv run python smollm_run.py sanity       # ~30s
uv run python smollm_run.py train --max_steps 1000 --lora_r 32 --lora_alpha 64   # ~50 min
uv run python smollm_run.py eval --adapter runs/smollm-words/final
```

### Observed result and lesson (1000 steps, r=32)

- Base accuracy (greedy, 200 ex): **0.0%**
- Trained accuracy (greedy, 200 ex): **17.0%**
- Training proximity reward: 0.04 → 0.57 (14×)

**But:** the model is reward-hacking. It learned to echo the prompt
("Write exactly 5 words about rain." → "Write exactly 5 words about
rain."), which is consistently ~6 words. That hits when N=6, missing
otherwise. Apparent gain comes from gaming, not counting.

This is the textbook RLHF failure mode: reward up, behavior degenerate.
Mitigations: penalize prompt echo (e.g. n-gram overlap with prompt),
raise KL `beta` to keep the policy near base, or use a sharper reward
shape that requires content (not just length) to score.

