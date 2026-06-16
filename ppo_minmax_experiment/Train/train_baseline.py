"""
BASELINE: PPO + KL divergence penalty (standard RLHF)
=========================================================
Control condition. TRL's PPOTrainer applies KL divergence internally as:
    effective_reward = your_reward - beta * KL[pi_theta || pi_ref]

This is the safety mechanism we are replacing with the Minmax penalty.
init_kl_coef=0.2 is the standard InstructGPT value and is the
hyperparameter our thesis claims to eliminate.

Usage:
    python Train/train_baseline.py
    python Train/train_baseline.py --steps 1000 --batch-size 4

Output:
    results/baseline_training_log.csv
    models/baseline/
"""

import argparse
import csv
import os
import torch
from datasets import load_dataset
from detoxify import Detoxify
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-2 baseline (PPO + KL).")
    parser.add_argument("--steps", type=int, default=500, help="Number of PPO training steps.")
    parser.add_argument("--batch-size", type=int, default=4, help="PPO batch size.")
    parser.add_argument("--mini-batch", type=int, default=2, help="PPO mini-batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max tokens generated per step.")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name.")
    parser.add_argument("--output-dir", default="results", help="Directory for training logs.")
    parser.add_argument("--model-save", default="models/baseline", help="Directory to save the trained model.")
    parser.add_argument("--learning-rate", type=float, default=1.41e-5, help="PPO learning rate.")
    parser.add_argument("--kl-coef", type=float, default=0.2, help="KL penalty coefficient (beta).")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.join(ROOT_DIR, args.output_dir)
    model_save = os.path.join(ROOT_DIR, args.model_save)
    log_file = "baseline_training_log.csv"
    condition = "baseline"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_save, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.model} with value head...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)

    model.v_head.summary.weight.data.zero_()
    model.v_head.summary.bias.data.zero_()

    config = PPOConfig(
        model_name=args.model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch,
        gradient_accumulation_steps=1,
        ppo_epochs=1,
        optimize_cuda_cache=True,
        kl_penalty="kl",
        init_kl_coef=args.kl_coef,
        target=6,
        horizon=10000,
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        log_with=None,
    )

    trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    print(f"Model device: {next(model.parameters()).device}")

    print("Loading Detoxify classifier...")
    toxicity_clf = Detoxify("original")

    def compute_reward(texts):
        scores = toxicity_clf.predict(texts)["toxicity"]
        return [torch.tensor(-float(s), dtype=torch.float32) for s in scores]

    print("Loading BeaverTails training prompts...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    all_prompts = [item["prompt"] for item in ds]
    print(f"Loaded {len(all_prompts)} prompts")

    print(f"\nStarting BASELINE training for {args.steps} steps...")
    print(f"Condition: PPO + KL divergence (beta={config.init_kl_coef})\n")

    log_rows = []

    for step in range(args.steps):
        batch_prompts = all_prompts[step * args.batch_size : (step + 1) * args.batch_size]
        if len(batch_prompts) < args.batch_size:
            break

        query_tensors = [
            tokenizer(
                p, return_tensors="pt", padding=False,
                truncation=True, max_length=128,
            )["input_ids"].squeeze(0)
            for p in batch_prompts
        ]

        valid_queries, response_tensors = [], []
        for query in query_tensors:
            if query.numel() == 0:
                continue
            response = trainer.generate(
                query.to(device),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            valid_queries.append(query)
            response_tensors.append(response.squeeze(0)[len(query):].cpu())
        query_tensors = valid_queries

        if not query_tensors:
            continue

        batch_responses = [
            tokenizer.decode(r, skip_special_tokens=True)
            for r in response_tensors
        ]
        rewards = compute_reward(batch_responses)

        stats = trainer.step(query_tensors, response_tensors, rewards)

        mean_reward = sum(r.item() for r in rewards) / len(rewards)
        kl = float(stats.get("objective/kl", 0.0))

        log_rows.append({
            "step": step,
            "condition": condition,
            "mean_reward": round(mean_reward, 4),
            "kl_divergence": round(kl, 4),
            "v_min": "N/A",
            "v_max": "N/A",
            "r_unsafe": "N/A",
            "r_unsafe_raw": "N/A",
        })

        if step % 10 == 0:
            print(
                f"Step {step:4d} | "
                f"reward={mean_reward:.4f} | "
                f"kl={kl:.4f}"
            )

    log_path = os.path.join(output_dir, log_file)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log saved to {log_path}")

    trainer.model.save_pretrained(model_save)
    tokenizer.save_pretrained(model_save)
    print(f"Model saved to {model_save}")
    print("\nBASELINE DONE.")


if __name__ == "__main__":
    main()
