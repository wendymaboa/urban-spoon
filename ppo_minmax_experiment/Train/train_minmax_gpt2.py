"""
PPO + MINMAX PENALTY  (Algorithm 1, LLM adaptation of ROSARL)
==============================================================
Experimental condition. The Minmax penalty replaces KL-as-safety.

Usage:
    python Train/train_minmax_gpt2.py
    python Train/train_minmax_gpt2.py --steps 1000

Output:
    results/minmax_training_log.csv
    models/minmax/
"""

import argparse
import csv
import os
import sys
import torch
from datasets import load_dataset
from detoxify import Detoxify
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from trl_compat import AutoModelForCausalLMWithValueHead, PPOTrainer, make_ppo_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-2 with Minmax penalty.")
    parser.add_argument("--steps", type=int, default=500, help="Number of PPO training steps.")
    parser.add_argument("--batch-size", type=int, default=4, help="PPO batch size.")
    parser.add_argument("--mini-batch", type=int, default=2, help="PPO mini-batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max tokens generated per step.")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name.")
    parser.add_argument("--output-dir", default="results", help="Directory for training logs.")
    parser.add_argument("--model-save", default="models/minmax", help="Directory to save the trained model.")
    parser.add_argument("--learning-rate", type=float, default=1.41e-5, help="PPO learning rate.")
    parser.add_argument("--kl-coef", type=float, default=0.01, help="KL stability anchor (not a safety parameter).")
    parser.add_argument(
        "--safety-threshold", type=float, default=-0.3,
        help="Raw reward below this triggers Minmax penalty (toxicity > 0.3).",
    )
    parser.add_argument(
        "--penalty-floor", type=float, default=-1.0,
        help="Engineering floor on R_unsafe to guard early-training instability.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.join(ROOT_DIR, args.output_dir)
    model_save = os.path.join(ROOT_DIR, args.model_save)
    log_file = "minmax_training_log.csv"
    condition = "minmax"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_save, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.model} with value head...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model).to(device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model).to(device)

    model.v_head.summary.weight.data.zero_()
    model.v_head.summary.bias.data.zero_()

    config = make_ppo_config(
        model_name=args.model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch,
        gradient_accumulation_steps=1,
        ppo_epochs=1,
        optimize_cuda_cache=True,
        kl_penalty="kl",
        init_kl_coef=args.kl_coef,
        adap_kl_ctrl=False,   # fixed KL anchor — no adaptive beta tuning
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
        return [-float(s) for s in scores]

    print("Loading BeaverTails training prompts...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    all_prompts = [item["prompt"] for item in ds]
    print(f"Loaded {len(all_prompts)} prompts")

    r_min, r_max = 0.0, 0.0
    v_min, v_max = 0.0, 0.0

    def apply_minmax_penalty(raw_rewards, trl_value_estimates):
        nonlocal r_min, r_max, v_min, v_max

        r_min = min(r_min, min(raw_rewards))
        r_max = max(r_max, max(raw_rewards))

        r_unsafe_raw = v_min - v_max
        floor_active = r_unsafe_raw < args.penalty_floor
        r_unsafe = max(r_unsafe_raw, args.penalty_floor)

        penalized = []
        for r in raw_rewards:
            if r < args.safety_threshold:
                penalized.append(torch.tensor(r_unsafe, dtype=torch.float32))
            else:
                penalized.append(torch.tensor(r, dtype=torch.float32))

        if trl_value_estimates is not None and trl_value_estimates.numel() > 0:
            v_min = min(v_min, r_min, trl_value_estimates.min().item())
            v_max = max(v_max, r_max, trl_value_estimates.max().item())
        else:
            v_min = min(v_min, r_min)
            v_max = max(v_max, r_max)

        return penalized, r_unsafe_raw, floor_active

    print(f"\nStarting MINMAX training for {args.steps} steps...")
    print(f"Condition: PPO + Algorithm 1 (Minmax penalty, beta={config.init_kl_coef})\n")
    print(
        f"Safety threshold: toxicity > {abs(args.safety_threshold):.1f}  "
        f"(raw reward < {args.safety_threshold})"
    )
    print(f"Penalty floor: {args.penalty_floor}  (engineering guard, NOT Algorithm 1)\n")

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
        raw_rewards = compute_reward(batch_responses)

        penalized_rewards, r_unsafe_raw, floor_active = apply_minmax_penalty(
            raw_rewards,
            trl_value_estimates=None,
        )

        stats = trainer.step(query_tensors, response_tensors, penalized_rewards)

        vpred_mean = stats.get("val/vpred", None)
        if vpred_mean is not None:
            vpred_tensor = torch.tensor([float(vpred_mean)], dtype=torch.float32)
            v_min = min(v_min, min(raw_rewards), vpred_tensor.min().item())
            v_max = max(v_max, max(raw_rewards), vpred_tensor.max().item())
        else:
            v_min = min(v_min, min(raw_rewards))
            v_max = max(v_max, max(raw_rewards))

        mean_reward = sum(r.item() for r in penalized_rewards) / len(penalized_rewards)
        kl = float(stats.get("objective/kl", 0.0))
        n_unsafe = sum(1 for r in raw_rewards if r < args.safety_threshold)

        log_rows.append({
            "step": step,
            "condition": condition,
            "mean_reward": round(mean_reward, 4),
            "kl_divergence": round(kl, 4),
            "v_min": round(v_min, 4),
            "v_max": round(v_max, 4),
            "r_unsafe": round(max(v_min - v_max, args.penalty_floor), 4),
            "r_unsafe_raw": round(r_unsafe_raw, 4),
            "floor_active": int(floor_active),
            "n_unsafe": n_unsafe,
        })

        if step % 10 == 0:
            print(
                f"Step {step:4d} | "
                f"reward={mean_reward:.4f} | "
                f"v_min={v_min:.4f} | v_max={v_max:.4f} | "
                f"r_unsafe_raw={r_unsafe_raw:.4f} | "
                f"floor={'YES' if floor_active else 'no ':3s} | "
                f"kl={kl:.4f} | "
                f"unsafe={n_unsafe}/{len(raw_rewards)}"
            )

    if not log_rows:
        raise RuntimeError("No training steps completed — check GPU, dataset access, and TRL install.")

    log_path = os.path.join(output_dir, log_file)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log saved to {log_path}")

    trainer.model.save_pretrained(model_save)
    tokenizer.save_pretrained(model_save)
    print(f"Model saved to {model_save}")
    print("\nMINMAX DONE.")


if __name__ == "__main__":
    main()
