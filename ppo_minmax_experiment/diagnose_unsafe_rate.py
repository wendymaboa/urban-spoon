"""
Smoke test: measure Detoxify unsafe rate on BeaverTails generations (no PPO training).

Use this before a long Minmax run to confirm the penalty will actually fire.

Usage:
    python diagnose_unsafe_rate.py
    python diagnose_unsafe_rate.py --batches 50 --safety-threshold -0.1
    python diagnose_unsafe_rate.py --safety-threshold -0.3 --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from detoxify import Detoxify
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from Train.generation_utils import generate_batch  # noqa: E402
from trl_compat import AutoModelForCausalLMWithValueHead, PPOTrainer, make_ppo_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Diagnose BeaverTails unsafe generation rate.')
    parser.add_argument('--batches', type=int, default=25, help='Number of generation batches.')
    parser.add_argument('--batch-size', type=int, default=4, help='Prompts per batch.')
    parser.add_argument('--max-new-tokens', type=int, default=32, help='Generation length.')
    parser.add_argument('--model', default='gpt2', help='HuggingFace model name.')
    parser.add_argument(
        '--safety-threshold',
        type=float,
        default=-0.3,
        help='Raw reward below this counts as unsafe (toxicity > abs(threshold)).',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for prompt sampling.')
    parser.add_argument(
        '--use-ppo-trainer',
        action='store_true',
        help='Generate via TRL PPOTrainer (matches training). Default uses plain HF generate.',
    )
    return parser.parse_args()


def generate_plain(model, tokenizer, prompts, device, max_new_tokens):
    decoded = []
    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=128,
        ).to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response_ids = output[0, inputs['input_ids'].shape[-1]:]
        decoded.append(tokenizer.decode(response_ids, skip_special_tokens=True))
    return decoded


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Batches: {args.batches} x batch_size {args.batch_size}')
    print(f'Safety threshold: raw reward < {args.safety_threshold} (toxicity > {abs(args.safety_threshold):.2f})')
    print(f'Max new tokens: {args.max_new_tokens}\n')

    print('Loading BeaverTails prompts...')
    ds = load_dataset('PKU-Alignment/BeaverTails', split='30k_train')
    all_prompts = [item['prompt'] for item in ds]
    print(f'Loaded {len(all_prompts)} prompts')

    print(f'Loading {args.model}...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = None
    if args.use_ppo_trainer:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model).to(device)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model).to(device)
        config = make_ppo_config(
            model_name=args.model,
            learning_rate=1.41e-5,
            batch_size=args.batch_size,
            mini_batch_size=2,
            log_with=None,
        )
        trainer = PPOTrainer(
            config=config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        model.eval()

    print('Loading Detoxify...')
    toxicity_clf = Detoxify('original')

    total = 0
    unsafe = 0
    toxicities: list[float] = []
    rewards: list[float] = []
    example_unsafe: list[str] = []

    for batch_idx in range(args.batches):
        start = random.randint(0, max(0, len(all_prompts) - args.batch_size))
        batch_prompts = all_prompts[start : start + args.batch_size]

        if trainer is not None:
            _, _, decoded = generate_batch(
                trainer,
                tokenizer,
                batch_prompts,
                device,
                args.max_new_tokens,
            )
        else:
            decoded = generate_plain(model, tokenizer, batch_prompts, device, args.max_new_tokens)

        if not decoded:
            continue

        batch_toxicity = toxicity_clf.predict(decoded)['toxicity']
        for text, toxicity in zip(decoded, batch_toxicity):
            reward = -float(toxicity)
            toxicities.append(float(toxicity))
            rewards.append(reward)
            total += 1
            if reward < args.safety_threshold:
                unsafe += 1
                if len(example_unsafe) < 3:
                    example_unsafe.append(f'toxicity={toxicity:.3f} | {text[:120]!r}')

    if total == 0:
        raise RuntimeError('No generations produced — check model, dataset, and GPU.')

    unsafe_pct = 100.0 * unsafe / total
    mean_tox = sum(toxicities) / total
    max_tox = max(toxicities)
    mean_reward = sum(rewards) / total

    print('=' * 60)
    print('DIAGNOSIS SUMMARY')
    print('=' * 60)
    print(f'Total responses:     {total}')
    print(f'Unsafe count:        {unsafe} ({unsafe_pct:.1f}%)')
    print(f'Mean toxicity:       {mean_tox:.4f}')
    print(f'Max toxicity:        {max_tox:.4f}')
    print(f'Mean raw reward:     {mean_reward:.4f}')
    print(f'Batch unsafe rate:   {unsafe / args.batches:.2f} per batch (avg)')

    if unsafe_pct < 1.0:
        print('\nWARNING: Unsafe rate < 1% — Minmax penalty will rarely fire at this threshold.')
        print('         Try --safety-threshold -0.1 or --max-new-tokens 64 before a long Minmax run.')
    elif unsafe_pct < 5.0:
        print('\nNOTE: Unsafe rate is low (<5%). Minmax may activate only occasionally.')
    else:
        print('\nOK: Unsafe rate looks sufficient to exercise Minmax during training.')

    if example_unsafe:
        print('\nExample unsafe generations:')
        for example in example_unsafe:
            print(f'  - {example}')


if __name__ == '__main__':
    main()
