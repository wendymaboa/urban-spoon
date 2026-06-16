# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
"""PPO trainer using Detoxify verifiable rewards (matches TRL experiment setup)."""

from __future__ import annotations

from typing import Any

import torch

from safe_rlhf.algorithms.ppo.trainer import PPOTrainer
from safe_rlhf.rewards.detoxify_reward import compute_reward
from safe_rlhf.utils import gather_log_probabilities, is_main_process


def _scalar(value: float | torch.Tensor) -> float:
    return value.item() if isinstance(value, torch.Tensor) else float(value)


class PPODetoxifyTrainer(PPOTrainer):
    """PPO with Detoxify rewards instead of a learned reward model."""

    TRAINING_TYPE = 'ppo'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._csv_rows: list[dict] = []

    @torch.no_grad()
    def post_rollout(
        self,
        prompt: torch.Tensor,
        sequence: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> dict[str, Any]:
        logits = self.actor_model(sequence, attention_mask=attention_mask).logits
        ref_logits = self.actor_reference_model(sequence, attention_mask=attention_mask).logits
        reward_values = self.reward_critic_model(sequence, attention_mask=attention_mask).scores
        reward_values = reward_values.squeeze(dim=-1)[:, :-1]

        log_probs = gather_log_probabilities(logits[:, :-1], sequence[:, 1:])
        ref_log_probs = gather_log_probabilities(ref_logits[:, :-1], sequence[:, 1:])

        prompt_len = prompt.size(-1)
        responses: list[str] = []
        for idx in range(sequence.size(0)):
            response_ids = sequence[idx, prompt_len:]
            response_mask = attention_mask[idx, prompt_len:]
            valid_ids = response_ids[response_mask]
            responses.append(self.tokenizer.decode(valid_ids, skip_special_tokens=True))

        reward = compute_reward(responses).to(device=self.args.device, dtype=torch.float32)

        return {
            'prompt': prompt,
            'log_probs': log_probs,
            'ref_log_probs': ref_log_probs,
            'reward': reward,
            'reward_values': reward_values,
            'input_ids': sequence,
            'attention_mask': attention_mask,
        }

    @torch.no_grad()
    def eval_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> dict[str, torch.Tensor]:
        generated = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        reward = compute_reward(generated).to(device=self.args.device, dtype=torch.float32)
        return {'eval/reward': reward}

    def rl_step(self, rl_batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        info = super().rl_step(rl_batch)
        if is_main_process() and getattr(self.args, 'training_log_csv', None):
            reward = rl_batch['reward']
            threshold = getattr(self.args, 'safety_threshold', -0.3)
            n_unsafe = int((reward < threshold).sum().item())
            # mean_reward: raw Detoxify score (matches TRL baseline logging)
            self._csv_rows.append({
                'step': self.global_step,
                'condition': 'baseline',
                'mean_reward': round(_scalar(info['train/reward']), 4),
                'kl_divergence': round(_scalar(info['train/kl_divergence']), 4),
                'v_min': 'N/A',
                'v_max': 'N/A',
                'r_unsafe': 'N/A',
                'r_unsafe_raw': 'N/A',
                'floor_active': 0,
                'n_unsafe': n_unsafe,
            })
        return info

    def save(self, model=None, ds_config=None) -> None:
        super().save(model=model, ds_config=ds_config)
        if is_main_process() and self._csv_rows and getattr(self.args, 'training_log_csv', None):
            import csv
            import os

            os.makedirs(os.path.dirname(self.args.training_log_csv) or '.', exist_ok=True)
            with open(self.args.training_log_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(self._csv_rows)
