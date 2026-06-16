# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
"""PPO + Algorithm 1 Minmax penalty (DeepSpeed), aligned with ppo_minmax_experiment TRL code."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from safe_rlhf.algorithms.ppo.trainer_detoxify import PPODetoxifyTrainer, _scalar
from safe_rlhf.utils import (
    gather_log_probabilities,
    get_all_reduce_max,
    get_all_reduce_mean,
    is_main_process,
    masked_mean,
)


class PPOMinmaxTrainer(PPODetoxifyTrainer):
    """PPO with Detoxify rewards and Minmax penalty on unsafe outputs."""

    TRAINING_TYPE = 'ppo'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.r_min = 0.0
        self.r_max = 0.0
        self.v_min = 0.0
        self.v_max = 0.0
        self.current_values = None
        self.safety_threshold = self.args.safety_threshold
        self.penalty_floor = self.args.penalty_floor
        self._csv_rows: list[dict[str, Any]] = []

    def apply_minmax_penalty(self, reward: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        """Match TRL apply_minmax_penalty: bounds update, floor, unsafe replacement."""
        self.r_min = min(self.r_min, reward.min().item())
        self.r_max = max(self.r_max, reward.max().item())

        r_unsafe_raw = self.v_min - self.v_max
        floor_active = r_unsafe_raw < self.penalty_floor
        r_unsafe = max(r_unsafe_raw, self.penalty_floor)

        penalized = torch.where(
            reward < self.safety_threshold,
            torch.full_like(reward, r_unsafe),
            reward,
        )

        if self.current_values is not None:
            self.v_min = min(self.v_min, self.r_min, self.current_values.min().item())
            self.v_max = max(self.v_max, self.r_max, self.current_values.max().item())
        else:
            self.v_min = min(self.v_min, self.r_min)
            self.v_max = max(self.v_max, self.r_max)

        return penalized, r_unsafe_raw, floor_active

    def rl_step(self, rl_batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        prompt = rl_batch['prompt']
        old_log_probs = rl_batch['log_probs']
        ref_log_probs = rl_batch['ref_log_probs']
        reward = rl_batch['reward']
        old_reward_values = rl_batch['reward_values']
        input_ids = rl_batch['input_ids']
        attention_mask = rl_batch['attention_mask']

        start = prompt.size(-1) - 1
        sequence_mask = attention_mask[:, 1:]

        with torch.no_grad():
            self.current_values = old_reward_values
            penalized_reward, r_unsafe_raw, floor_active = self.apply_minmax_penalty(reward)

            old_rewards = self.add_kl_divergence_regularization(
                penalized_reward,
                prompt,
                old_log_probs,
                ref_log_probs,
                sequence_mask,
            )
            reward_advantages, reward_returns = self.get_advantages_and_returns(
                old_reward_values,
                old_rewards,
                sequence_mask,
                start,
            )

        logits = self.actor_model(input_ids, attention_mask=attention_mask, use_cache=False).logits
        log_probs = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        actor_loss = self.actor_loss_fn(
            log_probs[:, start:],
            old_log_probs[:, start:],
            reward_advantages,
            sequence_mask[:, start:],
        )
        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        reward_values = self.reward_critic_model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).scores
        reward_values = reward_values.squeeze(dim=-1)[:, :-1]
        reward_critic_loss = self.critic_loss_fn(
            reward_values[:, start:],
            old_reward_values[:, start:],
            reward_returns,
            sequence_mask[:, start:],
        )
        self.reward_critic_model.backward(reward_critic_loss)
        self.reward_critic_model.step()

        with torch.no_grad():
            mask = sequence_mask[:, start:]
            kl_divergence = ((old_log_probs - ref_log_probs)[:, start:] * mask).sum(dim=-1).mean()
            mean_generated_length = mask.sum(dim=-1).float().mean()
            max_generated_length = mask.sum(dim=-1).float().max()

            batch_reward = reward.mean()
            reward_with_penalty = (old_rewards[:, start:] * mask).sum(dim=-1).mean()
            reward_advantage = masked_mean(reward_advantages, mask)
            reward_return = masked_mean(reward_returns, mask)
            reward_value = masked_mean(reward_values[:, start:], mask)
            n_unsafe = (reward < self.safety_threshold).sum().float()

            actor_loss = get_all_reduce_mean(actor_loss)
            reward_critic_loss = get_all_reduce_mean(reward_critic_loss)
            batch_reward = get_all_reduce_mean(batch_reward)
            reward_with_penalty = get_all_reduce_mean(reward_with_penalty)
            reward_advantage = get_all_reduce_mean(reward_advantage)
            reward_return = get_all_reduce_mean(reward_return)
            reward_value = get_all_reduce_mean(reward_value)
            kl_divergence = get_all_reduce_mean(kl_divergence)
            mean_generated_length = get_all_reduce_mean(mean_generated_length)
            max_generated_length = get_all_reduce_max(max_generated_length)
            n_unsafe = get_all_reduce_mean(n_unsafe)

            self.v_min = min(self.v_min, reward.min().item())
            self.v_max = max(self.v_max, reward.max().item(), reward_value.item())

        dist.barrier()

        r_unsafe = max(self.v_min - self.v_max, self.penalty_floor)
        info = {
            'train/actor_loss': actor_loss.item(),
            'train/reward_critic_loss': reward_critic_loss.item(),
            'train/reward': batch_reward.item(),
            'train/reward_with_kl_penalty': reward_with_penalty.item(),
            'train/reward_advantage': reward_advantage.item(),
            'train/reward_return': reward_return.item(),
            'train/reward_value': reward_value.item(),
            'train/kl_divergence': kl_divergence.item(),
            'train/actor_lr': self.actor_model.optimizer.param_groups[0]['lr'],
            'train/reward_critic_lr': self.reward_critic_model.optimizer.param_groups[0]['lr'],
            'train/mean_generated_length': mean_generated_length.item(),
            'train/max_generated_length': max_generated_length.item(),
            'minmax/v_min': self.v_min,
            'minmax/v_max': self.v_max,
            'minmax/r_unsafe': r_unsafe,
            'minmax/r_unsafe_raw': r_unsafe_raw,
            'minmax/floor_active': float(floor_active),
            'minmax/n_unsafe': n_unsafe.item(),
        }

        if is_main_process() and getattr(self.args, 'training_log_csv', None):
            self._csv_rows.append({
                'step': self.global_step,
                'condition': 'minmax',
                'mean_reward': round(_scalar(reward_with_penalty), 4),
                'kl_divergence': round(_scalar(kl_divergence), 4),
                'v_min': round(self.v_min, 4),
                'v_max': round(self.v_max, 4),
                'r_unsafe': round(r_unsafe, 4),
                'r_unsafe_raw': round(r_unsafe_raw, 4),
                'floor_active': int(floor_active),
                'n_unsafe': int(_scalar(n_unsafe)),
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
