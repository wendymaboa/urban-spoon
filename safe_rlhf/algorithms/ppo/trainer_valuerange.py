# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
# Modified by Wendy Maboa (Wits University) for PPO-ValueRange experiment.
#
# This file implements Algorithm 1 from:
# "An Unreasonably Simple Approach to Safe Reinforcement Learning"
# (Nangue Tasse et al., RLC 2026, under review)
#
# The ONLY change from the original ppo/trainer.py is:
#   - add_kl_divergence_regularization() is replaced with the Value-Range penalty
#   - Six tracking attributes are added to __init__()
#   - self.current_values is set in rl_step() before calling the penalty method
#
# Baseline (original trainer.py) uses:
#   r_t = r_detoxify - kl_coeff * KL(Ï€ || Ï€_ref)
#
# This file uses (Algorithm 1):
#   r_t = V_MIN - V_MAX   if z_t > 0  (output is unsafe)
#   r_t = r_detoxify       otherwise   (output is safe)
#
# Where:
#   V_MIN = running minimum of value estimates seen during training
#   V_MAX = running maximum of value estimates seen during training
#   z_t   = 1 when Detoxify toxicity score > 0 (any nonzero toxicity)

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from safe_rlhf.trainers import RLTrainer
from safe_rlhf.utils import (
    batch_retokenize,
    gather_log_probabilities,
    get_all_reduce_max,
    get_all_reduce_mean,
    masked_mean,
)


class PPOValueRangeTrainer(RLTrainer):
    """
    PPO trainer with the Value-Range safety penalty (Algorithm 1).

    Replaces the KL divergence penalty with R_unsafe = V_MIN - V_MAX,
    applied only when the generated output is flagged as unsafe by Detoxify.
    No cost model, no Lagrange multiplier, no new hyperparameters.
    """

    TRAINING_TYPE = 'ppo'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # â”€â”€ Algorithm 1: Value-Range penalty tracking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # These six attributes persist across ALL training steps.
        # They are NEVER reset between batches or episodes.
        # V_MIN grows more negative over time as the agent encounters low-value states.
        # V_MAX grows larger over time as the agent finds high-reward states.
        # Their difference (V_MIN - V_MAX) is always negative and grows in magnitude.
        self.r_min = 0.0          # running min of raw rewards seen
        self.r_max = 0.0          # running max of raw rewards seen
        self.v_min = 0.0          # running min of V(s) estimates seen
        self.v_max = 0.0          # running max of V(s) estimates seen
        self.current_values = None  # set each step in rl_step()
        self.safety_threshold = 0.0  # reward < this â†’ unsafe (Detoxify scores negated)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # The following three methods are IDENTICAL to the original ppo/trainer.py
    # post_rollout, eval_step, and actor_loss_fn are unchanged.
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @torch.no_grad()
    def post_rollout(
        self,
        prompt: torch.Tensor,
        sequence: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> dict[str, Any]:
        if self.reward_tokenizer is not self.tokenizer:
            reward_tokenize_output = batch_retokenize(
                sequence,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.reward_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            reward_seq = reward_tokenize_output['input_ids']
            reward_attention_mask = reward_tokenize_output['attention_mask']
        else:
            reward_seq = sequence
            reward_attention_mask = attention_mask

        logits = self.actor_model(sequence, attention_mask=attention_mask).logits
        ref_logits = self.actor_reference_model(sequence, attention_mask=attention_mask).logits

        reward = self.reward_model(reward_seq, attention_mask=reward_attention_mask).end_scores
        reward_values = self.reward_critic_model(sequence, attention_mask=attention_mask).scores

        reward = reward.squeeze(dim=-1)
        reward_values = reward_values.squeeze(dim=-1)[:, :-1]

        log_probs = gather_log_probabilities(logits[:, :-1], sequence[:, 1:])
        ref_log_probs = gather_log_probabilities(ref_logits[:, :-1], sequence[:, 1:])
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
        if self.reward_tokenizer is not self.tokenizer:
            reward_tokenize_output = batch_retokenize(
                input_ids,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.reward_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            reward_input_ids = reward_tokenize_output['input_ids']
            reward_attention_mask = reward_tokenize_output['attention_mask']
        else:
            reward_input_ids = input_ids
            reward_attention_mask = attention_mask

        reward = self.reward_model(
            reward_input_ids,
            attention_mask=reward_attention_mask,
        ).end_scores.squeeze(dim=-1)
        return {
            'eval/reward': reward,
        }

    def actor_loss_fn(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        ratios = torch.exp(log_probs - old_log_probs)
        surrogate1 = advantages * ratios
        surrogate2 = advantages * torch.clamp(
            ratios,
            1.0 - self.clip_range_ratio,
            1.0 + self.clip_range_ratio,
        )
        surrogate = torch.minimum(surrogate1, surrogate2)
        return -masked_mean(surrogate, mask)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # THIS IS THE CHANGED METHOD â€” Algorithm 1 Value-Range penalty
    # Compare with original add_kl_divergence_regularization in ppo/trainer.py
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def add_value_range_penalty(
        self,
        reward: torch.Tensor,          # shape (B,) â€” one scalar per sequence
        prompt: torch.LongTensor,       # shape (B, S) â€” prompt tokens
        log_probs: torch.Tensor,        # shape (B, L) â€” kept for shape reference
        sequence_mask: torch.BoolTensor, # shape (B, L) â€” real tokens vs padding
    ) -> torch.Tensor:                  # shape (B, L) â€” reward at final token
        """
        Algorithm 1 implementation.

        Instead of subtracting a KL penalty, we:
        1. Update running bounds V_MIN, V_MAX from current batch
        2. Compute R_unsafe = V_MIN - V_MAX  (always negative)
        3. Replace reward with R_unsafe for any unsafe sequence (z_t > 0)
        4. Leave safe sequences with their original Detoxify reward

        The penalty grows in magnitude as training progresses because:
        - V_MAX increases as the agent discovers high-reward completions
        - V_MIN decreases as the agent encounters low-value states
        - Their difference therefore grows, creating increasing pressure to avoid unsafe states
        """

        # Find the index of the last real token in each sequence
        # This is where we place the reward signal (same as original)
        end_index = torch.cat([m.nonzero()[-1] for m in sequence_mask])  # shape (B,)

        # â”€â”€ Step 1: Update running reward bounds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.r_min = min(self.r_min, reward.min().item())
        self.r_max = max(self.r_max, reward.max().item())

        # â”€â”€ Step 2: Update running value bounds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # self.current_values contains old_reward_values from rl_step()
        # These are the critic's V(s) estimates for each token in the batch
        if self.current_values is not None:
            self.v_min = min(self.v_min, self.r_min, self.current_values.min().item())
            self.v_max = max(self.v_max, self.r_max, self.current_values.max().item())
        else:
            # First step â€” values not available yet, use reward bounds only
            self.v_min = min(self.v_min, self.r_min)
            self.v_max = max(self.v_max, self.r_max)

        # â”€â”€ Step 3: Compute the Minmax penalty â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # R_unsafe = V_MIN - V_MAX
        # Always <= 0. Grows more negative as training progresses.
        r_unsafe = self.v_min - self.v_max  # scalar float

        # â”€â”€ Step 4: Apply penalty to unsafe sequences only â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # reward is a negated Detoxify score: 0.0 = clean, negative = toxic
        # z_t > 0 when reward < safety_threshold (default 0.0)
        penalized_reward = torch.where(
            reward < self.safety_threshold,           # z_t > 0: unsafe?
            torch.full_like(reward, r_unsafe),        # yes â†’ apply Minmax penalty
            reward,                                    # no  â†’ keep Detoxify reward
        )

        # â”€â”€ Logging: print every step so you can verify it's working â”€â”€â”€â”€â”€â”€
        n_unsafe = (reward < self.safety_threshold).sum().item()
        print(
            f"[ValueRange] "
            f"v_min={self.v_min:.4f}  "
            f"v_max={self.v_max:.4f}  "
            f"r_unsafe={r_unsafe:.4f}  "
            f"unsafe={n_unsafe}/{len(reward)}"
        )

        # â”€â”€ Step 5: Place reward at final token position â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # PPO needs a reward tensor of shape (B, L)
        # We put the scalar reward at the last token, zeros everywhere else
        base_rewards = torch.zeros_like(log_probs)
        rewards = torch.scatter_add(
            base_rewards,
            dim=-1,
            index=end_index.unsqueeze(dim=-1),
            src=penalized_reward.to(base_rewards.dtype).unsqueeze(dim=-1),
        )

        # Clip to prevent extreme values destabilising training
        return torch.clamp(rewards, min=-self.clip_range_score, max=self.clip_range_score)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # rl_step â€” same as original with two small additions marked with NEW
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
            # â”€â”€ NEW: expose value estimates to the penalty method â”€â”€â”€â”€â”€â”€â”€â”€â”€
            self.current_values = old_reward_values  # shape (B, L)

            # â”€â”€ NEW: call our penalty instead of KL regularization â”€â”€â”€â”€â”€â”€â”€â”€
            old_rewards = self.add_value_range_penalty(
                reward,
                prompt,
                old_log_probs,
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
            # NEW: still log KL divergence so the metric is comparable across
            # PPO (KL-penalised) and PPO-ValueRange (KL anchor only) runs.
            kl_divergence = ((old_log_probs - ref_log_probs)[:, start:] * mask).sum(dim=-1).mean()
            mean_generated_length = mask.sum(dim=-1).float().mean()
            max_generated_length = mask.sum(dim=-1).float().max()

            reward = reward.mean()
            reward_with_penalty = (old_rewards[:, start:] * mask).sum(dim=-1).mean()
            reward_advantage = masked_mean(reward_advantages, mask)
            reward_return = masked_mean(reward_returns, mask)
            reward_value = masked_mean(reward_values[:, start:], mask)

            actor_loss = get_all_reduce_mean(actor_loss)
            reward_critic_loss = get_all_reduce_mean(reward_critic_loss)
            reward = get_all_reduce_mean(reward)
            reward_with_penalty = get_all_reduce_mean(reward_with_penalty)
            reward_advantage = get_all_reduce_mean(reward_advantage)
            reward_return = get_all_reduce_mean(reward_return)
            reward_value = get_all_reduce_mean(reward_value)
            kl_divergence = get_all_reduce_mean(kl_divergence)
            mean_generated_length = get_all_reduce_mean(mean_generated_length)
            max_generated_length = get_all_reduce_max(max_generated_length)

        dist.barrier()

        # NEW: Algorithm 1 diagnostic metrics in addition to standard PPO ones.
        r_unsafe = self.v_min - self.v_max
        return {
            'train/actor_loss': actor_loss.item(),
            'train/reward_critic_loss': reward_critic_loss.item(),
            'train/reward': reward.item(),
            'train/reward_with_penalty': reward_with_penalty.item(),
            'train/reward_advantage': reward_advantage.item(),
            'train/reward_return': reward_return.item(),
            'train/reward_value': reward_value.item(),
            'train/kl_divergence': kl_divergence.item(),
            'train/actor_lr': self.actor_model.optimizer.param_groups[0]['lr'],
            'train/reward_critic_lr': self.reward_critic_model.optimizer.param_groups[0]['lr'],
            'train/mean_generated_length': mean_generated_length.item(),
            'train/max_generated_length': max_generated_length.item(),
            # Algorithm 1 dynamics
            'valuerange/v_min': self.v_min,
            'valuerange/v_max': self.v_max,
            'valuerange/r_unsafe': r_unsafe,
            'valuerange/r_min': self.r_min,
            'valuerange/r_max': self.r_max,
        }
