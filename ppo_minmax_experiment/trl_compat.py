"""TRL imports compatible across versions (Colab may install TRL 0.27+)."""

from __future__ import annotations

import importlib


def _import_ppo_symbols():
    try:
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

        return PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    except ImportError:
        pass

    try:
        from trl.experimental.ppo import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

        return PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    except ImportError as exc:
        raise ImportError(
            'Could not import TRL PPO classes. Install a compatible version, e.g. '
            'pip install "trl>=0.11,<0.17" or pip install -r requirements.txt'
        ) from exc


PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead = _import_ppo_symbols()

__all__ = ['PPOConfig', 'PPOTrainer', 'AutoModelForCausalLMWithValueHead']
