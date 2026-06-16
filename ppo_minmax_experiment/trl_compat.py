"""TRL imports and version-tolerant PPOConfig factory."""

from __future__ import annotations

import inspect


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
            'Could not import TRL PPO classes. Install a compatible version:\n'
            '  pip install "transformers>=4.37,<5" "trl==0.11.4"'
        ) from exc


PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead = _import_ppo_symbols()


def make_ppo_config(**kwargs):
    """Build PPOConfig, dropping kwargs unsupported by the installed TRL version."""
    params = inspect.signature(PPOConfig.__init__).parameters
    filtered = {key: value for key, value in kwargs.items() if key in params}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f'[trl_compat] PPOConfig ignoring unsupported keys: {dropped}')
    return PPOConfig(**filtered)


__all__ = [
    'PPOConfig',
    'PPOTrainer',
    'AutoModelForCausalLMWithValueHead',
    'make_ppo_config',
]
