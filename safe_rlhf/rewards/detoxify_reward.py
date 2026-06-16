"""Detoxify-based verifiable reward for GPT-2 Minmax experiments."""

from __future__ import annotations

import torch
from detoxify import Detoxify

_classifier: Detoxify | None = None


def get_classifier() -> Detoxify:
    global _classifier
    if _classifier is None:
        _classifier = Detoxify('original')
    return _classifier


def compute_reward(texts: list[str]) -> torch.Tensor:
    """Return negated toxicity scores in [-1.0, 0.0]."""
    scores = get_classifier().predict(texts)['toxicity']
    return torch.tensor([-float(s) for s in scores], dtype=torch.float32)


def is_unsafe(rewards: torch.Tensor, threshold: float) -> torch.Tensor:
    return rewards < threshold
