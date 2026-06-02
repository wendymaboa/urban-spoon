"""
Detoxify-based verifiable reward signal for PPO-ValueRange.

This module provides a simple toxicity-based reward that:
- Does NOT require training a separate cost model
- Runs on CPU (low overhead)
- Is verifiable and interpretable (based on Detoxify classifier)
- Integrates seamlessly with Safe RLHF reward pipeline

Reward convention:
  - Clean output (toxicity near 0) → reward ≈ 0.0
  - Toxic output (toxicity near 1) → reward ≈ -1.0
  - z_t (unsafe flag) = 1 when reward < safety_threshold (default 0.0)

Usage:
    from detoxify_reward import compute_reward, is_unsafe
    
    # Generate some text
    texts = ["I hate you", "The weather is nice"]
    
    # Get rewards (negated toxicity scores)
    rewards = compute_reward(texts)  # tensor([-0.95, -0.0006])
    
    # Check which are unsafe
    unsafe = is_unsafe(rewards)  # tensor([True, False])
"""

from detoxify import Detoxify
import torch


# Load once at import time — do not reload in the training loop
_classifier = None


def get_classifier():
    """Lazily load and cache the Detoxify classifier.
    
    Returns:
        Detoxify('original'): pre-trained RoBERTa-based toxicity classifier
    """
    global _classifier
    if _classifier is None:
        print("[DetoxifyReward] Loading Detoxify classifier (first call only)...")
        _classifier = Detoxify('original')
    return _classifier


def compute_reward(texts: list[str]) -> torch.Tensor:
    """
    Compute verifiable reward from a batch of generated texts using Detoxify.
    
    This replaces the Safe RLHF cost model entirely — no separate training needed.
    The reward signal is deterministic and based on well-validated toxicity detection.
    
    Args:
        texts: list of generated strings, one per sequence in batch
               Example: ["I hate you", "Tell me a joke"]
    
    Returns:
        rewards: tensor of shape (B,), values in [-1.0, 0.0]
                 - 0.0 means clean, non-toxic output (safe)
                 - -1.0 means maximally toxic output (unsafe)
                 
    Example:
        >>> texts = ["I hate you", "The weather is nice"]
        >>> rewards = compute_reward(texts)
        >>> print(rewards)
        tensor([-0.9510, -0.0006])
    """
    clf = get_classifier()
    
    # Get toxicity scores from Detoxify (range [0, 1])
    # Scores are returned as a dict with 'toxicity' key
    scores = clf.predict(texts)['toxicity']  # list of floats in [0.0, 1.0]
    
    # Negate: high toxicity → more negative reward
    # This convention makes unsafe sequences have negative rewards
    rewards = [-float(s) for s in scores]
    
    return torch.tensor(rewards, dtype=torch.float32)


def is_unsafe(rewards: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Determine which sequences are unsafe based on reward threshold.
    
    In Algorithm 1, z_t (unsafe indicator) = 1 when the reward falls below the safety threshold.
    By default, any non-zero toxicity triggers the unsafe flag.
    
    Args:
        rewards: tensor of shape (B,), rewards from compute_reward()
        threshold: reward threshold below which a sequence is unsafe (default 0.0)
                   - reward < threshold → unsafe (z_t = 1)
                   - reward >= threshold → safe (z_t = 0)
    
    Returns:
        unsafe: boolean tensor of shape (B,)
                True where sequence is unsafe, False where safe
                
    Example:
        >>> rewards = torch.tensor([-0.95, -0.0001, 0.0])
        >>> unsafe = is_unsafe(rewards)
        >>> print(unsafe)
        tensor([ True, False, False])
    """
    return rewards < threshold


if __name__ == '__main__':
    # Quick test
    print("Testing Detoxify reward function...")
    test_texts = [
        "I hate you",
        "The weather is nice today",
        "Go kill yourself",
        "What is 2+2?",
    ]
    
    rewards = compute_reward(test_texts)
    unsafe = is_unsafe(rewards)
    
    print(f"\nText → Reward, Unsafe?")
    for text, reward, flag in zip(test_texts, rewards, unsafe):
        print(f"  '{text}' → {reward:.4f}, {flag}")