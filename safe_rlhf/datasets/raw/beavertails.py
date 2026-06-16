# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
"""BeaverTails prompt-only dataset (PKU-Alignment/BeaverTails, 30k_train split).

Note: upstream safe-rlhf registers PKU-SafeRLHF (preference pairs) for standard PPO,
not this HuggingFace repo. We register BeaverTails here to match the TRL experiment in
ppo_minmax_experiment, which trains on prompt-only BeaverTails — not PKU-SafeRLHF/train.

For native PKU PPO (learned reward model), use --train_datasets PKU-SafeRLHF/train instead.
"""

from __future__ import annotations

from datasets import load_dataset

from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = ['BeaverTailsDataset']


class BeaverTailsDataset(RawDataset):
    NAME: str = 'BeaverTails'
    ALIASES: tuple[str, ...] = (
        'PKU-Alignment/BeaverTails',
        'beavertails',
        'BeaverTails/30k_train',
    )

    def __init__(self, path: str | None = None, split: str = '30k_train') -> None:
        self.data = load_dataset(path or 'PKU-Alignment/BeaverTails', split=split)

    def __getitem__(self, index: int) -> RawSample:
        return RawSample(input=self.data[index]['prompt'])

    def __len__(self) -> int:
        return len(self.data)
