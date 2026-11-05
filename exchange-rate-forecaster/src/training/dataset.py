"""PyTorch Dataset that converts (X, y) arrays into sliding-window sequences."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class RateSequenceDataset(Dataset):
    """
    Wraps feature/target arrays into overlapping sequences of length `seq_len`.

    Each sample is (X[t:t+seq_len], y[t+seq_len]) — the model predicts
    the rate at the next timestep given a window of historical features.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 30) -> None:
        if len(X) <= seq_len:
            raise ValueError(f"Dataset too short ({len(X)}) for seq_len={seq_len}")
        self.seq_len = seq_len
        # Convert once to avoid repeated casting inside __getitem__
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx : idx + self.seq_len]          # (seq_len, n_features)
        y_target = self.y[idx + self.seq_len].unsqueeze(0) # (1,)
        return x_seq, y_target
