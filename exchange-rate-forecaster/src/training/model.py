"""
LSTM model for exchange rate forecasting.

Architecture:
  Input → Stacked LSTM (dropout between layers) → FC head → scalar output

The model outputs a single normalised rate value.
Denormalisation happens at inference time using the saved y_scaler.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Multi-layer LSTM with a fully-connected regression head.

    Args:
        input_size:  Number of input features per timestep.
        hidden_size: Number of LSTM hidden units.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout probability between LSTM layers (ignored when num_layers=1).
        fc_hidden:   Hidden units in the intermediate FC layer (0 to skip).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        fc_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        head_layers: list[nn.Module] = []
        if fc_hidden > 0:
            head_layers += [
                nn.Linear(hidden_size, fc_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            head_layers.append(nn.Linear(fc_hidden, 1))
        else:
            head_layers.append(nn.Linear(hidden_size, 1))

        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, 1)
        """
        _, (h_n, _) = self.lstm(x)
        # Use the last layer's hidden state
        out = self.head(h_n[-1])
        return out


def build_model(input_size: int, config: dict) -> LSTMForecaster:
    return LSTMForecaster(
        input_size=input_size,
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.2),
        fc_hidden=config.get("fc_hidden", 64),
    )
