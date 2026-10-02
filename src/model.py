import torch
from torch import nn


class DemandLSTM(nn.Module):
    """LSTM model for multi-step demand forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 7,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            forecast_horizon,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a multi-step demand forecast."""
        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)

        forecast = self.fc(last_output)

        return forecast