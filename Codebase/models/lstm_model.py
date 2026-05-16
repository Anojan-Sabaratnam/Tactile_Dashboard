"""Bidirectional LSTM for tactile grasp stability prediction.

Treats the 3 fingers as a sequence (index -> middle -> thumb).
Each timestep is a 24-dimensional vector of raw taxel readings.
Input shape: (N, 3, 24)

Design rationale:
- Explores inter-finger relationships as a sequence, testing if
  there's an inherent "flow" or dependency across the grasp digits.
- The attention mechanism highlights which finger contributes most
  to the stability prediction (e.g., if the thumb slipping is critical).
- Provides a deep learning sequence-model contrast to the CNN and RF.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TactileBiLSTM(nn.Module):
    """BiLSTM with simple attention over fingers."""

    def __init__(
        self,
        input_size: int = 24,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM: (N, seq_len=3, 24) -> (N, 3, hidden_size*2)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Attention layer over the 3 timesteps (fingers)
        # Learnable context vector to score each timestep
        self.attention_weights = nn.Linear(hidden_size * 2, 1)

        # Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        # Input is hidden_size * 2 (bidirectional)
        self.fc1 = nn.Linear(hidden_size * 2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        x: (batch_size, 3, 24)
        returns: (batch_size, 1) raw logits
        """
        # lstm_out: (N, seq_len, 2*hidden_size)
        lstm_out, _ = self.lstm(x)

        # Calculate attention weights
        # attn_scores: (N, seq_len, 1)
        attn_scores = self.attention_weights(lstm_out)
        # attn_weights: (N, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Context vector: weighted sum over sequence length
        # (N, 2*hidden_size)
        context = torch.sum(lstm_out * attn_weights, dim=1)

        # Classify
        out = self.dropout(context)
        out = F.relu(self.fc1(out))
        logits = self.fc2(out)

        return logits

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

