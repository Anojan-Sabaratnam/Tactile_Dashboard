"""2D Convolutional Neural Network for tactile grasp stability prediction.

Treats the 24 taxels of each finger as a 4x6 spatial image.
The 3 fingers (index, middle, thumb) are stacked as 3 channels.
Input shape: (N, 3, 4, 6)

Design rationale:
- Explores spatial correlations: adjacent taxels on the sensor surface
  are processed together by the 3x3 convolution filters.
- Learns hierarchical features automatically, contrasting with the
  Random Forest's handcrafted features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TactileCNN2D(nn.Module):
    """2D CNN for 3-channel (finger) 4x6 tactile grids."""

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()

        # Conv Block 1: (N, 3, 4, 6) -> (N, 32, 4, 6)
        # padding="same" keeps spatial dimensions
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv Block 2: (N, 32, 4, 6) -> (N, 64, 4, 6)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Global Average Pooling: (N, 64, 4, 6) -> (N, 64)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        x: (batch_size, 3, 4, 6)
        returns: (batch_size, 1) raw logits
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        logits = self.fc(x)

        return logits

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

