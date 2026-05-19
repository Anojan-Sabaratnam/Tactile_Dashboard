"""3-Finger Specialist Ensemble for tactile grasp stability prediction.

Mirrors the MATLAB champion model from step_4_retune_fused_model.m:
three independent 24→64→32→1 NNs, one per finger, whose slip probabilities
are averaged at inference time.

Input shape: (N, 3, 24)  — same layout as TactileBiLSTM
Output:      (N, 1)      — logit of the averaged slip probability

Why specialisation beats a single fused model:
- Each specialist learns only its finger's contact geometry — no cross-finger
  interference during training.
- Averaging prevents one dominant finger from masking subtle signals in the
  others (e.g. early thumb slip while index/middle still secure).
- Per-finger z-score normalisation (applied upstream) removes inter-finger
  baseline bias — a critical physical correction the MATLAB pipeline validated.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FingerSpecialist(nn.Module):
    """Single-finger binary classifier: 24 → 64 → 32 → 1 logit."""

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(24, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 24) → (N, 1) logits."""
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


class TactileSpecialistEnsemble(nn.Module):
    """Three-finger specialist ensemble.

    Trains three FingerSpecialist networks independently (one per finger),
    averages their sigmoid probabilities, and returns the logit of that
    average — keeping BCEWithLogitsLoss compatibility with the shared
    training loop.
    """

    FINGER_NAMES = ["index", "middle", "thumb"]

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        self.specialists = nn.ModuleList([
            FingerSpecialist(dropout_rate) for _ in range(3)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, 24) → (N, 1) logit of averaged slip probability."""
        probs = []
        for i, specialist in enumerate(self.specialists):
            probs.append(torch.sigmoid(specialist(x[:, i, :])))   # (N, 1)

        avg_prob = torch.stack(probs, dim=0).mean(dim=0)           # (N, 1)
        avg_prob = avg_prob.clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(avg_prob)

    def finger_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-finger slip probabilities as (N, 3), eval mode only."""
        probs = [torch.sigmoid(spec(x[:, i, :])) for i, spec in enumerate(self.specialists)]
        return torch.cat(probs, dim=1)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
