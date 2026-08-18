"""Small 1D CNN + tabular fusion head for Zmodel."""

from __future__ import annotations

import torch
import torch.nn as nn


class ZNetV01(nn.Module):
    def __init__(self, tab_dim: int = 12):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.tab = nn.Sequential(
            nn.Linear(tab_dim, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(seq.transpose(1, 2))
        hidden = hidden * mask.unsqueeze(1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = hidden.sum(dim=-1) / denom
        fused = torch.cat([pooled, self.tab(tab)], dim=-1)
        return self.head(fused).squeeze(-1)
