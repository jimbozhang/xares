import os
from typing import Callable

import torch
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        out_features: int | None = None,
        criterion: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "CrossEntropyLoss",
    ):
        super().__init__()
        out_features = out_features or in_features
        self.ln = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, out_features)
        self.criterion = getattr(nn, criterion)() if isinstance(criterion, str) else criterion

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, return_loss: bool = False):
        x = self.ln(x)
        x = self.fc(x)
        if y is not None and return_loss:
            return self.criterion(x, y)
        return x, y
