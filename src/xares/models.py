import torch.nn as nn
import torch


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 out_features: int | None = None,
                 criterion="CrossEntropyLoss"):
        super().__init__()
        out_features = out_features or in_features
        self.ln = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, out_features)
        self.criterion = getattr(nn, criterion)()

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor | None = None,
                return_loss: bool = False):
        x = self.ln(x)
        x = self.fc(x)
        if y is not None and return_loss:
            return self.criterion(x, y)
        return x, y
