import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.ln = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        return x
