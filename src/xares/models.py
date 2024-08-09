from typing import Optional

import torch.nn as nn

from xares.audio_encoder_base import AudioEncoderBase


class ModelBase(nn.Module):
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def reinit(self):
        self._init_weights()


class Mlp(ModelBase):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Havn't tested this, may not work
# Useful for not freezing the encoder when training the output layer
class AudioEncoderWithMlpOutput(nn.Module):
    def __init__(self, encoder: AudioEncoderBase, output_dim: int, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.output_layer = Mlp(
            in_features=self.encoder.output_dim,
            out_features=output_dim,
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.output_layer(x)
        return x
