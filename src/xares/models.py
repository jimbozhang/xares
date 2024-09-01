import torch.nn as nn

from xares.audio_encoder_base import AudioEncoderBase


class ModelBase(nn.Module):
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def reinit(self):
        self._init_weights()


class Mlp(ModelBase):
    def __init__(self, in_features, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.ln = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        return x


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
