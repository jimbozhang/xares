from dataclasses import dataclass

import torch

from xares.audio_encoder_base import AudioEncoderBase
from example.ced.ced.audiotransformer_split import CEDConfig, AudioTransformer


@dataclass
class CedEncoder(AudioEncoderBase):
    ced_config = CEDConfig()
    ced_checkpoint = torch.load('pretrained_models/audio_encoder/audiotransformer_base_mAP_4999.pt')
    model = AudioTransformer(ced_config, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4)
    model.load_state_dict(ced_checkpoint, strict=False)
    sampling_rate = 16_000 # model sr
    output_dim = 768

    def __call__(self, audio, sampling_rate = 44_100): # dataset sr
        # Since the "dasheng" model is already in the required in/out format, we directly use the super class method
        return super().__call__(audio, sampling_rate)
