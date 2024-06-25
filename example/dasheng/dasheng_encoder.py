from dataclasses import dataclass

from dasheng import dasheng_base

from xares.audio_encoder_base import AudioEncoderBase


@dataclass
class DashengEncoder(AudioEncoderBase):
    model = dasheng_base()
    sampling_rate = 16_000
    output_dim = 768

    def __call__(self, audio, sampling_rate):
        # Since the "dasheng" model is already in the required in/out format, we directly use the super class method
        return super().__call__(audio, sampling_rate)
