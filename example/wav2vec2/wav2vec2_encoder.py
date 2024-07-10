from dataclasses import dataclass

from loguru import logger
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from xares.audio_encoder_base import AudioEncoderBase


@dataclass
class Wav2vec2Encoder(AudioEncoderBase):
    output_dim = 768

    def __post_init__(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        super().__post_init__()

    def __call__(self, audio, sampling_rate):
        input_values = (
            self.feature_extractor(
                self.pre_process_audio(audio, sampling_rate), sampling_rate=self.sampling_rate, return_tensors="pt"
            )
            .input_values.squeeze()
            .to(self.device)
        )

        encoded_audio = self.encode_audio(input_values)["last_hidden_state"]

        if not self.check_encoded_audio(encoded_audio):
            raise ValueError("Invalid encoded audio")

        return encoded_audio
