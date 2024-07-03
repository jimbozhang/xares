import torch
from loguru import logger
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2Model
from dataclasses import dataclass
from xares.audio_encoder_base import AudioEncoderBase


@dataclass
class Wav2vec2Encoder(AudioEncoderBase):
    def __init__(self):
        super().__init__()
        self.feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.sampling_rate = 16_000 # model sr
        self.output_dim = 768

    def __call__(self, audio, sampling_rate = 44_100): # dataset sr
        if self.model is None:
            return None

        self.model.to(self.device)
        self.model.eval()

        if not self.check_input_audio(audio, sampling_rate):
            raise ValueError("Invalid input audio")

        audio = self.resample_audio_if_needed(audio, ori_sr=sampling_rate, target_sr=self.sampling_rate)

        input_values = self.feat_extractor(audio, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.squeeze()

        with torch.inference_mode():
            encoded_audio = self.model(input_values.to(self.device))['last_hidden_state']

        if not self.check_encoded_audio(encoded_audio):
            raise ValueError("Invalid encoded audio")

        return encoded_audio