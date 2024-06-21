from loguru import logger
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from xares.audio_encoder_base import AudioEncoderBase


class Wav2vec2Encoder(AudioEncoderBase):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def __call__(self, audio, sampling_rate):
        # TODO: implement the audio encoding
        pass
