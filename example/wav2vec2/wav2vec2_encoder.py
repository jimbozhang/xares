from loguru import logger
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class Wav2vec2Encoder:
    def __init__(self):
        self.model_name = "facebook/wav2vec2-large-100k-voxpopuli"
        self.cache_dir = "pretrained/"
        self.output_dim = 1024
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name, cache_dir=self.cache_dir)

    def __call__(self, audio, sampling_rate):
        if sampling_rate != 16000:
            logger.warning(f"Unmatched sampling rate: {sampling_rate}")

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
