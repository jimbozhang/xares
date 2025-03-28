from pathlib import Path

import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class Wav2vec2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sampling_rate = 16000
        self.model_name = "facebook/wav2vec2-large-100k-voxpopuli"
        self.output_dim = 1024
        self.hop_size_in_ms = 20

        if not Path(self.model_name).exists():
            self.download_from_hub(self.model_name)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)

    @classmethod
    def download_from_hub(self, model_name: str, output_root: str = "."):
        # Saving to local to avoid multiprocessing issues
        output_dir = Path(output_root) / model_name
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.processor.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)

    def forward(self, audio):
        input_values = self.feature_extractor(
            audio, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_values.squeeze()

        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)

        return self.model(input_values.to(self.model.device))["last_hidden_state"]
