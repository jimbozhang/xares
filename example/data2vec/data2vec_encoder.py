from pathlib import Path

import torch.nn as nn
from transformers import AutoFeatureExtractor, Data2VecAudioModel


class Data2VecEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sampling_rate = 16000
        self.model_name = "facebook/data2vec-audio-base"
        self.output_dim = 768
        self.hop_size_in_ms = 20

        if not Path(self.model_name).exists():
            self.download_from_hub(self.model_name)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = Data2VecAudioModel.from_pretrained(self.model_name)

    @classmethod
    def download_from_hub(self, model_name: str, output_root: str = "."):
        # Saving to local to avoid multiprocessing issues
        output_dir = Path(output_root) / model_name
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Data2VecAudioModel.from_pretrained(model_name)
        self.processor.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)

    def forward(self, audio):
        input_values = self.feature_extractor(
            audio, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_values.squeeze()

        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)

        return self.model(input_values.to(self.model.device))["last_hidden_state"]


if __name__ == "__main__":
    from xares.audio_encoder_checker import check_audio_encoder

    encoder = Data2VecEncoder()
    assert check_audio_encoder(encoder)
