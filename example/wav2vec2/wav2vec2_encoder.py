from pathlib import Path

import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class Wav2vec2Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sampling_rate = 16000
        self.model_name = "facebook/wav2vec2-large-100k-voxpopuli"
        self.output_dim = 1024
        self.hop_size_in_ms = 20
        self.max_length = int(10 * self.sampling_rate)

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

    def forward(self, audio: torch.Tensor):
        assert isinstance(audio, torch.Tensor)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[-1] > self.max_length:
            output = []
            for chunk in audio.split(self.max_length, dim=-1):
                if chunk.shape[-1] < self.sampling_rate:
                    chunk = torch.nn.functional.pad(chunk, (0, self.sampling_rate - chunk.shape[-1]))

                input_values = self.feature_extractor(
                    chunk, sampling_rate=self.sampling_rate, return_tensors="pt"
                ).input_values.squeeze()

                if input_values.dim() == 1:
                    input_values = input_values.unsqueeze(0)

                tmp_output = self.model(input_values.to(self.model.device))["last_hidden_state"]
                output.append(tmp_output)
            output = torch.cat(output, dim=1)
        else:
            input_values = self.feature_extractor(
                audio, sampling_rate=self.sampling_rate, return_tensors="pt"
            ).input_values.squeeze()

            if input_values.dim() == 1:
                input_values = input_values.unsqueeze(0)

            output = self.model(input_values.to(self.model.device))["last_hidden_state"]

        return output


if __name__ == "__main__":
    from xares.audio_encoder_checker import check_audio_encoder

    encoder = Wav2vec2Encoder()
    assert check_audio_encoder(encoder)
