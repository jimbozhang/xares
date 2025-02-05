from pathlib import Path

import torch
from transformers import WhisperModel, WhisperProcessor


class WhisperEncoder(torch.nn.Module):
    def __init__(self, model_name="openai/whisper-base"):
        super().__init__()

        if not Path(model_name).exists():
            self.download_from_hub(model_name)

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name).get_encoder()

        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        self.output_dim = self.model.config.d_model
        self.hop_size_in_ms = self.processor.feature_extractor.hop_length

    @classmethod
    def download_from_hub(self, model_name: str, output_root: str = "."):
        # Saving to local to avoid multiprocessing issues
        output_dir = Path(output_root) / model_name
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name)
        self.processor.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)

    def forward(self, audio: torch.Tensor):
        assert isinstance(audio, torch.Tensor)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        audio = audio.cpu().numpy()

        # Feature extraction
        features = self.processor(audio, sampling_rate=self.sampling_rate, return_tensors="pt")

        # Model forward pass
        features["input_features"] = features["input_features"].to(self.model.device)
        return self.model(**features).last_hidden_state


if __name__ == "__main__":
    from xares.audio_encoder_checker import check_audio_encoder

    encoder = WhisperEncoder()
    assert check_audio_encoder(encoder)
