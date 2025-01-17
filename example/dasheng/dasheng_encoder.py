import torch
import torchaudio
from dasheng import dasheng_base


class DashengEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dim = 768

        self.required_sampling_rate = 16000
        self.model = dasheng_base().to(self.device)

    def resample_audio_if_needed(self, audio: torch.Tensor, ori_sr: int, target_sr: int):
        if ori_sr == target_sr:
            return audio
        return torchaudio.functional.resample(audio, int(ori_sr), int(target_sr))

    def __call__(self, audio, sampling_rate: int | None = None):
        if sampling_rate is not None:
            audio = self.resample_audio_if_needed(audio, sampling_rate, self.required_sampling_rate)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        self.model.eval()
        with torch.inference_mode():
            encoded_audio = self.model(audio.to(self.device))

        return encoded_audio


if __name__ == "__main__":
    from xares.audio_encoder_checker import check_audio_encoder

    encoder = DashengEncoder()
    assert check_audio_encoder(encoder)
