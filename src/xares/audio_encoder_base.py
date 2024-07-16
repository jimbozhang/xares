from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio
from accelerate import Accelerator
from loguru import logger


@dataclass
class AudioEncoderBase(ABC):
    model = None
    min_duration = 0.1
    sampling_rate = 16_000
    output_dim = 0
    resample_warned = False
    device = 'cuda:1'#Accelerator().device

    def __post_init__(self):
        self.model.to(self.device)
        self.model.eval()

    def pre_process_audio(self, audio: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        if not self.check_input_audio(audio, sampling_rate):
            raise ValueError("Invalid input audio")

        audio = self.resample_audio_if_needed(audio, ori_sr=sampling_rate, target_sr=self.sampling_rate)
        return audio.to(self.device)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            encoded_audio = self.model(audio)

        return encoded_audio

    @abstractmethod
    def __call__(self, audio: torch.Tensor, sampling_rate: int) -> Optional[torch.Tensor]:
        assert self.model is not None, "Model is not defined"

        encoded_audio = self.encode_audio(self.pre_process_audio(audio, sampling_rate))

        if not self.check_encoded_audio(encoded_audio):
            raise ValueError("Invalid encoded audio")

        return encoded_audio

    def resample_audio_if_needed(self, audio: torch.Tensor, ori_sr: int, target_sr: int):
        if ori_sr == target_sr:
            return audio

        if not self.resample_warned:
            logger.warning(f"Resample from {ori_sr} to {target_sr}.")
            self.resample_warned = True
        return torchaudio.functional.resample(audio, int(ori_sr), int(target_sr))

    def check_input_audio(self, audio: torch.Tensor, sampling_rate: int):
        if audio.dim() != 2:  # [B, T]
            logger.error(f"Expected 2D tensor [B, T] for audio, got {audio.dim()}D tensor")
            return False
        if audio.size(1) < sampling_rate * self.min_duration:
            logger.error(f"Audio duration is too short: {audio.size(1)} < {int(sampling_rate * self.min_duration)}")
            return False
        return True

    @classmethod
    def check_encoded_audio(self, encoded_audio: torch.Tensor):
        if encoded_audio.dim() != 3:  # [B, T, D]
            logger.error(f"Expected 3D tensor [B, T, D] for encoded_audio, got {encoded_audio.dim()}D tensor")
            return False
        return True
