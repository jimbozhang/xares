from typing import Any

import torch
from loguru import logger


def check_audio_encoder(encoder: Any):
    if not hasattr(encoder, "device"):
        logger.error("Encoder must have a 'device' attribute")
        return False

    if not hasattr(encoder, "output_dim"):
        logger.error("Encoder must have a 'output_dim' attribute")
        return False

    sample_audio = torch.randn(2, 50000)
    try:
        encoded_audio = encoder(sample_audio, 16000)
    except Exception as e:
        logger.error(f"Failed to encode the sample audio: {e}")
        return False

    if not isinstance(encoded_audio, torch.Tensor):
        logger.error(f"Expected tensor for encoded_audio, got {type(encoded_audio)}")
        return False

    if encoded_audio.ndim != 3:  # [B, T, D]
        logger.error(f"Expected 3D tensor [B, T, D] for encoded_audio, got {encoded_audio.dim()}D tensor")
        return False

    logger.info("Encoder check passed.")
    return True
