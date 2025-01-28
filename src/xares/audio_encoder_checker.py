import torch
from loguru import logger


def check_audio_encoder(encoder: torch.nn.Module):
    if not type(encoder).__name__.endswith("Encoder"):
        logger.error("Class name must end with 'Encoder'")
        return False

    if not isinstance(encoder, torch.nn.Module):
        logger.error(f"Expected torch.nn.Module for encoder, got {type(encoder)}")
        return False

    if not hasattr(encoder, "output_dim"):
        logger.error("Encoder must have a 'output_dim' attribute")
        return False

    if not hasattr(encoder, "sampling_rate"):
        logger.error("Encoder must have a 'sampling_rate' attribute")
        return False

    sample_audio = torch.randn(2, 50000)
    try:
        encoded_audio = encoder(sample_audio)
    except Exception as e:
        logger.error(f"Failed to encode the sample audio: {e}")
        return False

    if not isinstance(encoded_audio, torch.Tensor):
        logger.error(f"Expected tensor for encoded_audio, got {type(encoded_audio)}")
        return False

    if encoded_audio.ndim != 3:  # [B, T, D]
        logger.error(f"Expected 3D tensor [B, T, D] for encoded_audio, got {encoded_audio.dim()}D tensor")
        return False

    if encoded_audio.size(0) != sample_audio.size(0):
        logger.error(f"Expected batch size={sample_audio.size(0)} for encoded_audio, got {encoded_audio.size(0)}")
        return False

    if encoded_audio.size(2) != encoder.output_dim:
        logger.error(f"Expected output_dim={encoder.output_dim} for encoded_audio, got {encoded_audio.size(2)}")
        return False

    return True
