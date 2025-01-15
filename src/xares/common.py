from dataclasses import dataclass


@dataclass()
class XaresSettings:
    env_root: str = "./env"
    audio_ready_filename: str = ".audio_tar_ready"
    encoded_ready_filename: str = ".encoded_tar_ready"
