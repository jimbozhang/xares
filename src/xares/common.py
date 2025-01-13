from dataclasses import dataclass
from pathlib import Path


@dataclass()
class XaresSettings:
    env_root: str = "./env"
    audio_ready_filename: str = ".audio_tar_ready"
    encoded_ready_filename: str = ".encoded_tar_ready"
    streaming_wds_default: bool = False  # Not compatible with audiowebdataset.expand_with_brace
