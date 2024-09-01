import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from xares.audiowebdataset import write_audio_tar
from xares.models import Mlp
from xares.task_base import TaskBase
from xares.utils import download_file, mkdir_if_not_exists, untar_file


@dataclass
class LibriSpeechMaleFemaleTask(TaskBase):
    trim_length = 33_440
    output_dim = 2
    splits = ["train-clean-100", "dev-clean", "test-clean", "test-other"]

    def __post_init__(self):
        self.ori_data_root = self.env_dir / "LibriSpeech"
        self.wds_audio_paths_dict = {split: self.env_dir / f"wds-audio-{split}-*.tar" for split in self.splits}
        self.wds_encoded_paths_dict = {split: self.env_dir / f"wds-encoded-{split}-*.tar" for split in self.splits}
        self.model = Mlp(in_features=self.encoder.output_dim, out_features=self.output_dim).to(self.encoder.device)
        self.checkpoint_dir = self.env_dir / "checkpoints"

    def make_audio_tar(self):
        if not self.force_generate_audio_tar and self.audio_tar_ready_file.exists():
            logger.info(f"Skip making audio tar. {self.audio_tar_ready_file} already exists.")
            return

        # Download and extract LibriSpeech dataset
        mkdir_if_not_exists(self.env_dir)
        for split in self.splits:
            download_file(
                f"https://www.openslr.org/resources/12/{split}.tar.gz",
                self.env_dir / f"{split}.tar.gz",
                force=self.force_download,
            )

        if not self.ori_data_root.exists():
            for split in self.splits:
                logger.info(f"Extracting {self.env_dir / f'{split}.tar.gz'} to {self.env_dir}...")
                untar_file(self.env_dir / f"{split}.tar.gz", self.env_dir)
        else:
            logger.info(f"Directory {self.ori_data_root} already exists. Skip.")

        # Create tar file with audio files
        with open(self.ori_data_root / "SPEAKERS.TXT", "r") as f:
            id_gender = {
                int(line.split("|")[0]): 1 if line.split("|")[1].strip() == "F" else 0
                for line in f
                if not line.strip().startswith(";")
            }

        for split in self.splits:
            paths = os.walk(self.ori_data_root / split)
            file_paths = []
            speakers = []
            for path, _, file_lst in paths:
                for file_name in file_lst:
                    file_path = Path(os.path.join(path, file_name))
                    if str(file_path).endswith(".flac"):
                        file_paths.append(file_path.as_posix())
                        speaker = int(str(file_path.as_posix()).split("/")[4])
                        speakers.append(speaker)

            audio_speaker = dict(zip(file_paths, speakers))
            targets = [id_gender[key] for key in audio_speaker.values()]
            write_audio_tar(
                audio_paths=file_paths,
                labels=targets,
                tar_path=self.wds_audio_paths_dict[split].as_posix(),
                suffix="flac",
                force=self.force_generate_audio_tar,
            )

        self.audio_tar_ready_file.touch()

    def run_all(self) -> float:
        self.make_audio_tar()
        self.make_encoded_tar()
        self.train_mlp(
            [self.wds_encoded_paths_dict["train-clean-100"].as_posix()],
            [self.wds_encoded_paths_dict["dev-clean"].as_posix()],
        )
        acc_clean = self.evaluate_mlp(
            [self.wds_encoded_paths_dict["test-clean"].as_posix()], metric=self.metric, load_ckpt=True
        )
        logger.info(f"The test_clean accuracy is: {acc_clean}")

        acc_other = self.evaluate_mlp(
            [self.wds_encoded_paths_dict["test-other"].as_posix()], metric=self.metric, load_ckpt=True
        )
        logger.info(f"The test_other accuracy is: {acc_other}")

        return acc_clean
