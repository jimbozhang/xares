import csv
import os
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from xares.audiowebdataset import write_audio_tar
from xares.models import Mlp
from xares.task_base import TaskBase
from xares.utils import download_file, mkdir_if_not_exists, unzip_file


@dataclass
class VocalSoundTask(TaskBase):
    trim_length = 80_000  # 16k * 5s
    output_dim = 6

    def __post_init__(self):
        self.ori_data_root = self.env_dir / "vocalsound"
        self.audio_tar_name_of_split = {split: self.env_dir / f"wds-audio-{split}-*.tar" for split in self.splits}
        self.encoded_tar_path_of_split = {split: self.env_dir / f"wds-encoded-{split}-*.tar" for split in self.splits}
        self.model = Mlp(in_features=self.encoder.output_dim, out_features=self.output_dim).to(self.encoder.device)
        self.ckpt_dir = self.env_dir / "checkpoints"

    def make_audio_tar(self):
        if not self.force_generate_audio_tar and self.audio_tar_ready_file.exists():
            logger.info(f"Skip making audio tar. {self.audio_tar_ready_file} already exists.")
            return

        mkdir_if_not_exists(self.env_dir)
        download_file(
            "https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1",
            self.env_dir / "vs_release_16k.zip",
            force=self.force_download,
        )
        if not self.ori_data_root.exists():
            logger.info(f"Extracting {self.env_dir / 'vs_release_16k.zip'} to {self.ori_data_root}...")
            unzip_file(self.env_dir / "vs_release_16k.zip", self.ori_data_root)
        else:
            logger.info(f"Directory {self.ori_data_root} already exists. Skip.")

        train_csv_path = self.ori_data_root / "meta/tr_meta.csv"
        valid_csv_path = self.ori_data_root / "meta/val_meta.csv"
        test_csv_path = self.ori_data_root / "meta/te_meta.csv"

        spk_to_split = {}
        spk_to_split.update({row[0]: "train" for row in csv.reader(open(train_csv_path, "r", encoding="utf-8"))})
        spk_to_split.update({row[0]: "valid" for row in csv.reader(open(valid_csv_path, "r", encoding="utf-8"))})
        spk_to_split.update({row[0]: "test" for row in csv.reader(open(test_csv_path, "r", encoding="utf-8"))})

        target_to_index = {
            "laughter": 0,
            "sigh": 1,
            "throatclearing": 2,
            "cough": 3,
            "sneeze": 4,
            "sniff": 5,
        }

        with open(self.ori_data_root / "meta/vocalsound.csv", mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["filename", "split", "target"])

            for filename in os.listdir(self.ori_data_root / "audio_16k"):
                if (self.ori_data_root / "audio_16k" / filename).is_file():
                    spk, *_, target = filename.split("_")
                    split = spk_to_split[spk]
                    index = target_to_index[os.path.splitext(target)[0]]
                    writer.writerow([filename, split, index])

        # Create tar file with audio files
        df = pd.read_csv(self.ori_data_root / "meta/vocalsound.csv", usecols=["filename", "split", "target"])
        df.filename = df.filename.apply(lambda x: (self.ori_data_root / "audio_16k" / x).as_posix())

        assert set(df.split.unique().tolist()) == set(self.splits)
        for split in self.splits:
            df_split = df[df.split == split].drop(columns="split")
            write_audio_tar(
                audio_paths=df_split.filename.tolist(),
                labels=df_split.target.tolist(),
                tar_path=self.audio_tar_name_of_split[split].as_posix(),
                force=self.force_generate_audio_tar,
            )

        self.audio_tar_ready_file.touch()
