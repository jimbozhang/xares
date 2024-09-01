import subprocess
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from xares.audiowebdataset import write_audio_tar
from xares.models import Mlp
from xares.task_base import TaskBase
from xares.utils import mkdir_if_not_exists


@dataclass
class CREMADTask(TaskBase):
    trim_length = 96_000
    output_dim = 6
    batch_size_train = 16
    learning_rate = 1e-3
    epochs = 20

    def __post_init__(self):
        self.ori_data_root = self.env_dir / "CREMA-D"
        self.model = Mlp(in_features=self.encoder.output_dim, out_features=self.output_dim)
        self.checkpoint_dir = self.env_dir / "checkpoints"
        self.wds_audio_paths_dict = {split: self.env_dir / f"wds_audio_{split}-*.tar" for split in self.splits}
        self.wds_encoded_paths_dict = {split: self.env_dir / f"wds_encoded_{split}-*.tar" for split in self.splits}

    def make_audio_tar(self):
        if not self.force_generate_audio_tar and self.audio_tar_ready_file.exists():
            logger.info(f"Skip making audio tar. {self.audio_tar_ready_file} already exists.")
            return

        # Download CREMA-D dataset
        mkdir_if_not_exists(self.env_dir)
        if not self.ori_data_root.exists():
            git_lfs_command = [
                "git",
                "lfs",
                "clone",
                "https://github.com/CheyneyComputerScience/CREMA-D.git",
                self.ori_data_root,
            ]
            try:
                subprocess.run(git_lfs_command, check=True)
                logger.info(f"Repository cloned to {self.ori_data_root}")
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Error cloning repository: {e}. "
                    f"You may want to 'git lfs clone https://github.com/CheyneyComputerScience/CREMA-D.git' "
                    f"to clone the dataset to {self.ori_data_root} manually."
                )
        else:
            logger.info(f"Directory {self.ori_data_root} already exists. Skip.")

        # Create tar file with audio files
        df = pd.read_csv(self.ori_data_root / "finishedResponses.csv", usecols=["clipName", "dispEmo", "queryType"])
        df = df[df["queryType"] == 1].drop(columns=["queryType"])  # Keep data with audio feedback only
        df = df.drop_duplicates(subset="clipName")
        emotion_map = {"H": 0, "S": 1, "A": 2, "F": 3, "D": 4, "N": 5}
        df["dispEmo"] = df["dispEmo"].apply(lambda x: emotion_map[x])
        df.clipName = df.clipName.apply(lambda x: (self.ori_data_root / "AudioWAV" / f"{x}.wav").as_posix())
        df = df.sample(frac=1).reset_index(drop=True)

        # 15% for test；15% for valid；70% for train
        test_size = int(len(df) * 0.15)
        split_df_dic = {
            "test": df.iloc[:test_size],
            "valid": df.iloc[test_size : 2 * test_size],
            "train": df.iloc[2 * test_size :],
        }
        for split in self.splits:
            df_split = split_df_dic[split]
            write_audio_tar(
                audio_paths=df_split.clipName.tolist(),
                labels=df_split.dispEmo.tolist(),
                tar_path=self.wds_audio_paths_dict[split].as_posix(),
                force=self.force_generate_audio_tar,
                num_shards=self.num_shards_rawaudio,
            )

        self.audio_tar_ready_file.touch()
