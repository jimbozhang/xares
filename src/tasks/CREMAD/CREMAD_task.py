import json
import subprocess
from dataclasses import dataclass
from typing import List

import pandas as pd
from loguru import logger
from tqdm import tqdm
from webdataset import TarWriter, WebLoader

from xares.audiowebdataset import Audiowebdataset_Fluid, proxy_read
from xares.models import Mlp
from xares.task_base import TaskBase
from xares.utils import mkdir_if_not_exists


@dataclass
class CREMADTask(TaskBase):
    splits = ["test", "valid", "train"]
    save_encoded_per_batches = 1000  # If OOM, reduce this number
    batch_size = 32
    trim_length = 96_000
    output_dim = 6
    metric = "accuracy"

    def __post_init__(self):
        self.ori_data_root = self.env_dir / "CREMA-D"
        self.model = Mlp(in_features=self.encoder.output_dim, out_features=self.output_dim)
        self.checkpoint_dir = self.env_dir / "checkpoints"

        self.wds_audio_paths_dict = {
            "test": self.env_dir / "wds_audio_test.tar",
            "valid": self.env_dir / "wds_audio_valid.tar",
            "train": self.env_dir / "wds_audio_train.tar",
        }

        self.wds_encoded_paths_dict = {
            "test": self.env_dir / "wds_encoded_test.tar",
            "valid": self.env_dir / "wds_encoded_valid.tar",
            "train": self.env_dir / "wds_encoded_train.tar",
        }

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
                    f"Please use the command 'git lfs clone https://github.com/CheyneyComputerScience/CREMA-D.git' "
                    f"to download the files to {self.ori_data_root}"
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
            split_df = split_df_dic[split]
            audio_path = self.wds_audio_paths_dict[split]
            if not self.force_generate_audio_tar and audio_path.exists():
                logger.info(f"Tar file {audio_path} already exists.")
            else:
                with TarWriter(audio_path.as_posix()) as ostream:
                    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
                        sample = proxy_read(row.to_dict(), "clipName")
                        ostream.write(sample)

        self.audio_tar_ready_file.touch()

    def make_encoded_tar(self):
        def write_encoded_batches_to_wds(encoded_batches: List, ostream: TarWriter, identifier: str = None):
            if identifier is not None:
                logger.info(f"Writing encoded batches for {identifier} ...")

            for batch, label, keys in encoded_batches:
                for example, label, key in zip(batch, label["dispEmo"], keys):
                    sample = {
                        "pth": example,
                        "json": json.dumps({"target": label.item()}).encode("utf-8"),
                        "__key__": key,
                    }
                    ostream.write(sample)

        for split in self.splits:
            audio_path = self.wds_audio_paths_dict[split]
            encoded_path = self.wds_encoded_paths_dict[split]
            if not self.force_generate_encoded_tar and encoded_path.exists():
                logger.info(f"Tar file {encoded_path} already exists.")
                continue

            ds = Audiowebdataset_Fluid(
                [audio_path.as_posix()],
                crop_size=self.trim_length,
                with_json=True,
            )
            dl = WebLoader(ds, batch_size=self.batch_size, num_workers=self.num_encoder_workers)

            logger.info(f"Encoding audio for {split} datafile ...")
            batch_buf = []
            with TarWriter(encoded_path.as_posix()) as ostream:
                for batch, label, keys in tqdm(dl):
                    encoded_batch = self.encoder(batch, 16_000)
                    batch_buf.append([encoded_batch, label, keys])

                    if len(batch_buf) >= self.save_encoded_per_batches:
                        write_encoded_batches_to_wds(batch_buf, ostream, identifier=split)
                        batch_buf.clear()
                if len(batch_buf) > 0:
                    write_encoded_batches_to_wds(batch_buf, ostream, identifier=split)

    def run_all(self):
        self.make_audio_tar()
        self.make_encoded_tar()

        self.ckpt_name = "best_model.pt"
        self.ckpt_path = self.checkpoint_dir / self.ckpt_name

        self.train_mlp(
            [self.wds_encoded_paths_dict["train"].as_posix()],
            [self.wds_encoded_paths_dict["valid"].as_posix()],
        )
        acc = self.evaluate_mlp([self.wds_encoded_paths_dict["test"].as_posix()], metric=self.metric, load_ckpt=True)
        logger.info(f"Accuracy: {acc}")
