import csv
import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from webdataset import TarWriter, WebLoader

from xares.audiowebdataset import create_rawaudio_webdataset, write_audio_tar
from xares.models import Mlp
from xares.task_base import TaskBase
from xares.utils import download_file, mkdir_if_not_exists, unzip_file


@dataclass
class VocalSoundTask(TaskBase):
    splits = ["test", "valid", "train"]
    save_encoded_per_batches = 1000  # If OOM, reduce this number
    batch_size = 32
    trim_length = 80_000  # 16k * 5s
    output_dim = 6
    metric = "accuracy"

    def __post_init__(self):
        self.ori_data_root = self.env_dir / "vocalsound"
        self.wds_audio_paths_dict = {split: self.env_dir / f"wds-audio-{split}-*.tar" for split in self.splits}
        self.wds_encoded_paths_dict = {split: self.env_dir / f"wds-encoded-{split}-*.tar" for split in self.splits}
        self.model = Mlp(in_features=self.encoder.output_dim, out_features=self.output_dim).to(self.encoder.device)
        self.checkpoint_dir = self.env_dir / "checkpoints"
        self.ckpt_name = "best_model.pt"
        self.ckpt_path = self.checkpoint_dir / self.ckpt_name
        self.target_to_index = {
            "laughter": 0,
            "sigh": 1,
            "throatclearing": 2,
            "cough": 3,
            "sneeze": 4,
            "sniff": 5,
        }
        self.spk_to_split = {}

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

        self.spk_to_split.update({row[0]: "train" for row in csv.reader(open(train_csv_path, "r", encoding="utf-8"))})
        self.spk_to_split.update({row[0]: "valid" for row in csv.reader(open(valid_csv_path, "r", encoding="utf-8"))})
        self.spk_to_split.update({row[0]: "test" for row in csv.reader(open(test_csv_path, "r", encoding="utf-8"))})

        with open(self.ori_data_root / "meta/vocalsound.csv", mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["filename", "split", "target"])

            for filename in os.listdir(self.ori_data_root / "audio_16k"):
                if (self.ori_data_root / "audio_16k" / filename).is_file():
                    spk, *_, target = filename.split("_")
                    split = self.spk_to_split[spk]
                    index = self.target_to_index[os.path.splitext(target)[0]]
                    writer.writerow([filename, split, index])

        # Create tar file with audio files
        df = pd.read_csv(self.ori_data_root / "meta/vocalsound.csv", usecols=["filename", "split", "target"])
        df.filename = df.filename.apply(lambda x: (self.ori_data_root / "audio_16k" / x).as_posix())

        assert set(df.split.unique().tolist()) == set(self.splits)
        for split in self.splits:
            wds_audio_path = self.wds_audio_paths_dict[split]
            df_split = df[df.split == split].drop(columns="split")
            write_audio_tar(
                df_split.filename.tolist(),
                df_split.target.tolist(),
                wds_audio_path.as_posix(),
                force=self.force_generate_audio_tar,
            )

        self.audio_tar_ready_file.touch()

    def make_encoded_tar(self, num_shards: int = 20):
        def write_encoded_batches_to_wds(encoded_batches: List, ostream: TarWriter):

            for batch, label, keys in encoded_batches:
                for example, label, key in zip(batch, label, keys):
                    sample = {
                        "pth": example,
                        "json": json.dumps({"target": label["label"]}).encode("utf-8"),
                        "__key__": key,
                    }
                    ostream.write(sample)

        for split in self.splits:
            if not self.force_generate_encoded_tar and self.encoded_tar_ready_file.exists():
                logger.info(f"Skip making encoded tar. {self.encoded_tar_ready_file} already exists.")
                continue

            logger.info(f"Encoding audio for {split} ...")
            for shard in range(num_shards):
                sharded_tar_path = self.wds_audio_paths_dict[split].as_posix().replace("*", f"0{shard:05d}")
                dl = create_rawaudio_webdataset(
                    [sharded_tar_path],
                    batch_size=self.batch_size,
                    num_workers=self.num_encoder_workers,
                    crop_size=self.trim_length,
                    with_json=True,
                )

                batch_buf = []
                sharded_encoded_tar_path = self.wds_encoded_paths_dict[split].as_posix().replace("*", f"0{shard:05d}")
                with TarWriter(sharded_encoded_tar_path) as ostream:
                    for batch, length, label, keys in dl:
                        encoded_batch = self.encoder(batch, 16_000)
                        batch_buf.append([encoded_batch, label, keys])

                        if len(batch_buf) >= self.save_encoded_per_batches:
                            write_encoded_batches_to_wds(batch_buf, ostream)
                            batch_buf.clear()
                    if len(batch_buf) > 0:
                        write_encoded_batches_to_wds(batch_buf, ostream)

        self.encoded_tar_ready_file.touch()

    def run_all(self):
        self.make_audio_tar()
        self.make_encoded_tar()

        self.train_mlp(
            [self.wds_encoded_paths_dict["train"].as_posix()],
            [self.wds_encoded_paths_dict["valid"].as_posix()],
        )

        acc = self.evaluate_mlp([self.wds_encoded_paths_dict["test"].as_posix()], metric=self.metric, load_ckpt=True)

        logger.info(f"The accuracy: {acc}")

        return acc
