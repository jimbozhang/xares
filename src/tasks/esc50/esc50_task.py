from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from xares.audiowebdataset import write_audio_tar
from xares.models import Mlp
from xares.task_base import TaskBase
from xares.utils import download_file, mkdir_if_not_exists, unzip_file


@dataclass
class ESC50Task(TaskBase):
    splits = range(1, 6)  # This dataset requires 5-fold validation in evaluation
    trim_length = 220_500
    output_dim = 50
    save_encoded_per_batches = 2

    def __post_init__(self):
        self.ori_data_root = self.env_dir / "ESC-50-master"
        self.wds_audio_paths_dict = {fold: self.env_dir / f"wds-audio-fold-{fold}-*.tar" for fold in self.splits}
        self.wds_encoded_paths_dict = {fold: self.env_dir / f"wds-encoded-fold-{fold}-*.tar" for fold in self.splits}
        self.model = Mlp(in_features=self.encoder.output_dim, out_features=self.output_dim).to(self.encoder.device)
        self.checkpoint_dir = self.env_dir / "checkpoints"

    def make_audio_tar(self):
        if not self.force_generate_audio_tar and self.audio_tar_ready_file.exists():
            logger.info(f"Skip making audio tar. {self.audio_tar_ready_file} already exists.")
            return

        # Download and extract ESC-50 dataset
        mkdir_if_not_exists(self.env_dir)
        download_file(
            "https://github.com/karoldvl/ESC-50/archive/master.zip",
            self.env_dir / "master.zip",
            force=self.force_download,
        )
        if not self.ori_data_root.exists():
            logger.info(f"Extracting {self.env_dir / 'master.zip'} to {self.env_dir}...")
            unzip_file(self.env_dir / "master.zip", self.env_dir)
        else:
            logger.info(f"Directory {self.ori_data_root} already exists. Skip.")

        # Create tar file with audio files
        df = pd.read_csv(self.ori_data_root / "meta/esc50.csv", usecols=["filename", "fold", "target"])
        df.filename = df.filename.apply(lambda x: (self.ori_data_root / "audio" / x).as_posix())

        assert df.fold.unique().tolist() == list(self.splits)
        for fold in self.splits:
            df_split = df[df.fold == fold].drop(columns=["fold"])
            write_audio_tar(
                audio_paths=df_split.filename.tolist(),
                labels=df_split.target.tolist(),
                tar_path=self.wds_audio_paths_dict[fold].as_posix(),
                force=self.force_generate_audio_tar,
                num_shards=self.num_shards_rawaudio,
            )

        self.audio_tar_ready_file.touch()

    def run_all(self) -> float:
        self.make_audio_tar()
        self.make_encoded_tar()

        # k-fold:
        acc = []
        wds_encoded_training_fold_k = {
            fold: [f"{self.env_dir}/wds-encoded-fold-{f}-*.tar" for f in self.splits if f != fold]
            for fold in self.splits
        }

        for k in self.splits:
            self.ckpt_path = self.checkpoint_dir / f"fold_{k}_best_model.pt"
            self.model.reinit()
            self.model = self.model.to(self.encoder.device)
            self.train_mlp(
                wds_encoded_training_fold_k[k],
                [self.wds_encoded_paths_dict[k].as_posix()],
            )
            acc.append(
                self.evaluate_mlp([self.wds_encoded_paths_dict[k].as_posix()], metric=self.metric, load_ckpt=True)
            )

        for k in range(len(self.splits)):
            logger.info(f"Fold {k+1} accuracy: {acc[k]}")

        avg_acc = np.mean(acc)
        logger.info(f"The averaged accuracy of 5 folds is: {avg_acc}")

        return avg_acc
