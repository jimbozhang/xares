import copy
import json
import copy
import numpy as np
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from dasheng.prepare.wavlist_to_tar import proxy_read
from dasheng.train.audiowebdataset import Audiowebdataset_Fluid
from dasheng.train.models import Mlp
from loguru import logger
from tqdm import tqdm
from webdataset import TarWriter, WebLoader

from xares.task_base import TaskBase
from xares.utils import download_file, mkdir_if_not_exists, unzip_file


@dataclass
class ESC50Task(TaskBase):
    folds = range(1, 6)  # This dataset requires 5-fold validation in evaluation
    save_encoded_per_batches = 1000  # If OOM, reduce this number
    batch_size = 32
    trim_length = 220_500
    output_dim = 50
    metric = "accuracy"

    def __post_init__(self):
        self.ori_data_root = self.env_dir / "ESC-50-master"
        self.wds_audio_paths_dict = {fold: self.env_dir / f"wds-audio-fold-0{fold}.tar" for fold in self.folds}
        self.wds_encoded_paths_dict = {fold: self.env_dir / f"wds-encoded-fold-0{fold}.tar" for fold in self.folds}
        self.model = Mlp(in_features=self.encoder.output_dim, out_features=self.output_dim).to(self.encoder.device)
        self.checkpoint_dir = self.env_dir / "checkpoints"

        self.wds_encoded_training_fold_k = {
            fold: [f"{self.env_dir}/wds-encoded-fold-0{f}.tar" for f in self.folds if f != fold] for fold in self.folds
        }

    def make_audio_tar(self):
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

        assert df.fold.unique().tolist() == list(self.folds)
        for fold in self.folds:
            wds_audio_path = self.wds_audio_paths_dict[fold]

            if not self.force_generate_audio_tar and wds_audio_path.exists():
                logger.info(f"Tar file {wds_audio_path} already exists.")
                continue

            df_split = df[df.fold == fold].drop(columns=["fold"])
            with TarWriter(wds_audio_path.as_posix()) as ostream:
                for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
                    sample = proxy_read(row.to_dict(), "filename")
                    ostream.write(sample)

    def make_encoded_tar(self):
        def write_encoded_batches_to_wds(encoded_batches: List, ostream: TarWriter, identifier: str = None):
            if identifier is not None:
                logger.info(f"Writing encoded batches for {identifier} ...")

            for batch, label, keys in encoded_batches:
                for example, label, key in zip(batch, label["target"], keys):
                    sample = {
                        "pth": example,
                        "json": json.dumps({"target": label.item()}).encode("utf-8"),
                        "__key__": key,
                    }
                    ostream.write(sample)

        for split in self.wds_audio_paths_dict:
            wds_encoded_path = self.wds_encoded_paths_dict[split]
            if not self.force_generate_encoded_tar and wds_encoded_path.exists():
                logger.info(f"Tar file {wds_encoded_path} already exists.")
                continue

            ds = Audiowebdataset_Fluid(
                [self.wds_audio_paths_dict[split].as_posix()],
                crop_size=self.trim_length,
                drop_crops=True,
                with_json=True,
            )
            dl = WebLoader(ds, batch_size=self.batch_size, num_workers=1)

            logger.info(f"Encoding audio for fold {split} ...")
            batch_buf = []
            with TarWriter(wds_encoded_path.as_posix()) as ostream:
                for batch, label, keys in tqdm(dl):
                    encoded_batch = self.encoder(batch, 44_100)
                    batch_buf.append([encoded_batch, label, keys])

                    if len(batch_buf) >= self.save_encoded_per_batches:
                        write_encoded_batches_to_wds(batch_buf, ostream, identifier=f"fold-{split}")
                        batch_buf.clear()
                if len(batch_buf) > 0:
                    write_encoded_batches_to_wds(batch_buf, ostream, identifier=f"fold-{split}")

    def run_all(self):
        self.make_audio_tar()
        self.make_encoded_tar()

        # k-fold:
        model = copy.deepcopy(self.model)
        acc = []
        for k in self.folds:
            self.ckpt_name = f"fold_0{k}_best_model.pt"
            self.ckpt_path = self.checkpoint_dir / self.ckpt_name
            self.train_mlp(
                self.wds_encoded_training_fold_k[k],
                [self.wds_encoded_paths_dict[k].as_posix()],
            )
            acc.append(self.evaluate_mlp([self.wds_encoded_paths_dict[k].as_posix()], load_ckpt=True))
            self.model = copy.deepcopy(model).to(self.encoder.device)

        for k in range(len(self.folds)):
            logger.info(f"Fold {k+1} accuracy: {acc[k]}")
        logger.info(f"The averaged accuracy of 5 folds is: {np.mean(acc)}")
