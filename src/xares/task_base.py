from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import ignite.metrics
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from xares.audiowebdataset import create_embedding_webdataset, create_rawaudio_webdataset, write_encoded_batches_to_wds
from xares.common import XaresSettings
from xares.models import Mlp
from xares.trainer import MetricType, Trainer, inference
from xares.utils import download_zenodo_record, mkdir_if_not_exists


@dataclass
class TaskConfig:
    xares_settings: XaresSettings = field(default_factory=XaresSettings)
    env_root: Path | str | None = None

    # Splits
    train_split: None | str = "train"
    valid_split: None | str = "valid"
    test_split: None | str = "test"
    k_fold_splits: None | List[int | str] = None

    # Audio tar
    force_download: bool = False
    force_generate_audio_tar: bool = False
    audio_tar_name_of_split: Dict[Any, Any] = field(default_factory=lambda: dict())
    num_shards_rawaudio = 4
    zenodo_id: str | None = None

    # Encoded tar
    force_encoded: bool = False
    encoder: Any = None
    encoded_tar_name_of_split: Dict[Any, Any] = field(default_factory=lambda: dict())
    trim_length = None
    save_encoded_per_batches: int = 64
    batch_size_encode: int = 16
    num_encoder_workers: int = 0

    # MLP
    force_retrain_mlp: bool = False
    ckpt_dir_name = "checkpoints"
    ckpt_name = "best.ckpt"
    batch_size_train: int = 32
    learning_rate: float = 3e-3
    epochs: int = 10
    num_training_workers: int = 4
    num_validation_workers: int = 4
    model: nn.Module | None = None
    output_dim: int | None = None
    metric = "accuracy"

    def __post_init__(self):
        self.update_tar_name_of_split()
        if self.env_root is None:
            self.env_root = self.xares_settings.env_root

    def update_tar_name_of_split(self):
        self.audio_tar_name_of_split = {
            self.train_split: f"wds-audio-{self.train_split}-*.tar",
            self.valid_split: f"wds-audio-{self.valid_split}-*.tar",
            self.test_split: f"wds-audio-{self.test_split}-*.tar",
        }
        self.encoded_tar_name_of_split = {
            self.train_split: f"wds-encoded-{self.train_split}-*.tar",
            self.valid_split: f"wds-encoded-{self.valid_split}-*.tar",
            self.test_split: f"wds-encoded-{self.test_split}-*.tar",
        }


class TaskBase(ABC):
    def __init__(self, encoder: Any, config: None | TaskConfig = None):
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").setLevel(logging.ERROR)

        self.config = config or TaskConfig()
        self.encoder = encoder
        self.mlp = None

        self.env_dir = Path(self.config.env_root) / self.__class__.__name__.lower().strip("task")
        mkdir_if_not_exists(self.env_dir)
        self.ckpt_dir = self.env_dir / self.config.ckpt_dir_name
        self.ckpt_path = self.ckpt_dir / self.config.ckpt_name

        if self.config.k_fold_splits:
            self.config.train_split = self.config.valid_split = self.config.test_split = None

        self.label_processor = lambda x: x["target"]

    @abstractmethod
    def run(self) -> float:
        pass

    @abstractmethod
    def make_encoded_tar(self):
        pass

    def default_run(self) -> float:
        self.make_encoded_tar()

        self.train_mlp(
            [self.encoded_tar_path_of_split[self.config.train_split].as_posix()],
            [self.encoded_tar_path_of_split[self.config.valid_split].as_posix()],
        )
        acc = self.evaluate_mlp(
            [self.encoded_tar_path_of_split[self.config.test_split].as_posix()],
            metric=self.config.metric,
            load_ckpt=True,
        )
        logger.info(f"The accuracy: {acc}")

        score = acc
        return score

    def default_run_k_fold(self) -> float:
        self.make_encoded_tar()

        acc = []
        splits = self._make_splits()
        wds_encoded_training_fold_k = {
            k: [self.encoded_tar_path_of_split[j].as_posix() for j in splits if j != k] for k in splits
        }

        for k in splits:
            self.config.ckpt_name = f"fold_{k}_best_model.pt"
            self.ckpt_path = self.ckpt_dir / self.config.ckpt_name
            self.train_mlp(
                wds_encoded_training_fold_k[k],
                [self.encoded_tar_path_of_split[k].as_posix()],
            )
            acc.append(
                self.evaluate_mlp(
                    [self.encoded_tar_path_of_split[k].as_posix()], metric=self.config.metric, load_ckpt=True
                )
            )

        for k in range(len(splits)):
            logger.info(f"Fold {k+1} accuracy: {acc[k]}")

        avg_score = np.mean(acc)
        logger.info(f"The averaged accuracy of 5 folds is: {avg_score}")

        return avg_score

    def default_make_encoded_tar(self):
        self.encoded_tar_path_of_split = {
            split: (self.env_dir / self.config.encoded_tar_name_of_split[split])
            for split in self.config.encoded_tar_name_of_split
        }

        encoded_ready_path = self.env_dir / self.config.xares_settings.encoded_ready_filename
        if not self.config.force_encoded and encoded_ready_path.exists():
            logger.info(f"Skip making encoded tar.")
            return

        audio_ready_path = self.env_dir / self.config.xares_settings.audio_ready_filename
        if not audio_ready_path.exists():
            download_zenodo_record(self.config.zenodo_id, self.env_dir, force_download=self.config.force_download)
            audio_ready_path.touch()

        audio_tar_path_of_split = {
            split: (self.env_dir / self.config.audio_tar_name_of_split[split]).as_posix()
            for split in self.config.audio_tar_name_of_split
        }

        for split in audio_tar_path_of_split:
            logger.info(f"Encoding audio for split {split} ...")

            dl = create_rawaudio_webdataset(
                [audio_tar_path_of_split[split]],
                batch_size=self.config.batch_size_encode,
                num_workers=self.config.num_encoder_workers,
                crop_size=self.config.trim_length,
                with_json=True,
            )

            batch_buf = []
            shard = 0
            for batch, _, label, keys in tqdm(dl, desc=f"Encoding {split}"):
                batch = batch.to(self.encoder.device)
                encoded_batch = self.encoder(batch, 44_100).to("cpu").detach()
                batch_buf.append([encoded_batch, label, keys])

                if len(batch_buf) >= self.config.save_encoded_per_batches:
                    write_encoded_batches_to_wds(
                        batch_buf,
                        self.encoded_tar_path_of_split[split].as_posix().replace("*", f"0{shard:05d}"),
                        num_workers=self.config.num_encoder_workers,
                    )
                    batch_buf.clear()
                    shard += 1

            if len(batch_buf) > 0:
                write_encoded_batches_to_wds(
                    batch_buf,
                    self.encoded_tar_path_of_split[split].as_posix().replace("*", f"0{shard:05d}"),
                    num_workers=self.config.num_encoder_workers,
                )
        encoded_ready_path.touch()

    def train_mlp(self, train_url: list, validation_url: list) -> None:
        self.mlp = Mlp(in_features=self.encoder.output_dim, out_features=self.config.output_dim).to(self.encoder.device)

        if not self.config.force_retrain_mlp and self.ckpt_path.exists():
            logger.info(f"Checkpoint {self.ckpt_path} already exists. Skip training.")
            self.mlp.load_state_dict(torch.load(self.ckpt_path))
            return

        trainer = Trainer(
            self.mlp,
            ckpt_dir=self.ckpt_dir,
            ckpt_name=self.config.ckpt_name,
            metric=self.config.metric,
            lr=self.config.learning_rate,
            max_epochs=self.config.epochs,
        )

        dl_train = create_embedding_webdataset(
            train_url,
            tar_shuffle=2000,
            batch_size=self.config.batch_size_train,
            num_workers=self.config.num_training_workers,
            training=True,
            label_processor=self.label_processor,
        )
        dl_val = create_embedding_webdataset(
            validation_url,
            batch_size=self.config.batch_size_train,
            num_workers=self.config.num_validation_workers,
            training=False,
            label_processor=self.label_processor,
        )

        trainer.run(dl_train, dl_val)

    def evaluate_mlp(self, eval_url: list, metric: str = "Accuracy", load_ckpt: bool = False) -> float:
        if self.mlp is None:
            raise ValueError("Train the model first before evaluation.")

        if load_ckpt:
            if self.ckpt_path.exists():
                self.mlp.load_state_dict(torch.load(self.ckpt_path))
                logger.info(f"Loaded model parameters from {self.ckpt_path}")
            else:
                logger.warning(f"No checkpoint found at {self.ckpt_path}. Skip loading.")

        dl = create_embedding_webdataset(
            eval_url,
            batch_size=self.config.batch_size_train,
            num_workers=self.config.num_validation_workers,
            label_processor=self.label_processor,
        )
        preds, labels = inference(self.mlp, dl)

        metric_func = MetricType[metric].__name__
        try:
            evaluator = getattr(ignite.metrics, metric_func)()
        except AttributeError:
            raise ValueError(f"Metric {metric} not found in ignite.metrics")
        evaluator.update(output=(preds, labels))
        result = evaluator.compute()
        logger.info(f"{metric}: {result}")
        return result

    def train_knn(self):
        raise NotImplementedError

    def evaluate_knn(self):
        raise NotImplementedError

    def _make_splits(self):
        splits = self.config.k_fold_splits or [self.config.train_split, self.config.valid_split, self.config.test_split]
        return list(filter(None, splits))
