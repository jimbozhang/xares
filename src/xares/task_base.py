import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ignite.metrics
import torch
from loguru import logger

from xares.audio_encoder_base import AudioEncoderBase
from xares.audiowebdataset import create_embedding_webdataset, create_rawaudio_webdataset, write_encoded_batches_to_wds
from xares.models import ModelBase
from xares.trainer import MetricType, Trainer, inference


@dataclass
class TaskBase(ABC):
    env_root: Path | str = "/tmp/xares-env"
    splits = ["test", "valid", "train"]

    # Audio tar
    force_download: bool = False
    force_generate_audio_tar: bool = False
    wds_audio_paths_dict = {}

    # Encoded tar
    force_generate_encoded_tar: bool = False
    encoder: AudioEncoderBase = None
    wds_encoded_paths_dict = {}
    trim_length: Optional[int] = None
    save_encoded_per_batches = 4
    batch_size_encode: int = 16
    num_encoder_workers: int = 0

    # MLP
    force_retrain_mlp: bool = False
    ckpt_name = "best.ckpt"
    batch_size_train: int = 32
    num_training_workers: int = 8
    num_validation_workers: int = 4
    model: ModelBase = None
    metric = "accuracy"

    @property
    def env_dir(self) -> Path:
        return Path(self.env_root) / self.__class__.__name__.replace("Task", "").lower()

    @property
    def audio_tar_ready_file(self):
        return self.env_dir / ".audio_tar_ready"

    @property
    def encoded_tar_ready_file(self):
        return self.env_dir / ".encoded_tar_ready"

    def run_all(self) -> float:
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").setLevel(logging.ERROR)

        self.make_audio_tar()
        self.make_encoded_tar()
        self.train_mlp(
            [self.wds_encoded_paths_dict["train"].as_posix()],
            [self.wds_encoded_paths_dict["valid"].as_posix()],
        )
        acc = self.evaluate_mlp([self.wds_encoded_paths_dict["test"].as_posix()], metric=self.metric, load_ckpt=True)
        logger.info(f"The accuracy: {acc}")
        return acc

    @abstractmethod
    def make_audio_tar(self, force_download=False, force_generate_tar=False) -> None:
        pass

    def make_encoded_tar(self):
        if not self.force_generate_encoded_tar and self.encoded_tar_ready_file.exists():
            logger.info(f"Skip making encoded tar. {self.encoded_tar_ready_file} already exists.")
            return

        for split in self.splits:
            logger.info(f"Encoding audio for split {split} ...")

            dl = create_rawaudio_webdataset(
                [self.wds_audio_paths_dict[split].as_posix()],
                batch_size=self.batch_size_encode,
                num_workers=self.num_encoder_workers,
                crop_size=self.trim_length,
                with_json=True,
            )

            batch_buf = []
            shard = 0
            for batch, _, label, keys in dl:
                batch = batch.to(self.encoder.device)
                encoded_batch = self.encoder(batch, 44_100).to("cpu").detach()
                batch_buf.append([encoded_batch, label, keys])

                if len(batch_buf) >= self.save_encoded_per_batches:
                    write_encoded_batches_to_wds(
                        batch_buf,
                        self.wds_encoded_paths_dict[split].as_posix().replace("*", f"0{shard:05d}"),
                        num_workers=self.num_encoder_workers,
                    )
                    batch_buf.clear()
                    shard += 1

            if len(batch_buf) > 0:
                write_encoded_batches_to_wds(
                    batch_buf,
                    self.wds_encoded_paths_dict[split].as_posix().replace("*", f"0{shard:05d}"),
                    num_workers=self.num_encoder_workers,
                )

        self.encoded_tar_ready_file.touch()

    def train_mlp(self, train_url: list, validation_url: list) -> None:
        if not self.force_retrain_mlp and self.ckpt_path.exists():
            logger.info(f"Checkpoint {self.ckpt_path} already exists. Skip training.")
            return

        assert self.model is not None
        trainer = Trainer(self.model, checkpoint_dir=self.checkpoint_dir, ckpt_name=self.ckpt_name, metric=self.metric)

        dl_train = create_embedding_webdataset(
            train_url, tar_shuffle=2000, batch_size=self.batch_size_train, num_workers=self.num_training_workers
        )
        dl_val = create_embedding_webdataset(
            validation_url, tar_shuffle=2000, batch_size=self.batch_size_train, num_workers=self.num_validation_workers
        )

        trainer.run(dl_train, dl_val)

    def evaluate_mlp(self, eval_url: list, metric: str = "Accuracy", load_ckpt: bool = False) -> float:
        if load_ckpt:
            if self.ckpt_path.exists():
                self.model.load_state_dict(torch.load(self.ckpt_path))
                logger.info(f"Loaded model parameters from {self.ckpt_path}")
            else:
                logger.warning(f"No checkpoint found at {self.ckpt_path}. Skip loading.")

        dl = create_embedding_webdataset(
            eval_url, tar_shuffle=2000, batch_size=self.batch_size_train, num_workers=self.num_validation_workers
        )
        preds, labels = inference(self.model, dl)

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
        pass

    def evaluate_knn(self):
        pass
