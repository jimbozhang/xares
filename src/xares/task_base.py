import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import ignite.metrics
import torch
from loguru import logger

from xares.audio_encoder_base import AudioEncoderBase
from xares.audiowebdataset import create_embedding_webdataset
from xares.models import ModelBase
from xares.trainer import MetricType, Trainer, inference


@dataclass
class TaskBase(ABC):
    env_root: Path | str = "/tmp/xares-env"

    force_download: bool = False
    force_generate_audio_tar: bool = False
    force_generate_encoded_tar: bool = False
    force_retrain_mlp: bool = False

    encoder: AudioEncoderBase = None
    wds_audio_paths_dict = {}
    wds_encoded_paths_dict = {}

    batch_size_encode: int = 16
    batch_size_train: int = 32

    num_encoder_workers: int = 4
    num_training_workers: int = 8
    num_validation_workers: int = 0

    model: ModelBase = None

    @property
    def env_dir(self) -> Path:
        return Path(self.env_root) / self.__class__.__name__.replace("Task", "").lower()

    @property
    def audio_tar_ready_file(self):
        return self.env_dir / ".audio_tar_ready"

    @property
    def encoded_tar_ready_file(self):
        return self.env_dir / ".encoded_tar_ready"

    def run_all(self):
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").setLevel(logging.ERROR)

        self.make_audio_tar()
        self.make_encoded_tar()
        self.train_mlp(self.wds_encoded_paths_dict["train"], self.wds_encoded_paths_dict["validation"])
        self.evaluate_mlp(self.wds_encoded_paths_dict["eval"])

    @abstractmethod
    def make_audio_tar(self, force_download=False, force_generate_tar=False) -> None:
        pass

    @abstractmethod
    def make_encoded_tar(self) -> None:
        pass

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
