from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import ignite.metrics
import torch
from loguru import logger
from webdataset import WebLoader

from xares.audio_encoder_base import AudioEncoderBase
from xares.dataset import EmbeddingWebdataset
from xares.trainer import Trainer, inference


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
    num_training_workers: int = 0
    num_validation_workers: int = 0

    @property
    def env_dir(self):
        return Path(self.env_root) / self.__class__.__name__.replace("Task", "").lower()

    def run_all(self):
        self.make_audio_tar()
        self.make_encoded_tar()
        self.train_mlp(self.wds_encoded_paths_dict["train"], self.wds_encoded_paths_dict["validation"])
        self.evaluate_mlp(self.wds_encoded_paths_dict["eval"])

    @abstractmethod
    def make_audio_tar(self, force_download=False, force_generate_tar=False):
        pass

    @abstractmethod
    def make_encoded_tar(self):
        pass

    def train_mlp(self, train_url: list, validation_url: str):
        if not self.force_retrain_mlp and self.ckpt_path.exists():
            logger.info(f"Checkpoint {self.ckpt_path} already exists. Skip training.")
            return

        trainer = Trainer(self.model, checkpoint_dir=self.checkpoint_dir)

        ds_train = EmbeddingWebdataset(train_url, shuffle=2000)
        dl_train = WebLoader(ds_train, batch_size=self.batch_size, num_workers=self.num_training_workers)

        ds_val = EmbeddingWebdataset(validation_url)
        dl_val = WebLoader(ds_val, batch_size=self.batch_size, num_workers=self.num_validation_workers)

        trainer.run(dl_train, dl_val)

    def evaluate_mlp(self, eval_url: str, metric: str = "Accuracy", load_ckpt: bool = False):
        if load_ckpt:
            ckpt_path = self.checkpoint_dir / "best_model.pt"

            if ckpt_path.exists():
                self.model.load_state_dict(torch.load(ckpt_path))
                logger.info(f"Loaded model parameters from {ckpt_path}")
            else:
                logger.warning(f"No checkpoint found at {ckpt_path}. Skip loading.")

        ds = EmbeddingWebdataset(eval_url)
        dl = WebLoader(ds, batch_size=self.batch_size, num_workers=0)
        preds, labels = inference(self.model, dl)

        try:
            evaluator = getattr(ignite.metrics, metric)()
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
