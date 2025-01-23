from __future__ import annotations

import io
import logging
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

import webdataset as wds
from xares.audiowebdataset import create_embedding_webdataset, create_rawaudio_webdataset
from xares.common import XaresSettings
from xares.metrics import METRICS_TYPE
from xares.models import Mlp
from xares.trainer import KNNTrainer, Trainer
from xares.utils import download_zenodo_record, mkdir_if_not_exists




@dataclass
class TaskConfig:
    xares_settings: XaresSettings = field(default_factory=XaresSettings)
    env_root: Path | str | None = None
    name: str = "default_task"

    # General
    private: bool = False
    torch_num_threads: int = 2  # Do not use too many otherwise slows down
    seed: int = 42  # manual seed for all experiments
    label_processor: Callable = field(default=lambda x: x)

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
    force_encode: bool = False
    encoder: Any = None
    encoded_tar_name_of_split: Dict[Any, Any] = field(default_factory=lambda: dict())
    trim_length = None
    save_encoded_per_batches: int = 2000
    batch_size_encode: int = 64
    num_encoder_workers: int = 4
    crop_length: None | float = None

    # MLP
    force_retrain_mlp: bool = False
    ckpt_dir_name = "checkpoints"
    embedding_dir_name = "embeddings"
    ckpt_name = "best.ckpt"
    criterion: Literal["CrossEntropyLoss", "BCEWithLogitsLoss", "L1Loss", "MSELoss"] = "CrossEntropyLoss"
    batch_size_train: int = 32
    learning_rate: float = 1e-3
    epochs: int = 10
    num_training_workers: int = 0
    num_validation_workers: int = 0
    model: nn.Module | None = None
    output_dim: int | None = None
    metric: Literal["accuracy","EER","mAP","recall@k","MAE", "MSE"] = "accuracy"

    def __post_init__(self, **kwargs):
        self.update_tar_name_of_split()
        if self.env_root is None:
            self.env_root = self.xares_settings.env_root
        torch.set_num_threads(self.torch_num_threads)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_tar_name_of_split(self):
        if self.k_fold_splits is not None:
            self.audio_tar_name_of_split = {fold: f"wds-audio-fold-{fold}-*.tar" for fold in self.k_fold_splits}
            self.encoded_tar_name_of_split = {fold: f"wds-encoded-fold-{fold}-*.tar" for fold in self.k_fold_splits}
        else:
            self.audio_tar_name_of_split = {
                self.train_split: f"{self.train_split}*.tar",
                self.valid_split: f"{self.valid_split}*.tar",
                self.test_split: f"{self.test_split}*.tar",
            }
            self.encoded_tar_name_of_split = {
                self.train_split: f"wds-encoded-{self.train_split}*.tar",
                self.valid_split: f"wds-encoded-{self.valid_split}*.tar",
                self.test_split: f"wds-encoded-{self.test_split}*.tar",
            }


class XaresTask:
    def __init__(self, config: TaskConfig):
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").setLevel(logging.ERROR)
        # Make the logger with this format the default for all loggers in this package
        logger.configure(
            handlers=[
                {
                    "sink": sys.stdout,
                    "format": "<fg #FF6900>(X-ARES)</fg #FF6900> [<yellow>{time:YYYY-MM-DD HH:mm:ss}</yellow>] {message}",
                    "level": "DEBUG",
                }
            ]
        )

        self.config = config
        self.encoder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.config.encoder.to(self.encoder_device)
        self.mlp = None

        torch.set_num_threads(self.config.torch_num_threads)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.env_dir = Path(self.config.env_root) / self.config.name
        self.encoder_name = self.encoder.__class__.__name__
        self.ckpt_dir = self.env_dir / self.config.ckpt_dir_name / self.encoder_name
        self.encoded_tar_dir = self.env_dir / self.config.embedding_dir_name / self.encoder_name
        self.ckpt_path = self.ckpt_dir / self.config.ckpt_name
        mkdir_if_not_exists(self.encoded_tar_dir)

        if self.config.k_fold_splits:
            self.config.train_split = self.config.valid_split = self.config.test_split = None

        self.label_processor = self.config.label_processor

    def make_encoded_tar(self):
        self.encoded_tar_path_of_split = {
            split: (self.encoded_tar_dir / self.config.encoded_tar_name_of_split[split])
            for split in self.config.encoded_tar_name_of_split
        }

        encoded_ready_path = self.encoded_tar_dir / self.config.xares_settings.encoded_ready_filename
        if not self.config.force_encode and encoded_ready_path.exists():
            logger.info(f"Skip making encoded tar.")
            return

        audio_ready_path = self.env_dir / self.config.xares_settings.audio_ready_filename
        if not audio_ready_path.exists():
            if self.config.private:
                raise ValueError("For private dataset, audio tar must be provided at local path.")
            download_zenodo_record(self.config.zenodo_id, self.env_dir, force_download=self.config.force_download)
            audio_ready_path.touch()

        audio_tar_path_of_split = {
            split: (self.env_dir / self.config.audio_tar_name_of_split[split]).as_posix()
            for split in self.config.audio_tar_name_of_split
        }


        for split in audio_tar_path_of_split:
            logger.info(f"Encoding audio for split {split} ...")
            logger.debug(f"Using data from {audio_tar_path_of_split[split]} ... ")
            dl = create_rawaudio_webdataset(
                [audio_tar_path_of_split[split]],
                target_sample_rate=self.encoder.sampling_rate,
                audio_key_name="audio",
                num_workers=self.config.num_encoder_workers,
                batch_size=self.config.batch_size_encode,
                crop_length=self.config.crop_length,
                pad_last=True, # Add crop
            )
            sink = wds.ShardWriter(
                self.encoded_tar_path_of_split[split].as_posix().replace("*", f"0%05d"),
                encoder=False,
                compress=False,
                maxcount=self.config.save_encoded_per_batches,
                verbose=False,
            )

            with torch.inference_mode():
                for enum_item,((audio, audio_length), json_data, filenames) in tqdm(enumerate(dl), desc=f"Encoding {split}", leave=True):
                    audio = audio.to(self.encoder_device)
                    embedding = self.encoder(audio).to("cpu").detach()
                    for embed, json_data_sample, filename in zip(embedding, json_data, filenames):
                        buf = io.BytesIO()
                        np.save(buf,embed.numpy())
                        sink.write({"npy": buf.getvalue(), 'json': json.dumps(json_data_sample).encode('utf-8'), '__key__': f"{filename}{enum_item}"})

        encoded_ready_path.touch()

    def run_mlp(self) -> float:
        if self.config.k_fold_splits:
            # K-fold cross validation
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
                        [self.encoded_tar_path_of_split[k].as_posix()], load_ckpt=True
                    )
                )

            for k in range(len(splits)):
                logger.info(f"Fold {k+1} {self.config.metric}: {acc[k]}")

            avg_score = np.mean(acc)
            logger.info(f"The averaged {self.config.metric} of 5 folds is: {avg_score}")

            return avg_score

        else:
            # Single split
            self.train_mlp(
                [self.encoded_tar_path_of_split[self.config.train_split].as_posix()],
                [self.encoded_tar_path_of_split[self.config.valid_split].as_posix()],
            )
            score = self.evaluate_mlp(
                [self.encoded_tar_path_of_split[self.config.test_split].as_posix()],
                load_ckpt=True,
            )
            logger.info(f"The {self.config.metric}: {score}")
            return score

    def train_mlp(self, train_url: list, validation_url: list) -> None:
        mlp = Mlp(
            in_features=self.encoder.output_dim, out_features=self.config.output_dim, criterion=self.config.criterion
        )

        self.trainer = Trainer(
            mlp,
            ckpt_dir=self.ckpt_dir,
            ckpt_name=self.config.ckpt_name,
            metric=self.config.metric,
            lr=self.config.learning_rate,
            max_epochs=self.config.epochs,

        )

        if not self.config.force_retrain_mlp and self.ckpt_path.exists():
            logger.info(f"Checkpoint {self.ckpt_path} already exists. Skip training.")
            self.trainer.load_state_dict(torch.load(self.ckpt_path))
            return


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

        self.trainer.run(dl_train, dl_val)

    def evaluate_mlp(self, eval_url: list, load_ckpt: bool = False) -> Dict[METRICS_TYPE,Any]:
        if self.trainer is None:
            raise ValueError("Train the model first before evaluation.")

        if load_ckpt:
            if self.ckpt_path.exists():
                self.trainer.load_state_dict(torch.load(self.ckpt_path))
                logger.info(f"Loaded model parameters from {self.ckpt_path}")
            else:
                logger.warning(f"No checkpoint found at {self.ckpt_path}. Skip loading.")

        dl = create_embedding_webdataset(
            eval_url,
            batch_size=self.config.batch_size_train,
            num_workers=self.config.num_validation_workers,
            label_processor=self.label_processor,
        )
        result = self.trainer.run_inference(dl)
        # for k,v in result.items():
            # logger.info(f"{k}: {v}")
        return result


    def run_knn(self):
        if self.config.k_fold_splits:
            # K-fold cross validation
            score = []
            splits = self._make_splits()
            wds_encoded_training_fold_k = {
                k: [self.encoded_tar_path_of_split[j].as_posix() for j in splits if j != k] for k in splits
            }

            for k in splits:
                score.append(
                self.train_knn(
                    wds_encoded_training_fold_k[k],
                    [self.encoded_tar_path_of_split[k].as_posix()],
                ))

            for k in range(len(splits)):
                logger.info(f"Fold {k+1} {self.config.metric}: {score[k]}")

            avg_score = np.mean(score)
            logger.info(f"The averaged KNN {self.config.metric} of 5 folds is: {avg_score}")

            return avg_score

        else:
            # Single split
            score = self.train_knn(
                [self.encoded_tar_path_of_split[self.config.train_split].as_posix()],
                [self.encoded_tar_path_of_split[self.config.test_split].as_posix()],
            )
            logger.info(f"The KNN score: {score}")
            score = score
            return score

    def train_knn(self, train_url, eval_url):
        dl_train = create_embedding_webdataset(
            train_url,
            tar_shuffle=2000,
            batch_size=self.config.batch_size_train,
            num_workers=self.config.num_training_workers,
            training=True,
            label_processor=self.label_processor,
        )
        dl_eval = create_embedding_webdataset(
            eval_url,
            batch_size=self.config.batch_size_train,
            num_workers=self.config.num_validation_workers,
            training=False,
            label_processor=self.label_processor,
        )
        knn_trainer = KNNTrainer(num_classes=self.config.output_dim)
        scores = knn_trainer.train(dl_train, dl_eval)
        return scores

    def run(self):
        self.make_encoded_tar()
        mlp_score = self.run_mlp()
        knn_score = self.run_knn()
        return mlp_score, knn_score

        

    def _make_splits(self):
        splits = self.config.k_fold_splits or [self.config.train_split, self.config.valid_split, self.config.test_split]
        return list(filter(None, splits))
