from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from ignite.contrib.metrics import AveragePrecision
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from loguru import logger
from torch import nn, optim
from tqdm import tqdm


@dataclass
class Trainer:
    model: nn.Module
    accelerator = Accelerator()
    criterion: str = "CrossEntropyLoss"
    optimizer: str = "Adam"
    lr: float = 3e-3
    max_epochs: int = 10
    checkpoint_dir: str = "checkpoints"
    best_ckpt_path: str = None
    ckpt_name: str = "best_model.pt"
    metric: str = "accuracy"
    best_metric: float = 0.0
    save_model: bool = True

    def __post_init__(self):
        try:
            self.criterion = getattr(nn.modules.loss, self.criterion)()
        except AttributeError:
            raise NotImplementedError(f"Loss {self.criterion} not implemented.")

        try:
            self.optimizer = getattr(optim, self.optimizer)(self.model.parameters(), lr=self.lr)
        except AttributeError:
            raise NotImplementedError(f"Optimizer {self.optimizer} not implemented.")

        self.ignite_trainer = Engine(self.train_step)
        self.ignite_evaluator = Engine(self.validation_step)

    @classmethod
    def decode_wds_batch(self, batch: Tuple):
        x, y, _ = batch
        return x.mean(1), y["target"].to(self.accelerator.device)

    def train_step(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = self.decode_wds_batch(batch)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.accelerator.backward(loss)
        self.optimizer.step()
        return loss.item()

    def validation_step(self, engine, batch):
        self.model.eval()
        with torch.inference_mode():
            x, y = self.decode_wds_batch(batch)
            y_pred = self.model(x)
            return y_pred, y

    def run(self, dl_train, dl_dev):
        self.model, self.optimizer, dl_train, dl_dev = self.accelerator.prepare(
            self.model, self.optimizer, dl_train, dl_dev
        )

        metrics = {"mAP": 100 * AveragePrecision(), "loss": Loss(self.criterion), "accuracy": Accuracy()}
        for name, metric in metrics.items():
            metric.attach(self.ignite_evaluator, name)

        @self.ignite_trainer.on(Events.ITERATION_COMPLETED(every=10))
        def log_training_loss(trainer):
            logger.info(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.5f}")

        @self.ignite_trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.ignite_evaluator.run(dl_dev)
            metrics = self.ignite_evaluator.state.metrics
            logger.info(
                f"Epoch: {trainer.state.epoch}  mAP: {metrics['mAP']:.3f} Acc: {metrics['accuracy']:.3f} Avg loss: {metrics['loss']:.5f}"
            )
            if metrics[self.metric]>self.best_metric:
                self.best_metric = metrics[self.metric]
                self.save_model=True
            else:
                self.save_model=False

        from ignite.handlers import ModelCheckpoint

        checkpoint_handler = ModelCheckpoint(
            dirname=self.checkpoint_dir,
            filename_pattern=self.ckpt_name,
            n_saved=1,
            create_dir=True,
            require_empty=False,
        )
        @self.ignite_trainer.on(Events.EPOCH_COMPLETED)
        def save_best_model(trainer):
            if self.save_model:
                logger.info(f"Epoch: {trainer.state.epoch} save checkpoint")
                checkpoint_handler(trainer, {"model": self.model})

        logger.info("Trainer Run.")
        self.ignite_trainer.run(dl_train, self.max_epochs)


def inference(model, dl_eval):
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        model.eval()
        tqdm_dataloader = tqdm(dl_eval, desc="Evaluating")
        for batch in tqdm_dataloader:
            x, y = Trainer.decode_wds_batch(batch)
            y_pred = model(x)
            all_preds.append(y_pred)
            all_targets.append(y)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_preds, all_targets