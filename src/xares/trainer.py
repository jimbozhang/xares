from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from accelerate import Accelerator
from ignite.contrib.metrics import AveragePrecision
from ignite.engine import Engine, Events
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Accuracy, Loss, RunningAverage
from loguru import logger
from torch import nn, optim
from tqdm import tqdm
import sys
# Make the logger with this format the default for all loggers in this package
logger.configure(handlers=[{
    "sink": sys.stderr,
    "format": "<fg #FF6900>(X-ARES)</fg #FF6900> [<yellow>{time:YYYY-MM-DD HH:mm:ss}</yellow>] {message}",
    'level': 'DEBUG',
}])


MetricType = {
    "accuracy": Accuracy,
    "mAP": AveragePrecision,
}


@dataclass
class Trainer:
    model: nn.Module
    accelerator :Accelerator = Accelerator()
    criterion: str = "CrossEntropyLoss"
    optimizer: str = "Adam"
    lr: float = 3e-3
    max_epochs: int = 10
    checkpoint_dir: str = "checkpoints"
    best_ckpt_path: str | None = None
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
        ProgressBar(bar_format=None, disable=not self.accelerator.is_main_process).attach(self.ignite_trainer, output_transform=lambda x:x)
        RunningAverage(output_transform=lambda x: x['loss']).attach(self.ignite_trainer, 'loss_avg')
        ProgressBar(bar_format=None, disable=not self.accelerator.is_main_process).attach(self.ignite_evaluator)

    @classmethod
    def decode_wds_batch(cls, batch: Iterable):
        (x,x_length), y, _ = batch
        # x= (B, D ,T ), x_length = (B,)
        x = x.mean(-1) # Pool across time
        y = torch.tensor([y_i["target"] for y_i in y])
        return x.to(cls.accelerator.device), y.to(cls.accelerator.device)

    def train_step(self, engine:Engine, batch:Tensor) -> Dict[str,Tensor]:
        self.model.train()
        with torch.enable_grad():
            self.optimizer.zero_grad()
            x, y = self.decode_wds_batch(batch)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            self.accelerator.backward(loss)
            self.optimizer.step()
            return {'loss':loss.item(), 'lr':self.optimizer.param_groups[0]['lr']}

    def validation_step(self, engine:Engine, batch:Iterable[Tensor]) -> Tuple[Tensor,Tensor]:
        self.model.eval()
        with torch.inference_mode():
            x, y = self.decode_wds_batch(batch)
            y_pred = self.model(x)
            return y_pred, y

    def run(self, dl_train, dl_dev):
        self.model, self.optimizer, dl_train, dl_dev = self.accelerator.prepare(
            self.model, self.optimizer, dl_train, dl_dev
        )

        metrics = {"loss": Loss(self.criterion), self.metric: MetricType[self.metric]()}
        for name, metric in metrics.items():
            metric.attach(self.ignite_evaluator, name)

        @self.ignite_trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.ignite_evaluator.run(dl_dev)
            metrics = self.ignite_evaluator.state.metrics
            logger.info(
                f"Epoch: {trainer.state.epoch}  {self.metric}: {metrics[self.metric]:.3f}  Avg loss: {metrics['loss']:.5f}"
            )
            if metrics[self.metric] > self.best_metric:
                self.best_metric = metrics[self.metric]
                self.save_model = True
            else:
                self.save_model = False

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
