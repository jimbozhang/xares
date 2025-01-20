from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Literal, Tuple
from ignite.handlers import EpochOutputStore, global_step_from_engine

import torch
import torch.nn as nn
from accelerate import Accelerator
from ignite.contrib.metrics import AveragePrecision
from ignite.engine import Engine, Events
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Accuracy, Loss, RunningAverage
from loguru import logger
from torch import Tensor, nn, optim



MetricType = {
    "accuracy": Accuracy,
    "mAP": AveragePrecision,
}


def length_to_mask(length):
    max_length = length.amax().item()
    idx = torch.arange(max_length, device=length.device)
    return idx.unsqueeze(0) < length.unsqueeze(-1)


def masked_mean(x, x_length, dim: int = -1):
    mask = length_to_mask(x_length)
    mask_target_shape = (len(mask),) + (1,) * (x.ndim - 2) + (mask.shape[-1],)
    mask = mask.view(mask_target_shape)
    return (x * mask).sum(dim) / mask.sum(dim)


def cast_to_tensor(y: Iterable):
    return torch.tensor(y) if not isinstance(y, torch.Tensor) else y


def prepare_wds_batch_default(batch: Tuple,
                              device: torch.device | str = 'cpu'):
    (x, x_length), y, _ = batch
    x = masked_mean(x, x_length=x_length, dim=-1)
    y = cast_to_tensor(y)
    return x.to(device), y.to(device)


def default_validation_func(model:torch.nn.Module, device: torch.device | str = 'cpu') -> Callable:
    def validation_step(engine: Engine, batch: Tuple) -> Tuple[Tensor, Tensor]:
        model.eval()
        with torch.inference_mode():
            y_pred, y = model(*prepare_wds_batch_default(batch,device), return_loss=False)
            return y_pred, y
    return validation_step


@dataclass
class EvalMetric:
    metric : Callable
    score: float = 1. # can also be -1


@dataclass
class Trainer:
    model: nn.Module
    accelerator: Accelerator = field(default_factory=Accelerator())
    validation_func: Callable | None = None
    optimizer: str = "Adam"
    lr: float = 3e-3
    max_epochs: int = 10
    ckpt_dir: str = "checkpoints"
    best_ckpt_path: str | None = None
    ckpt_name: str = "best_model.pt"
    metric: str = "accuracy"
    label_name: str = "target"
    best_metric: float = 0.0
    save_model: bool = True

    def __post_init__(self):
        try:
            self.optimizer = getattr(optim, self.optimizer)(self.model.parameters(), lr=self.lr)
        except AttributeError:
            raise NotImplementedError(f"Optimizer {self.optimizer} not implemented.")

        self.ignite_trainer = Engine(self.train_step)
        self.ignite_evaluator = Engine(
            default_validation_func(self.model, device=self.accelerator.device)
            if self.validation_func is None else self.
            validation_func(self.model, device=self.accelerator.device))
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        ProgressBar(bar_format=None,
                    disable=not self.accelerator.is_main_process).attach(
                        self.ignite_trainer, output_transform=lambda x: x)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(self.ignite_trainer, "loss_avg")
        ProgressBar(bar_format=None, disable=not self.accelerator.is_main_process).attach(self.ignite_evaluator)


    def train_step(self, engine: Engine, batch: Tuple) -> Dict[str, Tensor]:
        self.model.train()
        with torch.enable_grad(), self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            loss = self.model(*prepare_wds_batch_default(batch,self.accelerator.device), return_loss=True)
            self.accelerator.backward(loss)
            self.optimizer.step()
            return {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}

    def run_inference(self, dl_eval):
        local_evaluator = self.ignite_evaluator
        EpochOutputStore().attach(local_evaluator, 'output')
        self.ignite_evaluator.run(dl_eval)
        pred, tar = list(zip(*local_evaluator.state.output))
        return torch.cat(pred, dim=0), torch.cat(tar, dim=0)

    def run(self, dl_train, dl_dev):
        self.model, self.optimizer, dl_train, dl_dev = self.accelerator.prepare(
            self.model, self.optimizer, dl_train, dl_dev
        )

        metrics = {"loss": Loss(self.accelerator.unwrap_model(self.model).criterion), self.metric: MetricType[self.metric]()}
        for name, metric in metrics.items():
            metric.attach(self.ignite_evaluator, name)

        @self.ignite_trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.ignite_evaluator.run(dl_dev)
            metrics = self.ignite_evaluator.state.metrics
            logger.info(
                f"Epoch: {trainer.state.epoch} {self.metric}: {metrics[self.metric]:.3f}  Avg loss: {metrics['loss']:.5f}"
            )

        from ignite.handlers import ModelCheckpoint


        checkpoint_handler = ModelCheckpoint(
            dirname=self.ckpt_dir,
            filename_pattern=self.ckpt_name,
            n_saved=1,
            create_dir=True,
            require_empty=False,
            score_name=self.metric,
            global_step_transform=global_step_from_engine(self.ignite_trainer)
        )
        with self.ignite_evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                              checkpoint_handler,
                                                     dict(model=self.model)):
            logger.info("Trainer Run.")
            self.ignite_trainer.run(dl_train, self.max_epochs)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        return self
