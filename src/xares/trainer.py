from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Callable, Dict, Iterable, Literal, Tuple
from ignite.handlers import CosineAnnealingScheduler, EpochOutputStore, global_step_from_engine
from ignite.metrics import Loss, RunningAverage

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers.tqdm_logger import ProgressBar
from loguru import logger
from torch import Tensor, nn, optim
from xares.metrics import ALL_METRICS, METRICS_TYPE



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
    y = torch.tensor(y) if not isinstance(y, torch.Tensor) else y
    return y.float() if y.is_floating_point() else y


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
class Trainer:
    model: nn.Module
    validation_func: Callable | None = None
    device: torch.device | Literal['cpu'] = field(default_factory=lambda : torch.device('cpu'))
    optimizer: str = "Adam"
    lr: float = 3e-3
    max_epochs: int = 10
    ckpt_dir: str = "checkpoints"
    best_ckpt_path: str | None = None
    ckpt_name: str = "best_model.pt"
    metric: METRICS_TYPE = "accuracy"
    label_name: str = "target"
    save_model: bool = True
    decay_fraction: float = 0.1 # Decay learning rate

    def __post_init__(self):
        try:
            self.optimizer = getattr(optim, self.optimizer)(self.model.parameters(), lr=self.lr)
        except AttributeError:
            raise NotImplementedError(f"Optimizer {self.optimizer} not implemented.")

        self.ignite_trainer = Engine(self.train_step)
        self.ignite_evaluator = Engine(
            default_validation_func(self.model, device=self.device)
            if self.validation_func is None else self.
            validation_func(self.model, device=self.device))
        # Schedule cosine annealing during training
        scheduler = CosineAnnealingScheduler(
                self.optimizer, 'lr', self.optimizer.param_groups[0]['lr'],
                self.lr * self.decay_fraction, self.max_epochs)
        self.ignite_trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

        self.model = self.model.to(self.device)
        self._metric_obj = ALL_METRICS[self.metric]
        ProgressBar(bar_format=None,
                    ).attach(
                        self.ignite_trainer, output_transform=lambda x: x)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(self.ignite_trainer, "loss_avg")
        ProgressBar(bar_format=None).attach(self.ignite_evaluator)


    def train_step(self, engine: Engine, batch: Tuple) -> Dict[str, Tensor]:
        self.model.train()
        with torch.enable_grad():
            self.optimizer.zero_grad()
            loss = self.model(*prepare_wds_batch_default(batch,self.device), return_loss=True)
            loss.backward()
            self.optimizer.step()
            return {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}

    def run_inference(self, dl_eval: Iterable):
        local_evaluator = self.ignite_evaluator
        eval_metric = self._metric_obj.metric()
        eval_metric.attach(local_evaluator, self.metric)
        local_evaluator.run(dl_eval)
        return local_evaluator.state.metrics

    def run(self, dl_train, dl_dev):
        metrics = {'loss':Loss(self.model.criterion), self.metric : self._metric_obj.metric()}
        for name, metric in metrics.items():
            metric.attach(self.ignite_evaluator, name)

        @self.ignite_trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.ignite_evaluator.run(dl_dev)
            metrics = self.ignite_evaluator.state.metrics
            logger.info(
                f"Epoch: {trainer.state.epoch} {self.metric}: {metrics[self.metric]:.3f}  Avg loss: {metrics['loss']:.5f}"
            )

        from ignite.handlers import ModelCheckpoint, Checkpoint
        checkpoint_handler = ModelCheckpoint(
            dirname=self.ckpt_dir,
            filename_pattern=self.ckpt_name,
            n_saved=1,
            create_dir=True,
            require_empty=False,
            score_name=self.metric,
            score_function=Checkpoint.get_default_score_fn(self.metric, self._metric_obj.score),
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
