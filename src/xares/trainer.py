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
    lr: float = 1e-3
    max_epochs: int = 15
    checkpoint_dir: str = "checkpoints"
    best_ckpt_path: str = None

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

        torch.multiprocessing.set_start_method("spawn", force=True)

    @classmethod
    def decode_wds_batch(self, batch: Tuple):
        x, y, z = batch
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
        trainer = Engine(self.train_step)
        evaluator = Engine(self.validation_step)

        self.model, self.optimizer, dl_train, dl_dev = self.accelerator.prepare(
            self.model, self.optimizer, dl_train, dl_dev
        )

        metrics = {"mAP": 100 * AveragePrecision(), "loss": Loss(self.criterion), "accuracy": Accuracy()}
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

        @trainer.on(Events.ITERATION_COMPLETED(every=10))
        def log_training_loss(trainer):
            logger.info(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.5f}")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(dl_dev)
            metrics = evaluator.state.metrics
            logger.info(
                f"Epoch: {trainer.state.epoch}  mAP: {metrics['mAP']:.3f} Acc: {metrics['accuracy']:.3f} Avg loss: {metrics['loss']:.5f}"
            )

        from ignite.handlers import ModelCheckpoint

        checkpoint_handler = ModelCheckpoint(
            dirname=self.checkpoint_dir,
            filename_pattern="best_model.pt",
            n_saved=1,
            create_dir=True,
            require_empty=False,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": self.model})

        logger.info("Trainer Run.")
        trainer.run(dl_train, self.max_epochs)


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


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 out_features=None,):
        super().__init__()
        out_features = out_features or in_features
        self.ln = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        return x