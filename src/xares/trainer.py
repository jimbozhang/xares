from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple
from ignite.handlers import CosineAnnealingScheduler, EpochOutputStore, global_step_from_engine
from ignite.metrics import Loss, RunningAverage

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers.tqdm_logger import ProgressBar
from loguru import logger
from torch import Tensor, nn, optim
from torch.nn.functional import normalize
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

        self.model.to(self.accelerator.device)
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
        return local_evaluator.state.metrics[self.metric]

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


class KNN(torch.nn.Module):
    def __init__(self,
                 train_features:torch.Tensor,
                 train_labels:torch.LongTensor,
                 nb_knn: int = 10,
                 device = 'cpu',
                 T:float = 0.07,
                 num_classes=50):
        super().__init__()
        self.device = device
        self.train_features = normalize(train_features.T.to(self.device), dim=1, p=2)
        self.train_labels = train_labels.view(1, -1).to(self.device)

        self.nb_knn = nb_knn
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.nb_knn,
                                             largest=True,
                                             sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels


    def compute_neighbors(self, test_data:torch.Tensor) -> tuple[Tensor, Tensor]:

        similarity_rank = test_data @ self.train_features
        candidate_labels = self.train_labels.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def forward(self, test_features: torch.Tensor):
        test_features = normalize(test_features, dim=1, p=2)
        topk_sims, neighbors_labels = self.compute_neighbors(test_features)
        b = test_features.shape[0]
        topk_sims_transform = torch.nn.functional.softmax(
            topk_sims / self.T, 1)
        scores = torch.nn.functional.one_hot(
            neighbors_labels,
            num_classes=self.num_classes) * topk_sims_transform.view(b, -1, 1)
        return torch.sum(scores[:, :self.nb_knn, :], 1)

@dataclass
class KNNTrainer:
    num_classes: int
    device: torch.device | Literal['cpu'] = field(
        default_factory=lambda: torch.device('cpu'))
    nb_knn: int = 10
    temperature: float = 0.07
    metric: METRICS_TYPE = "accuracy"

    def __post_init__(self):
        self._metric_obj = ALL_METRICS[self.metric]
        self.trainer = Engine(lambda engine, batch: prepare_wds_batch_default(batch))
        ProgressBar(bar_format=None).attach(self.trainer)
        # self.evaluate_engine = Engine(lambda batch:)
        EpochOutputStore().attach(self.trainer, "output")

    def train(self, dl_train, dl_eval):
        logger.info("KNN Feature extraction run.")
        self.trainer.run(dl_train) # Store all features in memory, should be for most cases okay
        train_data, train_labels = zip(*self.trainer.state.output)
        train_data, train_labels = torch.cat(train_data,
                                             0), torch.cat(train_labels,
                                                           0).long()
        knn_model = KNN(train_data, train_labels, nb_knn=self.nb_knn, num_classes=self.num_classes, T= self.temperature)
        def test_step(engine, batch):
            x, y =  prepare_wds_batch_default(batch)
            return knn_model(x), y
        eval_engine = Engine(test_step)

        metrics = {self.metric : self._metric_obj.metric()}
        for name, metric in metrics.items():
            metric.attach(eval_engine, name)

        eval_engine.run(dl_eval)
        metrics =eval_engine.state.metrics
        logger.info(
            f"KNN {self.metric}: {metrics[self.metric]:.3f}"
        )
        return metrics[self.metric]






if __name__ == "__main__":
    train_x = torch.randn(1000, 64)
    targets = torch.empty(1000).random_(10).long()
    test_x = torch.randn(100, 64)


    model = KNN(train_x, targets, nb_knn=[10,20,50], num_classes=10)

    y = model(test_x)
    for k,v in y.items():
        print(k, v.shape)
