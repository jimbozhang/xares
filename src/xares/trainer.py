from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from functools import partial
from typing import Any, Dict, Iterable, Literal, Tuple

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import CosineAnnealingScheduler, EpochOutputStore, global_step_from_engine
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Loss, RunningAverage
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
    y = y if isinstance(y, torch.Tensor) else torch.tensor(y)
    return y.float() if y.is_floating_point() else y


def pad_or_trim(tensor, target_size):
    difference = target_size - tensor.shape[-1]
    if difference < 0:
        return tensor[..., :target_size].contiguous()  # Trim excess values
    elif difference > 0:
        return torch.nn.functional.pad(tensor, pad=(0, difference))
    return tensor


def prepare_clip_task_batch(batch: Tuple, device: torch.device | str = "cpu"):
    (x, x_length), y, _ = batch
    x = masked_mean(x, x_length=x_length, dim=-1)
    y = cast_to_tensor(y)
    return x.to(device), y.to(device)


def prepare_contrastive_task_batch(batch: Tuple, device: torch.device | str = "cpu"):
    (x, x_length), y, _ = batch
    # y is a list of strings
    x = masked_mean(x, x_length=x_length, dim=-1)
    # x, y are (B,D,T), but models need B,T,D
    return x.to(device), y


def prepare_frame_task_batch(batch: Tuple, device: torch.device | str = "cpu"):
    (x, x_length), (y, y_length), _ = batch
    y = cast_to_tensor(y)
    # Trim the labels
    y = pad_or_trim(y, x.shape[-1])
    # x, y are (B,D,T), but models need B,T,D
    return x.transpose(-2, -1).contiguous().to(device), y.transpose(-2, -1).contiguous().to(device)


def prepare_asr_task_batch(batch: Tuple, device: torch.device | str = "cpu", tokenizer: Any = None):
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for ASR task.")

    from transformers import DataCollatorForLanguageModeling

    (x, _), labels, _ = batch

    text_with_eos_bos = [f"<|vision_end|>{label['trans']}{tokenizer.eos_token}" for label in labels]
    tokens = [tokenizer(text) for text in text_with_eos_bos]
    data_collator_for_lm = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    y = data_collator_for_lm(tokens)["input_ids"]

    # x are (B,D,T), but models need B,T,D
    return x.transpose(-2, -1).contiguous().to(device), y.contiguous().to(device)


@dataclass
class Trainer:
    model: nn.Module
    device: torch.device | Literal["cpu"] = field(
        default_factory=lambda: torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    optimizer: str = "Adam"
    lr: float = 3e-3
    gradient_accumulation_steps: int = 1
    max_epochs: int = 10
    ckpt_dir: str = "checkpoints"
    best_ckpt_path: str | None = None
    ckpt_name: str = "best_model.pt"
    metric: METRICS_TYPE = "accuracy"
    metric_args: Dict[str, Any] = field(default_factory=lambda: dict())
    label_name: str = "target"
    save_model: bool = True
    decay_fraction: float = 0.1  # Decay learning rate
    task_type: InitVar[Literal["frame", "clip", "contrastive"]] = "clip"

    def __post_init__(self, task_type: Literal["frame", "clip", "contrastive"] = "clip"):
        try:
            self.optimizer = getattr(optim, self.optimizer)(self.model.parameters(), lr=self.lr)
        except AttributeError:
            raise NotImplementedError(f"Optimizer {self.optimizer} not implemented.")

        if task_type == "clip":
            self.prepare_batch_function = prepare_clip_task_batch
        elif task_type == "frame":
            self.prepare_batch_function = prepare_frame_task_batch
        elif task_type == "contrastive":
            self.prepare_batch_function = prepare_contrastive_task_batch
        elif task_type == "asr":
            self.prepare_batch_function = partial(prepare_asr_task_batch, tokenizer=self.model.tokenizer)
        else:
            raise NotImplementedError(f"Trainer.prepare_batch_function for task_type {task_type} not implemented.")

        self.ignite_trainer = Engine(self.train_step)
        self.ignite_evaluator = Engine(self.validation_step)
        # Schedule cosine annealing during training
        if self.max_epochs > 1:
            scheduler = CosineAnnealingScheduler(
                self.optimizer,
                "lr",
                self.optimizer.param_groups[0]["lr"],
                self.lr * self.decay_fraction,
                self.max_epochs,
            )
            self.ignite_trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

        self.model = self.model.to(self.device)
        self._metric_obj = ALL_METRICS[self.metric]
        ProgressBar(
            bar_format=None,
        ).attach(self.ignite_trainer, output_transform=lambda x: x)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(self.ignite_trainer, "loss_avg")
        ProgressBar(bar_format=None).attach(self.ignite_evaluator)

    def train_step(self, engine: Engine, batch: Tuple) -> Dict[str, Tensor]:
        self.model.train()
        with torch.enable_grad():
            self.optimizer.zero_grad()
            loss = self.model(*self.prepare_batch_function(batch, self.device), return_loss=True).loss
            loss.backward()

            if (engine.state.iteration - 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            return {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}

    def validation_step(self, engine: Engine, batch: Tuple) -> Tuple[Tensor, Tensor]:
        self.model.eval()
        with torch.inference_mode():
            y_pred, y = self.model(*self.prepare_batch_function(batch, self.device), return_loss=False)
            return y_pred, y

    def run_inference(self, dl_eval):
        local_evaluator = self.ignite_evaluator
        eval_metric = self._metric_obj.metric(**self.metric_args)
        eval_metric.attach(local_evaluator, self.metric)
        local_evaluator.run(dl_eval)

        dl_eval_size = local_evaluator.state.iteration
        return local_evaluator.state.metrics[self.metric], dl_eval_size

    def run(self, dl_train, dl_dev):
        metrics = {"loss": Loss(self.model.criterion), self.metric: self._metric_obj.metric(**self.metric_args)}
        for name, metric in metrics.items():
            metric.attach(self.ignite_evaluator, name)

        @self.ignite_trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.ignite_evaluator.run(dl_dev)
            metrics = self.ignite_evaluator.state.metrics
            logger.info(
                f"Epoch: {trainer.state.epoch} {self.metric}: {metrics[self.metric]:.3f}  Avg loss: {metrics['loss']:.5f}"
            )

        from ignite.handlers import Checkpoint, ModelCheckpoint

        checkpoint_handler = ModelCheckpoint(
            dirname=self.ckpt_dir,
            filename_pattern=self.ckpt_name,
            n_saved=1,
            create_dir=True,
            require_empty=False,
            score_name=self.metric,
            score_function=Checkpoint.get_default_score_fn(self.metric, self._metric_obj.score),
            global_step_transform=global_step_from_engine(self.ignite_trainer),
        )
        with self.ignite_evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint_handler, dict(model=self.model)
        ):
            logger.info("Trainer Run.")
            self.ignite_trainer.run(dl_train, self.max_epochs)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        return self


class KNN(torch.nn.Module):
    def __init__(
        self,
        train_features: torch.Tensor,
        train_labels: torch.LongTensor,
        nb_knn: int = 10,
        device="cpu",
        T: float = 0.07,
        num_classes=50,
    ):
        super().__init__()
        self.device = device
        self.train_features = normalize(train_features.T.to(self.device), dim=1, p=2)
        self.train_labels = train_labels.view(1, -1).to(self.device)

        self.nb_knn = nb_knn
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.nb_knn, largest=True, sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def compute_neighbors(self, test_data: torch.Tensor) -> tuple[Tensor, Tensor]:

        similarity_rank = test_data @ self.train_features
        candidate_labels = self.train_labels.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def forward(self, test_features: torch.Tensor):
        test_features = normalize(test_features, dim=1, p=2)
        topk_sims, neighbors_labels = self.compute_neighbors(test_features)
        b = test_features.shape[0]
        topk_sims_transform = torch.nn.functional.softmax(topk_sims / self.T, 1)
        scores = torch.nn.functional.one_hot(neighbors_labels, num_classes=self.num_classes) * topk_sims_transform.view(
            b, -1, 1
        )
        return torch.sum(scores[:, : self.nb_knn, :], 1)


@dataclass
class KNNTrainer:
    num_classes: int
    device: torch.device | Literal["cpu"] = field(default_factory=lambda: torch.device("cpu"))
    nb_knn: int = 10
    temperature: float = 0.07
    metric: METRICS_TYPE = "accuracy"

    def __post_init__(self):
        self._metric_obj = ALL_METRICS[self.metric]
        self.trainer = Engine(lambda engine, batch: prepare_clip_task_batch(batch))
        ProgressBar(bar_format=None).attach(self.trainer)
        EpochOutputStore().attach(self.trainer, "output")

    def train_and_eval(self, dl_train, dl_eval) -> Tuple[float, int]:
        logger.info("KNN Feature extraction run.")
        self.trainer.run(dl_train)  # Store all features in memory, should be for most cases okay
        train_data, train_labels = zip(*self.trainer.state.output)
        train_data, train_labels = torch.cat(train_data, 0), torch.cat(train_labels, 0).long()
        knn_model = KNN(train_data, train_labels, nb_knn=self.nb_knn, num_classes=self.num_classes, T=self.temperature)

        def test_step(engine, batch):
            x, y = prepare_clip_task_batch(batch)
            return knn_model(x), y

        eval_engine = Engine(test_step)

        metrics = {self.metric: self._metric_obj.metric()}
        for name, metric in metrics.items():
            metric.attach(eval_engine, name)

        eval_engine.run(dl_eval)
        dl_eval_size = sum(1 for _ in dl_eval)
        metrics = eval_engine.state.metrics
        logger.info(f"KNN {self.metric}: {metrics[self.metric]:.3f}")
        return metrics[self.metric], dl_eval_size


if __name__ == "__main__":
    train_x = torch.randn(1000, 64)
    targets = torch.empty(1000).random_(10).long()
    test_x = torch.randn(100, 64)
    print(pad_or_trim(train_x, 65).shape)

    # model = KNN(train_x, targets, nb_knn=[10, 20, 50], num_classes=10)

    # y = model(test_x)
    # for k, v in y.items():
    # print(k, v.shape)
