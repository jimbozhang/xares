from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
import torch
from ignite.metrics import Accuracy, AveragePrecision, EpochMetric, MeanAbsoluteError, MeanSquaredError, Metric

METRICS_TYPE = Literal["accuracy", "EER", "mAP", "recallatk_r1", "MAE", "MSE", "segmentf1", "WER"]


@dataclass
class EvalMetric:
    metric: Callable
    score: float = 1.0  # can also be -1


class ClapScore(Metric):
    def __init__(
        self,
        num_caps: int = 5,  # Clotho uses 5 caps
        select: Literal["r1", "r5", "r10", "mAP10"] | None = None,
        average: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._reset()
        self.average = average
        self.num_caps = num_caps
        self.select = select

    def _reset(self):
        self.embds_x, self.embds_y = [], []

    def update(self, output):
        """update.

        :param self: self object
        """
        embeds_x, embeds_y = output
        self.embds_x.append(embeds_x)
        self.embds_y.append(embeds_y)

    def reset(self):
        self._reset()

    def compute(self) -> Dict[str, float] | float:
        return _clapscore(
            torch.cat(self.embds_x, dim=0),
            torch.cat(self.embds_y, dim=0),
            num_caps=self.num_caps,
            average=self.average,
            select=self.select,
        )


class SegmentF1Metric(Metric):
    def __init__(self, *args, output_transform=lambda x: x, hop_size_in_ms: float, segment_length_in_s: float = 1.0):
        super().__init__(*args, output_transform=output_transform)
        self._reset()
        self.max_pooling_size = math.ceil(segment_length_in_s / (hop_size_in_ms / 1000.0))

    def _reset(self):
        self.pred, self.targets = [], []

    def update(self, output):
        pred, tar = output
        assert pred.ndim == tar.ndim, "Dims need to be identical"
        pred = (
            torch.nn.functional.max_pool1d(pred.transpose(-2, -1), kernel_size=self.max_pooling_size)
            .transpose(-2, -1)
            .flatten(0, 1)
        )
        tar = (
            torch.nn.functional.max_pool1d(tar.transpose(-2, -1), kernel_size=self.max_pooling_size)
            .transpose(-2, -1)
            .flatten(0, 1)
        )
        self.pred.append(pred)
        self.targets.append(tar)

    def reset(self):
        self._reset()

    def compute(self) -> Dict[str, float] | float:
        from sklearn.metrics import f1_score

        # Binary classification, preds need to be {0,1} and targets
        pred = torch.cat(self.pred).long().cpu().numpy()
        tar = torch.cat(self.targets).long().cpu().numpy()
        return f1_score(tar, pred, average="macro")


def compute_eer(pred, target, positive_label: int = 1):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(target, pred, positive_label)
    fnr = 1 - tpr
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer_1 + eer_2) / 2
    return eer


EERMetric = lambda: EpochMetric(compute_fn=compute_eer)


class WerScore(Metric):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._reset()

    def _reset(self):
        self.pred, self.target = [], []

    def update(self, output):
        pred, target = output
        self.pred.append(pred)
        self.target.append(target)

    def reset(self):
        self._reset()

    def compute(self) -> Dict[str, float] | float:
        from jiwer import wer

        wer_score = max(0, 1 - wer(self.target, self.pred))
        return wer_score


ALL_METRICS: Dict[METRICS_TYPE, EvalMetric] = dict(
    accuracy=EvalMetric(Accuracy, score=1.0),
    frame_mAP=EvalMetric(
        partial(AveragePrecision, output_transform=lambda x: (torch.flatten(x[0], 0, 1), torch.flatten(x[1], 0, 1))),
        score=1.0,
    ),
    segmentf1=EvalMetric(
        partial(SegmentF1Metric, output_transform=lambda x: (x[0].sigmoid().round(), x[1])), score=1.0
    ),
    mAP=EvalMetric(AveragePrecision, score=1.0),
    MAE=EvalMetric(MeanAbsoluteError, score=-1.0),
    MSE=EvalMetric(MeanSquaredError, score=-1.0),
    recallatk_r1=EvalMetric(partial(ClapScore, select="r1"), score=1.0),
    EER=EvalMetric(EERMetric, score=-1.0),
    WER=EvalMetric(WerScore, score=1.0),
)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, p=2, dim=1)
    b = torch.nn.functional.normalize(b, p=2, dim=1)
    return a @ b.transpose(0, 1)


def clap_metrics(
    audio_embds: torch.Tensor,  # size of (B, D)
    text_embds: torch.Tensor,  # size of (B, D)
    num_caps: int = 5,
    topk: int = 10,
    topk_recall: List[int] = [1, 5, 10],
):
    metrics = {}
    n_embds = len(audio_embds)
    logits_per_audio = cos_sim(audio_embds, text_embds).detach()
    logits_per_text = logits_per_audio.t().detach()

    logits = {"a2t": logits_per_audio, "t2a": logits_per_text}

    for name, logit in logits.items():
        # take the first sample when we have num_caps embeddings
        if name == "a2t":
            ground_truth = torch.arange(n_embds, device=audio_embds.device).view(-1, 1)
            logit = logit.view(n_embds // num_caps, num_caps, -1)[:, 0, :].repeat_interleave(num_caps, dim=0)
        else:
            ground_truth = (
                torch.arange(n_embds // num_caps, device=audio_embds.device).repeat_interleave(num_caps).view(-1, 1)
            )
            logit = logit.view(-1, n_embds // num_caps, num_caps)[:, :, 0]

        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        # Calcualte map@10
        ranks = np.sort(preds.view(-1, num_caps).cpu().numpy() + 1, axis=-1)
        if name == "a2t":
            ap = np.arange(1, num_caps + 1) / ranks
        else:
            ap = 1.0 / ranks
        ap[ranks > topk] = 0
        # Per caps
        ap = np.mean(ap, axis=-1)
        mAP10 = 100 * np.mean(ap)

        # Minimum rank for each sample
        if name == "a2t":
            preds = preds.view(-1, num_caps).min(-1)[0]

        preds = preds.detach().cpu().numpy()
        metric_dict = {}
        for k in topk_recall:
            metric_dict[f"r{k}"] = np.mean(preds < k) * 100.0
        metric_dict["meanr"] = preds.mean() + 1
        metric_dict["medr"] = np.floor(np.median(preds)) + 1
        metric_dict["mAP10"] = mAP10
        metrics[name] = metric_dict
    return metrics


def _clapscore(
    *args,
    num_caps: int = 1,
    select: Literal["r1", "r5", "r10", "mAP10"] | None = None,
    average: bool = True,
) -> Dict[str, float] | float:
    ret = clap_metrics(*args, num_caps=num_caps)
    if average:
        average_dict = defaultdict(list)
        for metric, metric_dict in ret.items():
            for k, v in metric_dict.items():
                average_dict[k].append(v)
        ret = {k: np.mean(v) for k, v in average_dict.items()}
    else:
        result = {}
        for metric, metric_dict in ret.items():
            for k, v in metric_dict.items():
                result[f"{metric}_{k}"] = v
        ret = result
    if select:
        ret = ret[select]
    return ret


def weighted_average(scores_dict: Dict[str, List[Tuple[float, int]]]) -> List[float]:
    """
    Compute the weighted average of scores.

    Args:
        scores_dict (Dict[str, List[Tuple[float, int]]]):
            A dictionary where the key is a dataset name and the value is a list of tuples.
            Each tuple contains a score and its corresponding weight.

    Returns:
        List[float]: A list of weighted average scores,
        where each element is the weighted average for scores at the same index across datasets.

    Example:
        scores_dict = {
            'dataset_1': [(0.8, 100), (0.6, 50)],
            'dataset_2': [(0.9, 200), (0.7, 100)]
        }
        result = weighted_average(scores_dict)
        # result should be approximately: [
        #   (0.8 * 100 + 0.9 * 200) / (100 + 200),  # Weighted average for the first score index across datasets
        #   (0.6 * 50 + 0.7 * 100) / (50 + 100)   # Weighted average for the second score index across datasets
        # ]
        # result should be approximately [0.8666666666666667, 0.6666666666666666]
    """

    output = []
    num_score_types = len(list(scores_dict.values())[0]) if scores_dict else 0  # Handle empty scores_dict
    for score_index in range(num_score_types):
        weighted_sum = 0
        total_weight = 0
        for dataset_scores in scores_dict.values():
            if score_index < len(dataset_scores):  # Check if score_index is valid for current dataset
                score, weight = dataset_scores[score_index]
                weighted_sum += score * weight
                total_weight += weight
        weighted_avg = weighted_sum / total_weight if total_weight != 0 else 0
        output.append(weighted_avg)
    return output
