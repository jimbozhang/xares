from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Literal, Callable
from ignite.metrics import Accuracy, EpochMetric, MeanSquaredError, MeanAbsoluteError, AveragePrecision, Metric

METRICS_TYPE = Literal["accuracy","EER","mAP","recallatk","MAE", "MSE"]

@dataclass
class EvalMetric:
    metric: Callable
    score: float = 1.  # can also be -1

class CLAPScore(Metric):

    def __init__(self,
                 num_caps: int = 5,# Clotho uses 5 caps
                 select: Literal['r1', 'r5', 'r10', 'mAP10'] | None = None,
                 average:bool = True,
                 *args,
                 **kwargs):
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

    def compute(self) -> Dict[str,float] |  float:
        return _clapscore(torch.cat(self.embds_x, dim=0),
                          torch.cat(self.embds_y, dim=0),
                          num_caps=self.num_caps,
                          average=self.average,
                          select=self.select)


def compute_eer(pred, target, positive_label:int=1):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(target, pred, positive_label)
    fnr = 1 - tpr
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer_1 + eer_2) / 2
    return eer


EERMetric = lambda : EpochMetric(compute_fn = compute_eer)

ALL_METRICS:Dict[METRICS_TYPE, EvalMetric] = dict(
        accuracy = EvalMetric(Accuracy, score = 1.),
        mAP = EvalMetric(AveragePrecision, score=1.),
        MAE = EvalMetric(MeanAbsoluteError, score=-1.),
        MSE = EvalMetric(MeanSquaredError, score=-1.),
        recallatk = EvalMetric(CLAPScore, score=1.),
        EER = EvalMetric(EERMetric, score=-1.),
        )

def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, p=2, dim=1)
    b = torch.nn.functional.normalize(b, p=2, dim=1)
    return a @ b.transpose(0, 1)


def clap_metrics(
    audio_embds: torch.Tensor, # size of (B, D)
    text_embds: torch.Tensor, # size of (B, D)
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
        if name == 'a2t':
            ground_truth = torch.arange(n_embds,
                                        device=audio_embds.device).view(-1, 1)
            logit = logit.view(n_embds // num_caps, num_caps,
                               -1)[:, 0, :].repeat_interleave(num_caps, dim=0)
        else:
            ground_truth = torch.arange(
                n_embds // num_caps,
                device=audio_embds.device).repeat_interleave(num_caps).view(
                    -1, 1)
            logit = logit.view(-1, n_embds // num_caps, num_caps)[:, :, 0]

        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        #Calcualte map@10
        ranks = np.sort(preds.view(-1, num_caps).cpu().numpy() + 1, axis=-1)
        if name == 'a2t':
            ap = np.arange(1, num_caps + 1) / ranks
        else:
            ap = 1. / ranks
        ap[ranks > topk] = 0
        # Per caps
        ap = np.mean(ap, axis=-1)
        mAP10 = 100 * np.mean(ap)

        #Minimum rank for each sample
        if name == 'a2t':
            preds = preds.view(-1, num_caps).min(-1)[0]

        preds = preds.detach().cpu().numpy()
        metric_dict = {}
        for k in topk_recall:
            metric_dict[f"r{k}"] = np.mean(preds < k) * 100.
        metric_dict["meanr"] = preds.mean() + 1
        metric_dict["medr"] = np.floor(np.median(preds)) + 1
        metric_dict["mAP10"] = mAP10
        metrics[name] = metric_dict
    return metrics



def _clapscore(
    *args,
    num_caps: int = 1,
    select: Literal['r1', 'r5', 'r10', 'mAP10'] | None = None,
    average: bool = True,
) -> Dict[str, float] |  float:
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


