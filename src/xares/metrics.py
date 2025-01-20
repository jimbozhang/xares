from dataclasses import dataclass
from typing import Dict, Literal, Callable
from ignite.metrics import Accuracy, MeanSquaredError, MeanAbsoluteError, AveragePrecision

METRICS_TYPE = Literal["accuracy","EER","mAP","recall@k","MAE", "MSE"]

@dataclass
class EvalMetric:
    metric: Callable
    score: float = 1.  # can also be -1

ALL_METRICS:Dict[METRICS_TYPE, EvalMetric] = dict(
        accuracy = EvalMetric(Accuracy, score = 1.),
        mAP = EvalMetric(AveragePrecision, score=1.),
        MAE = EvalMetric(MeanAbsoluteError, score=-1.),
        MSE = EvalMetric(MeanSquaredError, score=-1.),
        EER = EvalMetric(lambda x:x , score=-1.),
        )
