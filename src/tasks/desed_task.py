from math import ceil

import torch

from xares.task import TaskConfig


def desed_config(encoder) -> TaskConfig:
    # We train DESED somewhat different from the standard approach:
    # We use smear out all weak labels across the time axis
    # Then just use a frame-level estimator
    class_label_maps = {
        "Cat": 0,
        "Speech": 1,
        "Running_water": 2,
        "Dishes": 3,
        "Electric_shaver_toothbrush": 4,
        "Vacuum_cleaner": 5,
        "Blender": 6,
        "Alarm_bell_ringing": 7,
        "Dog": 8,
        "Frying": 9,
    }

    def sec_to_frames(sec):
        return ceil(sec / (encoder.hop_size_in_ms / 1000))

    def label_transform(samples):
        target = torch.zeros(len(class_label_maps), sec_to_frames(10), dtype=torch.float32)
        for sample in samples:
            start, end, labelnames = sample["onset"], sample["offset"], sample["label"]
            start = sec_to_frames(start)
            end = sec_to_frames(end)
            for label_name in labelnames.split(","):
                target[class_label_maps[label_name], start:end] = 1.0
        return target

    return TaskConfig(
        encoder=encoder,
        name="desed",
        train_split="wds-audio-train",
        test_split="wds-audio-eval",
        valid_split="wds-audio-eval",
        zenodo_id="14808180",
        output_dim=len(class_label_maps),
        label_processor=label_transform,
        criterion="BCEWithLogitsLoss",
        metric="segmentf1",
        metric_args=dict(hop_size_in_ms=encoder.hop_size_in_ms if encoder else 0, segment_length_in_s=1.0),
        task_type="frame",
        do_knn=False,
    )
