from math import ceil

import torch

from xares.task import TaskConfig


def maestro_config(encoder) -> TaskConfig:
    class_label_maps = {
        21: 0,
        22: 1,
        23: 2,
        24: 3,
        25: 4,
        26: 5,
        27: 6,
        28: 7,
        29: 8,
        30: 9,
        31: 10,
        32: 11,
        33: 12,
        34: 13,
        35: 14,
        36: 15,
        37: 16,
        38: 17,
        39: 18,
        40: 19,
        41: 20,
        42: 21,
        43: 22,
        44: 23,
        45: 24,
        46: 25,
        47: 26,
        48: 27,
        49: 28,
        50: 29,
        51: 30,
        52: 31,
        53: 32,
        54: 33,
        55: 34,
        56: 35,
        57: 36,
        58: 37,
        59: 38,
        60: 39,
        61: 40,
        62: 41,
        63: 42,
        64: 43,
        65: 44,
        66: 45,
        67: 46,
        68: 47,
        69: 48,
        70: 49,
        71: 50,
        72: 51,
        73: 52,
        74: 53,
        75: 54,
        76: 55,
        77: 56,
        78: 57,
        79: 58,
        80: 59,
        81: 60,
        82: 61,
        83: 62,
        84: 63,
        85: 64,
        86: 65,
        87: 66,
        88: 67,
        89: 68,
        90: 69,
        91: 70,
        92: 71,
        93: 72,
        94: 73,
        95: 74,
        96: 75,
        97: 76,
        98: 77,
        99: 78,
        100: 79,
        101: 80,
        102: 81,
        103: 82,
        104: 83,
        105: 84,
        106: 85,
        107: 86,
    }

    def sec_to_frames(sec):
        return ceil(sec / (encoder.hop_size_in_ms / 1000))

    def label_processor(sample):
        maximal_length = max(item["end"] for item in sample)
        maximal_length = sec_to_frames(maximal_length)
        target_vector = torch.zeros((len(class_label_maps), maximal_length), dtype=torch.float32)
        for item in sample:
            start, end, note = sec_to_frames(item["start"]), sec_to_frames(item["end"]), item["note"]
            label_idx = class_label_maps[note]
            target_vector[label_idx, start:end] = 1.0
        return target_vector

    config = TaskConfig(
        encoder=encoder,
        name="maestro",
        zenodo_id="14858022",
        train_split="maestro_train",
        valid_split="maestro_valid",
        test_split="maestro_test",
        task_type="frame",
        output_dim=len(class_label_maps),
        criterion="BCEWithLogitsLoss",
        metric="segmentf1",
        metric_args=dict(hop_size_in_ms=encoder.hop_size_in_ms if encoder else 0, segment_length_in_s=0.1),
        epochs=50,
        batch_size_encode=1,  # Long samples
        batch_size_train=1,  # Samples are very long, avoid extreme padding
        label_processor=label_processor,
    )
    return config
