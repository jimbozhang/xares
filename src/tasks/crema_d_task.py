from pathlib import Path

from xares.task import TaskConfig


def cremad_config(encoder) -> TaskConfig:
    config_params = {
        "encoder": encoder,
        "name": "cremad",
        "train_split": "wds-audio-train",
        "valid_split": "wds-audio-valid",
        "test_split": "wds-audio-test",
        "zenodo_id": "14646870",
        "output_dim": 6,
        "label_processor": lambda x: {"H": 0, "S": 1, "A": 2, "F": 3, "D": 4, "N": 5}.get(x["label"], -1),
    }

    config = TaskConfig(**config_params)
    return config
