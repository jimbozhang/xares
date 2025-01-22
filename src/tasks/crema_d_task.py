from pathlib import Path

from xares.task import TaskConfig


def cremad_config(**kwargs) -> TaskConfig:
    config_params = {
        "name": "cremad",
        "zenodo_id": "14646870",
        "output_dim": 6,
        "label_processor": lambda x: {"H": 0, "S": 1, "A": 2, "F": 3, "D": 4, "N": 5}.get(x["label"], -1),
    }
    config_params.update(kwargs)

    config = TaskConfig(**config_params)
    config.env_dir = (Path(config.env_root) / "cremad",)
    return config
