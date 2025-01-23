from xares.task import TaskConfig


def vocalsound_config(**kwargs) -> TaskConfig:
    config_params = {
        "name": "vocalsound",
        "zenodo_id": "14641593",
        "output_dim": 6,
        "label_processor": lambda x: {
            "laughter": 0,
            "sigh": 1,
            "throatclearing": 2,
            "cough": 3,
            "sneeze": 4,
            "sniff": 5,
        }.get(x["label"], -1),
    }
    config_params.update(kwargs)
    config = TaskConfig(**config_params)
    return config
