from xares.task import TaskConfig


def librispeech_male_female_config(**kwargs) -> TaskConfig:
    config_params = {
        "name": "librispeechmalefemale",
        "zenodo_id": "14641593",
        "output_dim": 2,
        "train_split": "train-clean-100",
        "valid_split": "dev-clean",
        "test_split": "test-clean",
        "label_processor": lambda x: 0 if x["gender"] == "M" else 1,
    }
    config_params.update(kwargs)
    return TaskConfig(**config_params)
