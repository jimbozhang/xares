from xares.task import TaskConfig


def voxlingua33_config(**kwargs) -> TaskConfig:
    class_label_maps = {
        "uk": 0,
        "zh": 1,
        "nn": 2,
        "hr": 3,
        "nl": 4,
        "fa": 5,
        "el": 6,
        "sv": 7,
        "hy": 8,
        "pl": 9,
        "hu": 10,
        "mk": 11,
        "is": 12,
        "pt": 13,
        "sl": 14,
        "ar": 15,
        "en": 16,
        "lv": 17,
        "ru": 18,
        "az": 19,
        "no": 20,
        "lt": 21,
        "es": 22,
        "tr": 23,
        "fr": 24,
        "de": 25,
        "et": 26,
        "ur": 27,
        "da": 28,
        "fi": 29,
        "it": 30,
        "ja": 31,
        "sr": 32,
    }

    config_params = {
        "name": "voxlingua33",
        "batch_size_train": 64,
        "learning_rate": 1e-3,
        "train_split": "train_subset",
        "test_split": "dev",
        "valid_split": "dev",
        "zenodo_id": "TODO",
        "output_dim": len(class_label_maps),
        "epochs": 50,
    }

    config_params.update(kwargs)

    task_config = TaskConfig(**config_params)

    return task_config
