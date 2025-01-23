from xares.task import TaskConfig


def ravdess_config(**kwargs) -> TaskConfig:
    class_label_maps = {
        "neutral": 0,
        "calm": 1,
        "happy": 2,
        "sad": 3,
        "angry": 4,
        "fearful": 5,
        "disgust": 6,
        "surprised": 7,
    }

    config_params = {
        "name": "ravdess",
        "zenodo_id": "TODO",
        "k_fold_splits": list(range(0, 4)),
        "output_dim": len(class_label_maps),
        "label_processor": lambda x: class_label_maps[x["emotion"]],
    }

    config_params.update(kwargs)

    config = TaskConfig(**config_params)

    config.audio_tar_name_of_split = {fold: f"ravdess_fold_{fold}_0000000.tar" for fold in config.k_fold_splits}
    config.encoded_tar_name_of_split = {fold: f"ravdess-wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits}

    return config
