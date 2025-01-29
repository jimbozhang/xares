from xares.task import TaskConfig


def ravdess_config(encoder) -> TaskConfig:
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

    config = TaskConfig(
        name="ravdess",
        encoder=encoder,
        zenodo_id="14722524",
        k_fold_splits=list(range(0, 4)),
        output_dim=len(class_label_maps),
        label_processor=lambda x: class_label_maps[x["emotion"]],
    )

    config.audio_tar_name_of_split = {fold: f"ravdess_fold_{fold}_0000000.tar" for fold in config.k_fold_splits}
    config.encoded_tar_name_of_split = {fold: f"ravdess-wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits}

    return config
