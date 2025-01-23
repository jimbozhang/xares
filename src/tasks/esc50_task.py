from xares.task import TaskConfig


def esc50_config(**kwargs) -> TaskConfig:
    config = TaskConfig(
        name="esc50",
        zenodo_id="14614287",
        k_fold_splits=list(range(1, 6)),
        output_dim=50,
        label_processor=lambda x: x["label"],
        **kwargs,
    )

    config.audio_tar_name_of_split = {fold: f"wds-audio-fold-{fold}-*.tar" for fold in config.k_fold_splits}
    config.encoded_tar_name_of_split = {fold: f"wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits}

    return config
