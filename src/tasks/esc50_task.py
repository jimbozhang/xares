from xares.task import TaskConfig


def esc50_config(encoder) -> TaskConfig:
    config = TaskConfig(
        encoder=encoder,
        name="esc50",
        zenodo_id="14614287",
        k_fold_splits=list(range(1, 6)),
        output_dim=50,
        label_processor=lambda x: x["label"],
    )

    config.audio_tar_name_of_split = {fold: f"wds-audio-fold-{fold}-*.tar" for fold in config.k_fold_splits}
    config.encoded_tar_name_of_split = {fold: f"wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits}

    return config
