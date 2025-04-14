from xares.task import TaskConfig


def esc50_config(encoder) -> TaskConfig:
    config = TaskConfig(
        encoder=encoder,
        eval_weight=400,
        formal_name="ESC-50",
        k_fold_splits=list(range(1, 6)),
        label_processor=lambda x: x["label"],
        name="esc50",
        output_dim=50,
        zenodo_id="14614287",
    )

    config.audio_tar_name_of_split = {fold: f"wds-audio-fold-{fold}-*.tar" for fold in config.k_fold_splits}
    config.encoded_tar_name_of_split = {fold: f"wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits}

    return config
