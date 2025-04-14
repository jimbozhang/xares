from xares.task import TaskConfig


def libricount_config(encoder) -> TaskConfig:
    task = "num_speakers"
    class_label_maps = {i: i for i in range(11)}

    config = TaskConfig(
        batch_size_train=64,
        encoder=encoder,
        eval_weight=1144,
        formal_name="LibriCount",
        k_fold_splits=list(range(0, 5)),
        label_processor=lambda x: class_label_maps[x[task]],
        learning_rate=1e-3,
        name="libricount",
        output_dim=len(class_label_maps),
        zenodo_id="14722478",
    )

    config.audio_tar_name_of_split = {fold: f"libricount_fold{fold:02d}.tar" for fold in config.k_fold_splits}
    config.encoded_tar_name_of_split = {
        fold: f"libricount-wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits
    }

    return config
