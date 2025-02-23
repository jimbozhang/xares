from xares.task import TaskConfig


def gtzan_genre_config(encoder) -> TaskConfig:
    class_label_maps = {
        "blues": 0,
        "classical": 1,
        "country": 2,
        "disco": 3,
        "hiphop": 4,
        "jazz": 5,
        "metal": 6,
        "pop": 7,
        "reggae": 8,
        "rock": 9,
    }
    config = TaskConfig(
        encoder=encoder,
        evalset_size=100,
        formal_name="GTZAN Genre",
        k_fold_splits=list(range(0, 10)),
        label_processor=lambda x: class_label_maps[x["genre"]],
        name="gtzan_genre",
        output_dim=len(class_label_maps),
        zenodo_id="14722472",
    )

    config.audio_tar_name_of_split = {fold: f"gtzan_fold_{fold}_0000000.tar" for fold in config.k_fold_splits}
    config.encoded_tar_name_of_split = {fold: f"gtzan-wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits}

    return config
