from xares.task import TaskConfig


def urbansound8k_config(encoder) -> TaskConfig:
    class_label_maps = {
        "air_conditioner": 0,
        "car_horn": 1,
        "children_playing": 2,
        "dog_bark": 3,
        "drilling": 4,
        "engine_idling": 5,
        "gun_shot": 6,
        "jackhammer": 7,
        "siren": 8,
        "street_music": 9,
    }
    config = TaskConfig(
        encoder=encoder,
        name="urbansound8k",
        zenodo_id="14722683",
        k_fold_splits=list(range(1, 11)),
        output_dim=len(class_label_maps),
        label_processor=lambda x: class_label_maps[x["soundevent"]],
    )
    config.audio_tar_name_of_split = {fold: f"urbansound_fold{fold}_0000000.tar" for fold in config.k_fold_splits}
    config.encoded_tar_name_of_split = {
        fold: f"urbansound-wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits
    }
    return config
