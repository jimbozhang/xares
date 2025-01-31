from xares.task import TaskConfig


def nysnthinstument_config(encoder) -> TaskConfig:
    data_key = "instrument_family_str"
    class_label_maps = {
        "guitar": 0,
        "bass": 1,
        "organ": 2,
        "keyboard": 3,
        "vocal": 4,
        "string": 5,
        "reed": 6,
        "flute": 7,
        "mallet": 8,
        "brass": 9,
        "synth_lead": 10,
    }
    config = TaskConfig(
        name="nsynthinstument",
        encoder=encoder,
        train_split="nsynth_train",
        test_split="nsynth_test",
        valid_split="nsynth_valid",
        zenodo_id="14725174",
        output_dim=len(class_label_maps),
        label_processor=lambda x: class_label_maps[x[data_key]],
    )

    if config.use_mini_dataset:
        config.audio_tar_name_of_split.update(
            {
                config.train_split: "nsynth_train_sub55k_0000000.tar",
                config.valid_split: "nsynth_valid_0000001.tar",
            }
        )

    return config
