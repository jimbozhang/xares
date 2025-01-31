from xares.task import TaskConfig


def asvspoof2015_config(encoder) -> TaskConfig:
    data_key = "binary_spoof"
    class_label_maps = {"human": 0, "spoof": 1}
    config = TaskConfig(
        name="asvspoof2015",
        encoder=encoder,
        batch_size_train=64,
        learning_rate=1e-3,
        train_split="asvspoof_train",
        valid_split="asvspoof_valid",
        test_split="asvspoof_eval",
        zenodo_id="14718430",
        output_dim=len(class_label_maps),
        label_processor=lambda x: class_label_maps[x[data_key]],
        epochs=5,
    )

    if config.use_mini_dataset:
        config.audio_tar_name_of_split = {
            config.train_split: "asvspoof_train_0000000.tar",
            config.valid_split: "asvspoof_valid_0000000.tar",
            config.test_split: "asvspoof_eval_0000000.tar",
        }

    return config
