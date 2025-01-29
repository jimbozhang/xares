from xares.task import TaskConfig


def asvspoof2015_config(encoder) -> TaskConfig:
    data_key = "binary_spoof"
    class_label_maps = {"human": 0, "spoof": 1}
    return TaskConfig(
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
