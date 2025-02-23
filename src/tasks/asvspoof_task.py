from loguru import logger

from xares.task import TaskConfig


def asvspoof2015_config(encoder) -> TaskConfig:
    data_key = "binary_spoof"
    class_label_maps = {"human": 0, "spoof": 1}
    config = TaskConfig(
        batch_size_train=64,
        encoder=encoder,
        epochs=5,
        evalset_size=190948,
        formal_name="ASV2015",
        label_processor=lambda x: class_label_maps[x[data_key]],
        learning_rate=1e-3,
        name="asvspoof2015",
        output_dim=len(class_label_maps),
        test_split="asvspoof_eval",
        train_split="asvspoof_train",
        valid_split="asvspoof_valid",
        zenodo_id="14718430",
    )

    if config.use_mini_dataset:
        logger.warning(f"Dataset {config.name} uses mini version for faster evaluation.")
        config.audio_tar_name_of_split = {
            config.train_split: "asvspoof_train_0000000.tar",
            config.valid_split: "asvspoof_valid_0000000.tar",
            config.test_split: "asvspoof_eval_0000000.tar",
        }

    return config
