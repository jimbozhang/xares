from loguru import logger

from xares.task import TaskConfig


def librispeech_male_female_config(encoder) -> TaskConfig:
    config = TaskConfig(
        name="librispeechmalefemale",
        encoder=encoder,
        zenodo_id="14716252",
        output_dim=2,
        train_split="train-clean-100",
        valid_split="dev-clean",
        test_split="test-clean",
        label_processor=lambda x: 0 if x["gender"] == "M" else 1,
    )

    if config.use_mini_dataset:
        logger.warning(f"Dataset {config.name} uses mini version for faster evaluation.")
        config.audio_tar_name_of_split[config.train_split] = "train-clean-100-000000.tar"

    return config
