from loguru import logger

from xares.task import TaskConfig


def librispeech_male_female_config(encoder) -> TaskConfig:
    config = TaskConfig(
        encoder=encoder,
        evalset_size=2620,
        formal_name="LibriSpeech-MF",
        label_processor=lambda x: 0 if x["gender"] == "M" else 1,
        name="librispeechmalefemale",
        output_dim=2,
        test_split="test-clean",
        train_split="train-clean-100",
        valid_split="dev-clean",
        zenodo_id="14716252",
    )

    if config.use_mini_dataset:
        logger.warning(f"Dataset {config.name} uses mini version for faster evaluation.")
        config.audio_tar_name_of_split[config.train_split] = "train-clean-100-000000.tar"

    return config
