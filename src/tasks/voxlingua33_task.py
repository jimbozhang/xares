from loguru import logger

from xares.task import TaskConfig


def voxlingua33_config(encoder) -> TaskConfig:
    class_label_maps = {
        "uk": 0,
        "zh": 1,
        "nn": 2,
        "hr": 3,
        "nl": 4,
        "fa": 5,
        "el": 6,
        "sv": 7,
        "hy": 8,
        "pl": 9,
        "hu": 10,
        "mk": 11,
        "is": 12,
        "pt": 13,
        "sl": 14,
        "ar": 15,
        "en": 16,
        "lv": 17,
        "ru": 18,
        "az": 19,
        "no": 20,
        "lt": 21,
        "es": 22,
        "tr": 23,
        "fr": 24,
        "de": 25,
        "et": 26,
        "ur": 27,
        "da": 28,
        "fi": 29,
        "it": 30,
        "ja": 31,
        "sr": 32,
    }

    config = TaskConfig(
        name="voxlingua33",
        encoder=encoder,
        batch_size_train=64,
        learning_rate=1e-3,
        train_split="train_subset",
        test_split="dev",
        valid_split="dev",
        zenodo_id="14723799",
        output_dim=len(class_label_maps),
        epochs=10,
        crop_length=10,
        label_processor=lambda x: class_label_maps[x["labels"]],
    )

    if config.use_mini_dataset:
        logger.warning(f"Dataset {config.name} uses mini version for faster evaluation.")
        config.audio_tar_name_of_split[config.train_split] = "train_subset_2k_00000{00..01}.tar"

    return config
