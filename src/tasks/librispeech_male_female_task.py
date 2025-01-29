from xares.task import TaskConfig


def librispeech_male_female_config(encoder) -> TaskConfig:
    return TaskConfig(
        name="librispeechmalefemale",
        encoder=encoder,
        zenodo_id="14716252",
        output_dim=2,
        train_split="train-clean-100",
        valid_split="dev-clean",
        test_split="test-clean",
        label_processor=lambda x: 0 if x["gender"] == "M" else 1,
    )
