from xares.task import TaskConfig


def cremad_config(encoder) -> TaskConfig:
    return TaskConfig(
        encoder=encoder,
        evalset_size=1116,
        formal_name="CREMA-D",
        label_processor=lambda x: {"H": 0, "S": 1, "A": 2, "F": 3, "D": 4, "N": 5}.get(x["label"], -1),
        name="cremad",
        output_dim=6,
        test_split="wds-audio-test",
        train_split="wds-audio-train",
        valid_split="wds-audio-valid",
        zenodo_id="14646870",
    )
