from xares.task import TaskConfig


def vocalsound_config(encoder) -> TaskConfig:
    return TaskConfig(
        encoder=encoder,
        label_processor=lambda x: {
            "laughter": 0,
            "sigh": 1,
            "throatclearing": 2,
            "cough": 3,
            "sneeze": 4,
            "sniff": 5,
        }.get(x["label"], -1),
        name="vocalsound",
        output_dim=6,
        zenodo_id="14722710",
    )
