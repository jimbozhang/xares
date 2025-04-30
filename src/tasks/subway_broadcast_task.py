from xares.task import TaskConfig


def subway_broadcast_config(encoder) -> TaskConfig:
    return TaskConfig(
        encoder=encoder,
        name="subway_broadcast",
        formal_name="Subway broadcast",
        private=True,
        epochs=20,
        label_processor=lambda x: int(x["broadcasting"]),
        output_dim=2,
        train_split="subway_broadcast",
        test_split="subway_broadcast",
        valid_split="subway_broadcast",
        eval_weight=5000,
    )
