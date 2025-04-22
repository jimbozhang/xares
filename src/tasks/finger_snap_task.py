from xares.task import TaskConfig


def finger_snap_config(encoder) -> TaskConfig:
    class_label_maps = {
        "T": 0,
        "F": 1,
    }

    return TaskConfig(
        encoder=encoder,
        name="finger_snap",
        formal_name="Finger snap sound",
        private=True,
        disabled=True,
        epochs=20,
        label_processor=lambda x: class_label_maps[x["tag"]],
        output_dim=len(class_label_maps),
        train_split="finger_snap_train",
        test_split="finger_snap_test",
        valid_split="finger_snap_test",
        eval_weight=5000,
    )
