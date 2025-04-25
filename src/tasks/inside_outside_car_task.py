from xares.task import TaskConfig


def finger_snap_config(encoder) -> TaskConfig:
    class_label_maps = {
        "Inside": 0,
        "Outside": 1,
    }

    return TaskConfig(
        encoder=encoder,
        name="inside_outside_car",
        formal_name="Inside/outside car",
        private=True,
        epochs=20,
        label_processor=lambda x: class_label_maps[x["position"]],
        output_dim=len(class_label_maps),
        train_split="inside_outside_car_train",
        test_split="inside_outside_car_test",
        valid_split="inside_outside_car_test",
        eval_weight=5000,
    )
