from xares.task import TaskConfig


def speechcommandsv1_config(encoder) -> TaskConfig:
    class_label_maps = {
        "sheila": 0,
        "up": 1,
        "bird": 2,
        "right": 3,
        "left": 4,
        "six": 5,
        "yes": 6,
        "on": 7,
        "one": 8,
        "off": 9,
        "zero": 10,
        "marvin": 11,
        "seven": 12,
        "wow": 13,
        "five": 14,
        "down": 15,
        "three": 16,
        "nine": 17,
        "no": 18,
        "happy": 19,
        "cat": 20,
        "go": 21,
        "bed": 22,
        "house": 23,
        "stop": 24,
        "four": 25,
        "tree": 26,
        "dog": 27,
        "two": 28,
        "eight": 29,
    }
    data_key = "labels"

    return TaskConfig(
        batch_size_train=64,
        encoder=encoder,
        evalset_size=2000,
        formal_name="Speech Commands V1",
        label_processor=lambda x: class_label_maps[x[data_key]],
        learning_rate=1e-3,
        name="speechcommandsv1",
        output_dim=len(class_label_maps),
        test_split="wds-audio-test",
        train_split="wds-audio-train",
        valid_split="wds-audio-valid",
        zenodo_id="14722647",
    )
