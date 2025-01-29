from xares.task import TaskConfig


def nysnthinstument_config(encoder) -> TaskConfig:
    data_key = "instrument_family_str"
    class_label_maps = {
        "guitar": 0,
        "bass": 1,
        "organ": 2,
        "keyboard": 3,
        "vocal": 4,
        "string": 5,
        "reed": 6,
        "flute": 7,
        "mallet": 8,
        "brass": 9,
        "synth_lead": 10,
    }
    return TaskConfig(
        name="nsynthinstument",
        encoder=encoder,
        train_split="train",
        test_split="test",
        valid_split="valid",
        zenodo_id="14725174",
        output_dim=len(class_label_maps),
        label_processor=lambda x: class_label_maps[x[data_key]],
    )
