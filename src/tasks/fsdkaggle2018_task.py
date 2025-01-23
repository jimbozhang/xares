from xares.task import TaskConfig


def fsdkaggle2018_config(encoder) -> TaskConfig:
    data_key = "sound"
    class_label_maps = {
        "Hi-hat": 0,
        "Saxophone": 1,
        "Trumpet": 2,
        "Glockenspiel": 3,
        "Cello": 4,
        "Knock": 5,
        "Gunshot_or_gunfire": 6,
        "Clarinet": 7,
        "Computer_keyboard": 8,
        "Keys_jangling": 9,
        "Snare_drum": 10,
        "Writing": 11,
        "Laughter": 12,
        "Tearing": 13,
        "Fart": 14,
        "Oboe": 15,
        "Flute": 16,
        "Cough": 17,
        "Telephone": 18,
        "Bark": 19,
        "Chime": 20,
        "Bass_drum": 21,
        "Bus": 22,
        "Squeak": 23,
        "Scissors": 24,
        "Harmonica": 25,
        "Gong": 26,
        "Microwave_oven": 27,
        "Burping_or_eructation": 28,
        "Double_bass": 29,
        "Shatter": 30,
        "Fireworks": 31,
        "Tambourine": 32,
        "Cowbell": 33,
        "Electric_piano": 34,
        "Meow": 35,
        "Drawer_open_or_close": 36,
        "Applause": 37,
        "Acoustic_guitar": 38,
        "Violin_or_fiddle": 39,
        "Finger_snapping": 40,
    }

    config_params = {
        "name": "fsdkaggle2018",
        "encoder":encoder,
        "batch_size_train": 64,
        "learning_rate": 1e-3,
        "train_split": "fsd18_train",
        "test_split": "fsd18_test",
        "valid_split": "fsd18_test",
        "zenodo_id": "TODO",
        "output_dim": len(class_label_maps),
        "label_processor": lambda x: class_label_maps[x[data_key]],
        "epochs": 20,
    }


    config = TaskConfig(**config_params)

    return config
