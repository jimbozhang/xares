from xares.task import TaskConfig


def fma_genre_config(encoder) -> TaskConfig:
    data_key = "genre"
    class_label_maps = {
        "Hip-Hop": 0,
        "Pop": 1,
        "Folk": 2,
        "Experimental": 3,
        "Rock": 4,
        "International": 5,
        "Electronic": 6,
        "Instrumental": 7,
    }
    return TaskConfig(
        encoder=encoder,
        name="freemusicarchive",
        train_split="fma_small_train",
        valid_split="fma_small_valid",
        test_split="fma_small_test",
        zenodo_id="14725056",
        output_dim=len(class_label_maps),
        label_processor=lambda x: class_label_maps[x[data_key]],
        epochs=3,
        crop_length=10,  # 10s
    )
